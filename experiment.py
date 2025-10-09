# pylint: disable=unused-import,abstract-method,unused-argument

from markupsafe import Markup
import psynet.experiment
from psynet.consent import MainConsent
from psynet.modular_page import (
    Prompt, ModularPage, PushButtonControl, Control,
)
from psynet.page import InfoPage, SuccessfulEndPage
from psynet.timeline import FailedValidation, Timeline, PageMaker, join

from psynet.trial.main import Trial
from psynet.trial.create_and_rate import (
    CreateAndRateNodeMixin,
    CreateAndRateTrialMakerMixin,
    CreateTrialMixin,
    SelectTrialMixin,
)
from psynet.trial.imitation_chain import (
    ImitationChainTrial,
    ImitationChainTrialMaker,
)
from psynet.trial.chain import ChainNode
from psynet.utils import get_logger

import numpy as np
from scipy.special import softmax
import random
from typing import List
import json

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import torch.nn.functional as F

logger = get_logger()


def int_to_hex(n):
    """Map an integer to a unique hex color."""
    hash_val = (n * 2654435761) % (256 ** 3)
    return f"#{hash_val:06x}"


def image_similarity_distribution(grids):
    """
    Compute probability distribution over proposed images based on similarity to target.

    Args:
        target_image_path: Path to the target image
        proposed_arrays: List of binary numpy arrays (NxN) where 0=white, 1=black

    Returns:
        torch.Tensor: Probability distribution over proposed images
    """
    # Load model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load target image
    target_image = Image.open("static/truth/car.webp")

    # Convert binary arrays to PIL Images (grayscale)
    # Binary (0=white, 1=black) -> (255, 0) grayscale
    proposed_images = [
        Image.fromarray(((1 - np.array(arr)) * 255).astype(np.uint8), mode='L')
        for arr in grids
    ]

    # Process all images
    with torch.no_grad():
        # Get target embedding
        target_inputs = processor(images=target_image, return_tensors="pt")
        target_embeds = model.get_image_features(**target_inputs)
        target_embeds = F.normalize(target_embeds, p=2, dim=-1)

        # Get proposed embeddings
        proposed_inputs = processor(images=proposed_images, return_tensors="pt")
        proposed_embeds = model.get_image_features(**proposed_inputs)
        proposed_embeds = F.normalize(proposed_embeds, p=2, dim=-1)

    # Compute similarities
    similarities = torch.matmul(target_embeds, proposed_embeds.T)  # (1, n)

    # Scale by temperature and get probabilities
    temperature = model.logit_scale.exp()
    logits = similarities * temperature
    probs = logits.softmax(dim=1).squeeze()

    logger.info(logits)
    logger.info(probs)

    return logits.detach().numpy()[0], probs.detach().numpy()

GRID_SIZE = 16
GRID_FILL = int(GRID_SIZE * GRID_SIZE * 0.5)
MAX_ACCURACY = GRID_SIZE * GRID_SIZE
NUM_EDITS = 32

N_TRIALS_PER_PARTICIPANT = 1
N_CREATORS_PER_GENERATION = 3
N_RATERS = 3
N_GRIDS = 1
N_GENERATIONS = 20
RECRUITER = "hotair"
DURATION_ESTIMATE = 120 + N_TRIALS_PER_PARTICIPANT * 30


# assert N_TRIALS_PER_PARTICIPANT % (N_CREATORS_PER_GENERATION + 1) == 0


# Utility function for grid HTML generation
def grid_to_html(
        grid_data: List[List[int]], cell_size="20px", border_size="1px",
):
    """Generate HTML representation of grid without creating a node"""
    html = f'<div style="display: inline-block; border: 1px solid #333;">'
    for row in grid_data:
        html += '<div style="display: flex; line-height: 0;">'
        for cell in row:
            color = "#000" if cell == 1 else "#fff"
            html += f'<div style="background-color: {color}; width: {cell_size}; height: {cell_size}; border: {border_size} solid #ccc;"></div>'
        html += '</div>'
    html += '</div>'
    return html


class ArtefactNode(ChainNode, CreateAndRateNodeMixin):
    def __init__(
            self,
            artefact=None,
            # Content of artefact (e.g. List[List[int]] for a grid, Str for a story,...
            **kwargs,
    ):
        if artefact is not None:
            kwargs['context'] = {
                "original": artefact, "artefact_type": self.get_artefact_type(),
            }

        super().__init__(**kwargs)
        self.artefact = artefact

    def get_artefact_type(self):
        """Override in subclasses to specify artefact type"""
        raise NotImplementedError(
            "Subclasses must implement get_artefact_type()",
        )

    def create_initial_seed(self, experiment=None, participant=None):
        return {
            "generation": 0, "last_winner": "", "accuracy": 0, "options": dict(),
        }

    def create_definition_from_seed(self, seed, experiment, participant):
        return seed


class GridNode(ArtefactNode):
    """Node class specifically for grid artefacts"""

    def __init__(
            self,
            grid_data: List[List[int]] = None,  # grid content
            size: int = 10,  # grid size
            n_fill: int = 24,
            random: bool = False,  # random initialization
            **kwargs,
    ):

        if grid_data is None and random is True:
            grid_data = self.random(size, n_fill)

        super().__init__(artefact=grid_data, **kwargs)

        if self.artefact is None:
            self.artefact = self.definition["last_winner"]

        # Validate grid during initialization
        self._validate_grid_format(self.artefact)

    @property
    def size(self):
        """Get grid size"""
        return len(self.artefact) if self.artefact else 0

    def get_artefact_type(self):
        return "grid"

    def random(self, size: int, n_fill: int):
        """Generate random grid with exactly 24% cells == 1 and 76% == 0"""
        total_cells = size * size
        num_ones = n_fill
        num_zeros = total_cells - num_ones

        flat_grid = [1] * num_ones + [0] * num_zeros
        np.random.shuffle(flat_grid)
        grid = np.array(flat_grid).reshape(size, size)

        return grid.tolist()

    def _validate_grid_format(self, grid_data):
        """Validate grid format with assertions"""
        assert isinstance(grid_data, list), "Grid must be a list"
        assert len(grid_data) > 0, "Grid cannot be empty"
        assert all(
            isinstance(row, list) for row in grid_data
        ), "All rows must be lists"
        assert len(
            set(len(row) for row in grid_data)
        ) == 1, "All rows must have same length"
        assert len(grid_data) == len(grid_data[0]), "Grid must be square (NxN)"
        assert all(
            all(cell in [0, 1] for cell in row) for row in grid_data
        ), "All cells must be 0 (white) or 1 (black)"

    def summarize_trials(self, trials, experiment, participant):
        """Summarize trials - behavior depends on trial maker type"""
        logger.info("Summarizing!!")
        # Create-and-rate mode: use selection logic
        logger.info(f"GridNode.summarize_trials: Create-and-rate mode")
        trial_maker = self.trial_maker

        # Get all rate trials for this node, for the last generation
        all_rate_trials = trial_maker.rater_class.query.filter_by(
            node_id=self.id, failed=False, finalized=True,
        ).all()

        last_generation = max(
            [trial.definition["generation"] for trial in all_rate_trials],
        )
        all_rate_trials = [trial for trial in all_rate_trials if
                           trial.definition[
                               "generation"] == last_generation]

        assert len(all_rate_trials) == N_RATERS

        # Get all targets of the rate trials
        all_targets = all_rate_trials[0].get_all_targets()
        count_dict = {i: 0 for i in range(len(all_targets))}

        # For each rate trial, keep track of the selected target
        for trial in all_rate_trials:
            if not isinstance(trial, SelectTrialMixin):
                continue

            # Find which target matches this trial's answer
            for i, target in enumerate(all_targets):
                if trial.answer == str(target):
                    count_dict[i] += 1
                    break

        # Get the index of the target with highest count
        winning_index = max(count_dict, key=count_dict.get)
        last_winner = all_targets[winning_index].answer['edit']

        # Seed for next generation
        definition = self.seed.copy()
        definition["generation"] += 1
        definition["last_winner"] = last_winner
        definition["winner_id"] = all_targets[winning_index].id
        definition["options"] = {
            all_targets[i].id: count_dict[i]
            for i in count_dict.keys()
        }
        definition["accuracy"] = all_targets[winning_index].var.accuracy

        logger.info(definition)

        return definition

    @classmethod
    def accuracy(cls, truth, attempt):
        truth = np.array(truth, dtype=int) * 1
        attempt = np.array(attempt, dtype=int) * 1
        return (truth == attempt).sum()

    def __repr__(self):
        return f"GridNode({self.size}x{self.size})"


class GridReproductionControl(Control):
    macro = "grid_reproduction_control"
    external_template = "grid-reproduction.html"

    def __init__(
            self, grid_size=10, prefill_grid=None, num_edits: int = NUM_EDITS,
            **kwargs,
    ):
        self.grid_size = grid_size
        self.prefill_grid = prefill_grid
        self.num_edits = num_edits
        super().__init__(**kwargs, bot_response=self.bot_response())

    def bot_response(self):
        if self.prefill_grid:
            grid = np.array(self.prefill_grid, dtype=int)
        else:
            grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        return grid.tolist()

    @property
    def metadata(self):
        return {
            "grid_size": self.grid_size,
            "prefill_grid": self.prefill_grid,
            "num_edits": self.num_edits,
        }


class GridInputPage(ModularPage):
    def __init__(
            self, label: str, prompt: str, time_estimate: float,
            prefill_grid=None,
            grid_size: int = 10,
    ):
        super().__init__(
            label,
            Prompt(prompt),
            control=GridReproductionControl(
                grid_size=grid_size,
                prefill_grid=prefill_grid,
                num_edits=NUM_EDITS,
            ),
            time_estimate=time_estimate,
        )
        self.prefill_grid = prefill_grid
        self.num_edits = NUM_EDITS
        logger.info(self.prefill_grid)

    def format_answer(self, raw_answer, **kwargs):
        if not isinstance(raw_answer, list):
            return "INVALID_RESPONSE"

        grid_data = raw_answer

        return self._validate_grid(grid_data)

    def _validate_grid(self, grid_data):
        # Ensure grid format consistency
        if not isinstance(grid_data, list):
            return [[0] * 8 for _ in range(8)]  # Default 8x8 white grid

        # Validate each row and cell
        validated_grid = []
        for row in grid_data:
            if isinstance(row, list):
                validated_row = [1 if cell == 1 else 0 for cell in row]
                validated_grid.append(validated_row)
            else:
                validated_grid.append([0] * len(grid_data))

        return validated_grid

    def validate(self, response, **kwargs):
        if response.answer == "INVALID_RESPONSE":
            return FailedValidation("Please enter a valid response.")
        if not isinstance(response.answer, list):
            return FailedValidation("Invalid response format.")

        grid_data = response.answer
        if len(grid_data) == 0:
            return FailedValidation("Please create a grid pattern.")

        if np.sum(
                np.abs(np.array(grid_data) - np.array(self.prefill_grid)),
        ) > self.num_edits:
            return FailedValidation(
                f"Please make fewer than {self.num_edits + 1} edits.",
            )

        return None


class GridDisplayPage(InfoPage):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            time_estimate=10,
        )


class GridCreateTrial(CreateTrialMixin, ImitationChainTrial):
    """Trial class specifically for grid creation tasks"""
    time_estimate = 30
    accumulate_answers = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var.accuracy = None

    def first_trial(self):
        """First trial - show original grid only with greenish background"""
        return InfoPage(
            "You will be asked to create a new grid from scratch.",
            time_estimate=10,
        )

    def other_trial(self):
        active_generation = self.definition["generation"]

        prior_proposals = self.query.filter_by(
            network_id=self.network_id, failed=False, finalized=True,
        ).all()

        proposals = dict()
        for proposal in prior_proposals:
            generation = proposal.node.definition["generation"]

            if generation >= active_generation:
                continue

            proposals[generation] = proposals.get(generation, []) + [
                proposal,
            ]

        selections = self.trial_maker.rater_class.query.filter_by(
            network_id=self.network_id, failed=False, finalized=True,
        ).all()
        winners = []
        for selection in selections:
            winners.append(selection.answer)

        html = "<div style='display: flex; flex-direction: row; justify-content: center; align-items: flex-start; gap: 10px;'>"

        for generation in sorted(proposals.keys()):
            # One column for each generation
            html += "<div style='display: flex; flex-direction: column; align-items: center; gap: 5px;'>"

            # Generation label
            html += f"<div style='margin-bottom: 2px;'>Step {generation + 1}</div>"

            # Proposals within this generation
            for i, proposal in enumerate(proposals[generation]):
                # if generation == sorted(proposals.keys())[-1]:
                html += "<div style='display: flex; flex-direction: row; align-items: center; gap: 5px;'>"

                # Display grids
                html += f"<div>{grid_to_html(proposal.answer['edit'], cell_size='5px', border_size='0px')}</div>"

                # Add colored circles for each selection
                html += "<div style='display: flex; flex-direction: column; gap: 2px; min-width: 15px;'>"
                for selection in selections:
                    if selection.answer == str(proposals[generation][i]):
                        # Add a colored circle (you can customize the color)
                        html += f"<div style='width: 10px; height: 10px; border: 1px solid black; border-radius: 50%; background-color: {int_to_hex(selection.participant_id)};'></div>"

                html += "</div>"

                # Close the row container (only for last generation)
                if generation == sorted(proposals.keys())[-1]:
                    row_label = chr(65 + i)
                    html += f"<div><strong>{row_label}</strong></div>"

                html += "</div>"

            html += "</div>"

        html += "</div>"

        choices = [proposal.id for proposal in proposals[active_generation - 1]]

        return ModularPage(
            label="copy",
            prompt=Markup(
                f"<h3>Choose a proposal to start from <i>(creation mode)</i></h3>"
                f"<p>Below, you may observe prior proposals and how many times they were selected by other participants at each step.</p>"
                f"<p>Choose a proposal to start from. On the next page, you will start from this proposal and change up to {NUM_EDITS} pixels.</p>"
                "<p>Remember that your goal is to have your creations selected as many times as possible!</p>"
                f"{html}",
            ),
            control=PushButtonControl(
                choices=choices,
                labels=[f"Start from {chr(65 + i)}" for i in
                        range(len(choices))],
                arrange_vertically=True,
            ),
            time_estimate=10,
            save_answer="copy",
        )

    def copy_page(self):
        """Show appropriate trial based on generation"""
        generation = self.definition["generation"]

        if generation == 0:
            return [self.first_trial()]
        else:
            return [
                self.other_trial(),
            ]

    def edit_page(self, participant):
        """Input page for grid creation"""
        generation = self.definition["generation"]
        grid_size = len(self.context["original"])

        # For generation 0, start from scratch (no prefill); for generation > 0, start with participant pick
        if generation == 0:
            prefill_grid = np.zeros(
                (GRID_SIZE, GRID_SIZE),
            ).tolist()  # Empty for first generation
        else:
            copy_id = participant.var.get("copy")
            copy_proposal = self.trial_maker.creator_class.query.filter_by(
                id=copy_id,
            ).one()
            prefill_grid = copy_proposal.answer['edit']

        text = Markup(
            f"<h3>Make a proposal <i>(creation mode)</i></h3>"
            f"<p>Please edit up to {NUM_EDITS} squares in the grid. You will receive a reward every time your proposal is selected.</p>",
        ) if generation == 0 else Markup(
            f"<h3>Edit the proposal <i>(creation mode)</i></h3>"
            f"<p>You are free to make up to {NUM_EDITS} modifications. You will receive a reward if your proposal is selected.</p>",
        )

        return GridInputPage(
            "edit",
            text,
            time_estimate=20,
            prefill_grid=prefill_grid,
            grid_size=grid_size,
        )

    def show_trial(self, experiment, participant):
        """Show the complete grid creation trial"""
        return join(
            self.copy_page(),
            PageMaker(
                lambda participant: self.edit_page(participant),
                time_estimate=10,
            ),
        )

    def score_answer(self, answer, definition):
        self.var.accuracy = int(
            GridNode.accuracy(self.context["original"], self.answer['edit']),
        )
        return self.var.accuracy


class GridSelectTrial(SelectTrialMixin, ImitationChainTrial):
    """Trial class specifically for grid selection tasks"""
    time_estimate = 45

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var.success = None
        self.var.accuracy = None

    def bot_response(self, artefacts):
        score = dict()
        for i, artefact in enumerate(artefacts):
            score[i] = np.sum(self.get_target_answer(artefact)['edit'])

        return str(artefacts[max(score.keys(), key=score.get)])

    def show_trial(self, experiment, participant):
        artefacts = self.get_all_targets()

        assert len(
            {artefact.node_id for artefact in artefacts},
        ) == 1, "All artefacts must be from the same node"
        assert len(
            artefacts,
        ) == N_CREATORS_PER_GENERATION, "There must be exactly N_CREATORS_PER_GENERATION artefacts"

        return ModularPage(
            "choice",
            Prompt(
                Markup(
                    "<h3>Choose the best image <i>(selection mode)</i></h3>"
                    f"<p>{N_CREATORS_PER_GENERATION} participants have produced images.</p>"
                    "<p>Based on your prior observations, pick the image that you think will give you the highest rewards.</p>"
                    "<div style='display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin: 20px 0;'>"
                    + "\n".join(
                        [
                            f"<div style='text-align: center;'><strong>Version {i + 1}:</strong><br>{grid_to_html(self.get_target_answer(artefact)['edit'])}</div>"
                            for i, artefact in enumerate(artefacts)
                        ],
                    )
                    + "</div>",
                ),
            ),
            control=PushButtonControl(
                choices=artefacts,
                labels=[f"Version {i + 1}" for i in range(len(artefacts))],
                arrange_vertically=False,
                bot_response=lambda: self.bot_response(artefacts),
            ),
            time_estimate=30,
        )

    def format_answer(self, answer, **kwargs):
        answer = answer["choice"]
        rated_target_strs = [f"{target}" for target in self.targets]
        logger.info(rated_target_strs)
        assert (
                answer in rated_target_strs
        ), "The answer must be one of the rated target_strs"
        return answer

    def score_answer(self, answer, definition):
        super().score_answer(answer, definition)

        artefacts = self.get_all_targets()

        fitness = np.zeros(len(artefacts))
        pick = None
        for i, artefact in enumerate(artefacts):
            # fitness[i] = GridNode.accuracy(
            #     self.get_target_answer(artefact)["edit"],
            #     self.context['original'],
            # )
            if str(artefact) == str(self.answer):
                pick = i

        # fitness = 32 * fitness / (GRID_SIZE*GRID_SIZE)
        # p = softmax(fitness)
        fitness, p = image_similarity_distribution([self.get_target_answer(artefact)["edit"] for artefact in artefacts])
        logger.info(fitness)

        best = np.max(fitness)
        self.var.success = bool(fitness[pick] == best)
        self.var.accuracy = int(fitness[pick])
        return float(p[pick])

    def show_feedback(self, experiment, participant):
        artefacts = self.get_all_targets()

        # Sort artefacts so picked one comes first
        artefacts_sorted = sorted(
            artefacts, key=lambda x: str(x) != self.answer
            )

        progress_value = int(self.score * 100)
        uncertainty = 100 - progress_value

        html = f"Your choice scored {self.score * 100:.0f} points out of 100 points (the total score of all images combined). "

        html += f"""<div style="display: flex; flex-direction: column; gap: 15px; max-width: 1400px; font-family: Arial, sans-serif;">
        """

        # Generate each row with grid and bar together
        for i, artefact in enumerate(artefacts_sorted):
            is_picked = str(artefact) == self.answer

            border_color = "red" if is_picked else "white"

            html += f"""
            <div style="display: flex; gap: 40px; align-items: center;">
                <div style="border: 3px solid {border_color}; overflow: hidden; background-color: white;">
                    {grid_to_html(self.get_target_answer(artefact)['edit'], cell_size="5px", border_size="0px")}
                </div>

                <div style="flex: 1; display: flex; align-items: center; gap: 15px;">
            """

            if is_picked:
                # Show bar with given progress_value
                html += f"""
                    <div style="height: 40px; border: 3px solid #2563eb; border-radius: 8px; position: relative; background-color: white; overflow: visible; width: 300px;">
                        <div style="position: absolute; left: {progress_value}%; top: -25px; transform: translateX(-50%); width: 0; height: 0; border-left: 12px solid transparent; border-right: 12px solid transparent; border-top: 20px solid #2563eb;"></div>
                        <div style="position: absolute; left: {progress_value}%; top: 0; bottom: 0; width: 3px; background-color: #2563eb; transform: translateX(-50%);"></div>
                    </div>
                    <div style="font-weight: bold; font-size: 18px; color: #2563eb; white-space: nowrap;">{progress_value} points</div>
                """
            else:
                # Show bar with uncertainty between 0 and 100-progress_value
                html += f"""
                    <div style="height: 40px; border: 3px solid #2563eb; border-radius: 8px; position: relative; background-color: white; overflow: hidden; width: 300px;">
                        <div style="height: 100%; background-color: #3b82f6; width: {uncertainty}%;"></div>
                        <div style="position: absolute; left: {uncertainty / 2}%; top: 50%; transform: translate(-50%, -50%); font-size: 28px; color: #2563eb; font-weight: bold;">?</div>
                    </div>
                    <div style="font-weight: bold; font-size: 18px; color: #1e40af; white-space: nowrap;">Between 0 and {uncertainty} points</div>
                """

            html += f"""
                </div>
            </div>
            """

        html += f"""</div>"""

        return InfoPage(
            Markup(html)
        )


class GridTrialMaker(CreateAndRateTrialMakerMixin, ImitationChainTrialMaker):
    def grow_network(self, network, experiment):
        grown = super().grow_network(network, experiment)

        if network.head.definition["accuracy"] == GRID_SIZE * GRID_SIZE:
            network.full = True

        return grown


seed_nodes_selection = [
    GridNode(size=GRID_SIZE, n_fill=GRID_FILL, random=True)
    for _ in range(N_GRIDS)
]
# Experiment setup
trial_maker_selection = GridTrialMaker(
    start_nodes=seed_nodes_selection,
    n_creators=N_CREATORS_PER_GENERATION,
    n_raters=N_RATERS,
    node_class=GridNode,
    creator_class=GridCreateTrial,
    rater_class=GridSelectTrial,
    include_previous_iteration=False,
    rate_mode="select",
    target_selection_method="all",
    verbose=True,
    id_="grid_trial_maker",
    chain_type="across",
    expected_trials_per_participant=N_TRIALS_PER_PARTICIPANT,
    max_trials_per_participant=N_TRIALS_PER_PARTICIPANT,
    chains_per_experiment=N_GRIDS,
    balance_across_chains=False,
    check_performance_at_end=True,
    check_performance_every_trial=False,
    propagate_failure=False,
    recruit_mode="n_trials",
    target_n_participants=None,
    wait_for_networks=False,
    max_nodes_per_chain=N_GENERATIONS,
    allow_revisiting_networks_in_across_chains=False,
)


def get_prolific_settings(experiment_duration):
    with open("qualification_prolific_en.json", "r") as f:
        qualification = json.dumps(json.load(f))

    return {
        "recruiter": "prolific",
        "base_payment": 0,
        "prolific_estimated_completion_minutes": DURATION_ESTIMATE / 60,
        "prolific_recruitment_config": qualification,
        "auto_recruit": False,
        "wage_per_hour": 10,
        "currency": "$",
        "show_reward": False,
    }


def get_cap_settings(experiment_duration):
    raise {"wage_per_hour": 9}


recruiter_settings = None
if RECRUITER == "prolific":
    recruiter_settings = get_prolific_settings(DURATION_ESTIMATE)
elif RECRUITER == "cap-recruiter":
    recruiter_settings = get_cap_settings(DURATION_ESTIMATE)


class Exp(psynet.experiment.Experiment):
    label = "Grid creation and selection"
    test_n_bots = 64
    test_mode = "serial"

    config = {
        "recruiter": RECRUITER,
        "wage_per_hour": 0,
        "auto_recruit": False,
        "show_reward": False,
        "initial_recruitment_size": 3,
    }

    if RECRUITER != "hotair":
        config.update(**recruiter_settings)

    timeline = Timeline(
        MainConsent(),
        # InfoPage(
        #     Markup(
        #         f"<h3>The game</h3>"
        #         f"<p>This game has two modes: <i>creation mode</i> and <i>selection mode</i>.</p>"
        #         f"<h4>Creation mode</h4>"
        #         f"<p>In this mode, you will see a grid pattern for 10 seconds, and you will have to reproduce it from memory (it is not expected that you can remember all of it!). </p>"
        #         f"<p>Another participant, who has <i>never</i> seen the original, will compare your grid to other proposals, and guess which is most likely correct. <b>Your goal is to have your proposal selected as many times as possible!</b></p>"
        #         f"<div style='display: flex'>"
        #         f"<div style='display: block; border: 1px solid black; margin: 2px'><img style='display: block;' src='/static/images/create1.png' width='260px' /></div>"
        #         f"<div style='display: block; border: 1px solid black; margin: 2px'><img style='display: block;' src='/static/images/create2.png' width='235px' /></div>"
        #         f"</div>",
        #     ),
        #     time_estimate=45,
        # ),
        trial_maker_selection,
        SuccessfulEndPage(),
    )
