# pylint: disable=unused-import,abstract-method,unused-argument

from markupsafe import Markup
import psynet.experiment
from psynet.consent import MainConsent
from psynet.modular_page import (
    Prompt, ModularPage, PushButtonControl, Control,
)
from psynet.page import InfoPage, SuccessfulEndPage
from psynet.timeline import FailedValidation, Timeline, Response, Event

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
import random
from typing import List
import json

logger = get_logger()

GRID_SIZE = 16
GRID_FILL = 48
MAX_ACCURACY = GRID_SIZE * GRID_SIZE
NUM_EDITS = 5
CREATOR_FEEDBACK = False

N_TRIALS_PER_PARTICIPANT = 2
N_CREATORS_PER_GENERATION = 3
N_RATERS = 1
N_GRIDS = 2
N_GENERATIONS = 20
RECRUITER = "hotair"
DURATION_ESTIMATE = 120 + N_TRIALS_PER_PARTICIPANT * 30


# assert N_TRIALS_PER_PARTICIPANT % (N_CREATORS_PER_GENERATION + 1) == 0


# Utility function for grid HTML generation
def grid_to_html(grid_data: List[List[int]], cell_size="20px", border_size="1px"):
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
        return {"generation": 0, "last_choice": "", "accuracy": 0}

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
            self.artefact = self.definition["last_choice"]

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
        last_choice = all_targets[winning_index].answer

        # Seed for next generation
        definition = self.seed.copy()
        definition["generation"] += 1
        definition["last_choice"] = last_choice
        definition["accuracy"] = all_targets[winning_index].var.accuracy

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
            truth=List[List[int]],
            **kwargs,
    ):
        self.grid_size = grid_size
        self.prefill_grid = prefill_grid
        self.num_edits = num_edits
        self.truth = np.array(truth)
        super().__init__(**kwargs, bot_response=self.bot_response())

    def bot_response(self):
        if self.prefill_grid:
            grid = np.array(self.prefill_grid, dtype=int)
        else:
            grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        x, y = np.nonzero(self.truth & (~grid))
        missing = [[x[i], y[i]] for i in range(len(x))]

        # pick elements to correct
        x, y, = np.nonzero((~self.truth) & grid)
        incorrect = [[x[i], y[i]] for i in range(len(x))]

        if len(incorrect) > 0:
            n_correct = np.random.poisson(1)
            to_correct = random.sample(
                incorrect, k=np.minimum(n_correct, len(incorrect)),
            )

            for x, y, in to_correct:
                grid[x, y] = 0

        # pick elements to add
        if len(missing):
            n_add = 1 + np.random.poisson(2)
            add = random.choices(missing, k=np.minimum(n_add, len(missing)))

            error_rate = 0.33
            for x, y in add:
                if np.random.uniform(0, 1) < error_rate:
                    x_possible = [x - 1, x + 1]
                    y_possible = [y - 1, y + 1]
                    x_possible = [x for x in x_possible if
                                  x >= 0 and x < self.grid_size]
                    y_possible = [y for y in y_possible if
                                  y >= 0 and y < self.grid_size]

                    x_choice = np.random.choice(x_possible)
                    y_choice = np.random.choice(y_possible)

                    grid[x_choice, y_choice] = 1
                else:
                    grid[x, y] = 1

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
            prefill_grid=None, truth: List[List[int]] = None,
            grid_size: int = 10,
    ):
        super().__init__(
            label,
            Prompt(prompt),
            control=GridReproductionControl(
                grid_size=grid_size,
                prefill_grid=prefill_grid,
                num_edits=NUM_EDITS,
                truth=truth,
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
    time_estimate = 30 + 1

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

        prior_proposals = self.query.filter_by(
            network_id=self.network_id, failed=False, finalized=True
        ).all()

        proposals = dict()
        trial_maker = None
        for proposal in prior_proposals:
            generation = proposal.node.definition["generation"]
            proposals[generation] = proposals.get(generation, []) + [
                proposal,
            ]
            trial_maker = proposal.trial_maker

        selections = trial_maker.rater_class.query.filter_by(
            network_id=self.network_id, failed=False, finalized=True
        ).all()
        winners = []
        for selection in selections:
            winners.append(selection.answer)

        logger.info(proposals)

        active_generation = self.definition["generation"]

        html = "<div style='display: flex; flex-direction: row; justify-content: center; align-items: flex-start; gap: 10px;'>"
        for generation in sorted(proposals.keys()):
            if generation >= active_generation:
                continue

            # Each generation column
            html += "<div style='display: flex; flex-direction: column; align-items: center; gap: 5px;'>"

            # Generation label
            html += f"<div style='margin-bottom: 2px;'>Step {generation}</div>"

            # Proposals within this generation
            for proposal in proposals[generation]:
                if str(proposal) in winners:
                    html += f"<div style='border: 2px solid green;'>{grid_to_html(proposal.answer, cell_size='5px', border_size='0px')}</div>"
                else:
                    html += f"<div>{grid_to_html(proposal.answer, cell_size='5px', border_size='0px')}</div>"

            html += "</div>"

        html += "</div>"

        print(html)

        return GridDisplayPage(
            Markup(
                f"<h3>Observe previous patterns <i>(creation mode)</i></h3>"
                f"<p>Below, you may observe prior proposals and the winner at each iteration.</p>"
                f"<p>On the next page, you will start from the latest winner and make up to five edits.</p>"
                f"{html}"
            ),
        )

    def trial(self):
        """Show appropriate trial based on generation"""
        generation = self.definition["generation"]

        if generation == 0:
            return self.first_trial()
        else:
            return self.other_trial()

    def input_page(self):
        """Input page for grid creation"""
        generation = self.definition["generation"]
        original_grid = self.context["original"]
        grid_size = len(original_grid)

        # For generation 0, start from scratch (no prefill); for generation > 0, start with last_choice
        if generation == 0:
            prefill_grid = np.zeros(
                (GRID_SIZE, GRID_SIZE),
            ).tolist()  # Empty for first generation
        else:
            prefill_grid = self.definition['last_choice']

        text = Markup(
            f"<h3>Make a proposal <i>(creation mode)</i></h3>"
            f"<p>Please add up to {NUM_EDITS} squares to the grid. You will receive a reward if your proposal is selected.</p>",
        ) if generation == 0 else Markup(
            f"<h3>Edit the latest proposal <i>(creation mode)</i></h3>"
            f"<p>You are free to make up to {NUM_EDITS} modifications. You will receive a reward if your proposal is selected.</p>",
        )

        return GridInputPage(
            "artefact",
            text,
            time_estimate=30,
            prefill_grid=prefill_grid,
            truth=self.context["original"],
            grid_size=grid_size,
        )

    def show_trial(self, experiment, participant):
        """Show the complete grid creation trial"""
        info_page = self.trial()
        input_page = self.input_page()
        return [info_page, input_page]

    def score_answer(self, answer, definition):
        self.var.accuracy = int(
            GridNode.accuracy(self.context["original"], self.answer),
        )
        return self.var.accuracy

    def show_feedback(self, experiment, participant):
        if CREATOR_FEEDBACK == False:
            return InfoPage(
                "Proposal received, thank you! Click 'next' when you are ready to see the next grid.",
            )

        if self.definition['generation'] == 0:
            return

        accuracy = GridNode.accuracy(self.context["original"], self.answer)
        previous_accuracy = GridNode.accuracy(
            self.context["original"], self.definition['last_choice'],
        )
        delta = accuracy - previous_accuracy

        if delta == 0:
            return InfoPage("Accuracy unchanged!")
        elif delta > 0:
            plural = "s are" if delta > 1 else " is"
            return InfoPage(
                f"Your proposal is more accurate ({delta} more cell{plural} correct)!",
            )
        else:
            plural = "s" if delta < -1 else ""
            return InfoPage(
                f"Your proposal is less accurate ({-delta} more error{plural})!",
            )


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
            score[i] = np.sum(self.get_target_answer(artefact))
            # score[i] = 0
            #
            # for j, artefact_compare in enumerate(artefacts):
            #     if i == j:
            #         continue
            #
            #     score[i] += GridNode.accuracy(
            #         self.get_target_answer(artefact),
            #         self.get_target_answer(artefact_compare)
            #     )

        return str(artefacts[max(score.keys(), key=score.get)])

    def show_trial(self, experiment, participant):
        artefacts = self.get_all_targets()
        logger.info(artefacts)

        for artefact in artefacts:
            logger.info(artefact.id)

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
                    "<h3>Choose the most accurate <i>(selection mode)</i></h3>"
                    f"<p>{N_CREATORS_PER_GENERATION} participants have attempted to reproduce a grid from memory.</p>"
                    "<p>Without having seen the original grid, try and choose the proposal you believe to be closer to the original.</p>"
                    "<div style='display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin: 20px 0;'>"
                    + "\n".join(
                        [
                            f"<div style='text-align: center;'><strong>Version {i + 1}:</strong><br>{grid_to_html(self.get_target_answer(artefact))}</div>"
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

    def score_answer(self, answer, definition):
        super().score_answer(answer, definition)

        artefacts = self.get_all_targets()

        accuracy = dict()
        for i, artefact in enumerate(artefacts):
            accuracy[str(artefact)] = GridNode.accuracy(
                self.get_target_answer(artefact),
                self.context['original'],
            )

        best = max(list(accuracy.values()))
        self.var.success = bool(accuracy[str(self.answer)] == best)
        self.var.accuracy = int(accuracy[str(self.answer)])
        return self.var.success * 1

    def show_feedback(self, experiment, participant):
        return InfoPage(
            Markup("Congratulations, you chose the most accurate artefact!")
            if self.score == 1
            else Markup(
                "This was not the most accurate artefact. Better luck next time!",
            ),
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
        # InfoPage(
        #     Markup(
        #         "<h4>Creation mode</h4>"
        #         f"<p>Often, your grid will be pre-filled based on the last selected proposal.</p>"
        #         f"<div style='display: flex'>"
        #         f"<div style='display: block; border: 1px solid black; margin: 2px'><img style='display: block;' src='/static/images/create3.png' width='260px' /></div>"
        #         f"<div style='display: block; border: 1px solid black; margin: 2px'><img style='display: block;' src='/static/images/create4.png' width='235px' /></div>"
        #         f"</div>",
        #     ),
        #     time_estimate=30,
        # ),
        # ModularPage(
        #     label="mock",
        #     prompt=Markup(
        #         f"<h4>Selection mode (less frequent)</h4>"
        #         f"<p>In this mode, you will see several attempts to reproduce a grid, and you will have to guess which is the most accurate.</p>"
        #         f"<p>In this mode, you will <b>never</b> see the original! Use your best judgment to decide which attempt is closer to the truth.</p>"
        #         f"<div style='display: flex'>"
        #         f"<img style='display: block; border: 1px solid black; margin: 2px' width='342px' src='/static/images/select.png' />"
        #         f"</div>"
        #         f"<div style='margin: 0 auto; text-align: center;'>Which version you would pick, in this example?</div>",
        #     ),
        #     control=PushButtonControl(
        #         choices=[0, 1, 2],
        #         labels=[f"Version {i + 1}" for i in range(3)],
        #         arrange_vertically=False,
        #         bot_response=lambda: "",
        #     ),
        #     time_estimate=30,
        # ),
        # InfoPage(
        #     "Your training is complete. Please click 'next' when you are ready to start the experiment!",
        #     time_estimate=10,
        # ),
        # PseudonymInputPage(),
        trial_maker_selection,
        SuccessfulEndPage(),
    )
