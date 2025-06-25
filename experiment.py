# pylint: disable=unused-import,abstract-method,unused-argument

from markupsafe import Markup
import psynet.experiment
from psynet.consent import NoConsent, MainConsent
from psynet.modular_page import Prompt, ModularPage, PushButtonControl, Control
from psynet.page import InfoPage, SuccessfulEndPage
from psynet.timeline import FailedValidation, Timeline
from psynet.trial.create_and_rate import (
    CreateAndRateNode,
    CreateAndRateNodeMixin,
    CreateAndRateTrialMakerMixin,
    CreateTrialMixin,
    SelectTrialMixin,
)
from psynet.trial.imitation_chain import (
    ImitationChainTrial,
    ImitationChainTrialMaker,
    ImitationChainNode,
    ImitationChainNetwork,
)
from psynet.trial.chain import ChainNode
from psynet.utils import get_logger

import numpy as np
import random
from typing import List

logger = get_logger()

N_CREATORS_PER_GENERATION = 3
N_GRIDS = 5
N_GENERATIONS = 10


# Utility function for grid HTML generation (no database interaction)
def grid_to_html(grid_data: List[List[int]], cell_size="20px"):
    """Generate HTML representation of grid without creating a node"""
    html = f'<div style="display: inline-block; border: 1px solid #333;">'
    for row in grid_data:
        html += '<div style="display: flex; line-height: 0;">'
        for cell in row:
            color = "#000" if cell == 1 else "#fff"
            html += f'<div style="background-color: {color}; width: {cell_size}; height: {cell_size}; border: 1px solid #ccc;"></div>'
        html += '</div>'
    html += '</div>'
    return html


class ArtefactNode(ChainNode, CreateAndRateNodeMixin):
    def __init__(self, artefact=None, **kwargs):
        # Set context before calling super().__init__
        if artefact is not None:
            kwargs['context'] = {"original": artefact, "artefact_type": self.get_artefact_type()}

        super().__init__(**kwargs)
        self.artefact = artefact

    def get_artefact_type(self):
        """Override in subclasses to specify artefact type"""
        raise NotImplementedError("Subclasses must implement get_artefact_type()")

    def validate_artefact(self):
        """Override in subclasses to validate artefact format"""
        return True

    def create_initial_seed(self, experiment=None, participant=None):
        return {"generation": 0, "last_choice": ""}

    def create_definition_from_seed(self, seed, experiment, participant):
        return seed

    def summarize_trials(self, trials, experiment, participant):
        logger.info(f"Summarizing trials for node {self.id}")
        trial_maker = self.trial_maker

        all_rate_trials = trial_maker.rater_class.query.filter_by(
            node_id=self.id, failed=False, finalized=True
        ).all()

        last_generation = max([trial.definition["generation"] for trial in all_rate_trials])
        all_rate_trials = [trial for trial in all_rate_trials if trial.definition["generation"] == last_generation]

        all_targets = all_rate_trials[0].get_all_targets()

        count_dict = {i: 0 for i in range(len(all_targets))}

        for trial in all_rate_trials:
            if not isinstance(trial, SelectTrialMixin):
                print("exclude ", trial)
                continue

            # Find which target matches this trial's answer
            for i, target in enumerate(all_targets):
                print(trial.answer, str(target), target.answer)
                if trial.answer == str(target):
                    count_dict[i] += 1
                    break

        print(count_dict)

        # Get the index of the target with highest count
        winning_index = max(count_dict, key=count_dict.get)
        last_choice = all_targets[winning_index].answer

        definition = self.seed.copy()
        definition["generation"] += 1
        definition["last_choice"] = last_choice

        return definition


class GridNode(ArtefactNode):
    """Node class specifically for grid artefacts"""

    def __init__(self, grid_data: List[List[int]] = None, size: int = 10, random: bool = False, **kwargs):
        if grid_data is None and random is True:
            grid_data = self.random(size)

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

    def random(self, size: int):
        """Generate random grid with 0s and 1s (white and black)"""
        return np.random.choice([0, 1], size=(size, size), p=[0.8, 0.2]).tolist()

    def _validate_grid_format(self, grid_data):
        """Validate grid format with assertions"""
        assert isinstance(grid_data, list), "Grid must be a list"
        assert len(grid_data) > 0, "Grid cannot be empty"
        assert all(isinstance(row, list) for row in grid_data), "All rows must be lists"
        assert len(set(len(row) for row in grid_data)) == 1, "All rows must have same length"
        assert len(grid_data) == len(grid_data[0]), "Grid must be square (NxN)"
        assert all(
            all(cell in [0, 1] for cell in row) for row in grid_data
        ), "All cells must be 0 (white) or 1 (black)"

    def validate_artefact(self):
        """Validate that grid is in correct format (non-asserting version)"""
        if not isinstance(self.artefact, list):
            return False

        if len(self.artefact) == 0:
            return False

        for row in self.artefact:
            if not isinstance(row, list):
                return False
            if len(row) != len(self.artefact):
                return False
            if not all(cell in [0, 1] for cell in row):
                return False

        return True

    def html(self, cell_size="20px"):
        """Generate HTML representation of grid"""
        return grid_to_html(self.artefact, cell_size)

    def get_html_display(self, cell_size="20px"):
        """Get HTML representation of grid"""
        return self.html(cell_size)

    @classmethod
    def from_grid_data(cls, grid_data: List[List[int]], **kwargs):
        """Create GridNode from existing grid data"""
        return cls(grid_data=grid_data, **kwargs)

    @classmethod
    def generate_random(cls, size: int = 10, **kwargs):
        """Create GridNode with random pattern"""
        return cls(size=size, random=True, **kwargs)

    @classmethod
    def accuracy(cls, truth, attempt):
        """Create GridNode with random pattern"""
        truth = np.array(truth) * 1
        attempt = np.array(attempt) * 1

        return (truth == attempt).sum()

    def __repr__(self):
        return f"GridNode({self.size}x{self.size})"


class GridReproductionControl(Control):
    macro = "grid_reproduction_control"
    external_template = "grid-reproduction.html"

    def __init__(self, grid_size=8, prefill_grid=None, **kwargs):
        self.grid_size = grid_size
        self.prefill_grid = prefill_grid
        super().__init__(**kwargs, bot_response=self.bot_response())

    def bot_response(self):
        if self.prefill_grid:
            grid = np.array(self.prefill_grid)
        else:
            grid = np.random.choice([0, 1], size=(self.grid_size, self.grid_size), p=[0.8, 0.2])

        x, y = np.random.choice(self.grid_size, 2)
        grid[x, y] = 1 - grid[x, y]

        return grid.tolist()

    @property
    def metadata(self):
        return {
            "grid_size": self.grid_size,
            "prefill_grid": self.prefill_grid
        }


class GridInputPage(ModularPage):
    def __init__(self, label: str, prompt: str, time_estimate: float, prefill_grid=None, grid_size: int = 10):
        super().__init__(
            label,
            Prompt(prompt),
            control=GridReproductionControl(
                grid_size=grid_size,
                prefill_grid=prefill_grid
            ),
            time_estimate=time_estimate,
        )

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
            return FailedValidation("Please enter a response.")
        if not isinstance(response.answer, list):
            return FailedValidation("Invalid response format.")
        grid_data = response.answer
        if len(grid_data) == 0:
            return FailedValidation("Please create a grid pattern.")
        return None


class GridCreateTrial(CreateTrialMixin, ImitationChainTrial):
    """Trial class specifically for grid creation tasks"""
    time_estimate = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var.accuracy = None

    def trial(self):
        """First trial - show original grid"""
        return InfoPage(
            Markup(
                f"<h3>Grid Pattern Reproduction</h3>"
                f"<p>Please study and memorize this grid pattern:</p>"
                f"<div style='padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50; text-align: center;'>"
                f"<strong>Original Pattern:</strong><br>{grid_to_html(self.context['original'])}"
                f"</div>"
            ),
            time_estimate=self.time_estimate,
        )

    def input_page(self):
        """Input page for grid creation"""
        generation = self.definition["generation"]
        original_grid = self.context["original"]
        grid_size = len(original_grid)

        # For generation 0, start from scratch (no prefill); for generation > 0, start with last_choice
        if generation == 0:
            prefill_grid = None  # No prefill for first generation
        else:
            prefill_grid = self.definition['last_choice']

        text = Markup(
            f"<h3>Now, reproduce the grid!</h3>"
            f"<p>Please reproduce the grid you just saw below. Then, another participant <b>who has not seen the original</b> will see multiple proposals, including yours, and decide which is most likely correct. Can you beat the others?</p>"
            f"<p><strong>Instructions:</strong> Click cells to toggle between black and white.</p>"
        ) if generation == 0 else Markup(
            f"<h3>Now, reproduce the grid!</h3>"
            f"<p>Please reproduce the grid you just saw below. Then, another participant <b>who has not seen the original</b> will see multiple proposals, including yours, and decide which is most likely correct. Can you beat the others?</p>"
            f"<p>You are starting from the grid that was chosen last.</p>"
            f"<p><strong>Instructions:</strong> Click cells to toggle between black and white.</p>"
        )

        return GridInputPage(
            "artefact",
            text,
            time_estimate=120,
            prefill_grid=prefill_grid,
            grid_size=grid_size,
        )

    def show_trial(self, experiment, participant):
        """Show the complete grid creation trial"""
        info_page = self.trial()

        input_page = self.input_page()
        return [info_page, input_page]

    def show_feedback(self, experiment, participant):
        if self.definition['generation'] == 0:
            return

        accuracy = GridNode.accuracy(self.context["original"], self.answer)
        previous_accuracy = GridNode.accuracy(self.context["original"], self.definition['last_choice'])
        delta = accuracy - previous_accuracy

        if delta == 0:
            return InfoPage("Accuracy unchanged!")
        elif delta > 0:
            plural = "s are" if delta > 1 else " is"
            return InfoPage(f"Your proposal is more accurate ({delta} more cell{plural} correct)!")
        else:
            plural = "s" if delta < -1 else ""
            return InfoPage(f"Your proposal is less accurate ({-delta} more error{plural})!")


class GridSelectTrial(SelectTrialMixin, ImitationChainTrial):
    """Trial class specifically for grid selection tasks"""
    time_estimate = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var.success = None
        self.var.accuracy = None

    def show_trial(self, experiment, participant):
        artefacts = self.get_all_targets()

        return ModularPage(
            "choice",
            Prompt(Markup(
                "<h3>Choose the most accurate</h3>"
                f"<p>{N_CREATORS_PER_GENERATION} participants have attempted to reproduce a grid from memory.</p>"
                "<p>Without having seen the original grid, try and choose the proposal you believe to be its most accurate reproduction.</p>"
                "<div style='display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin: 20px 0;'>"
                + "\n".join([
                    f"<div style='text-align: center;'><strong>Version {i + 1}:</strong><br>{grid_to_html(self.get_target_answer(artefact))}</div>"
                    for i, artefact in enumerate(artefacts)
                ])
                + "</div>"
            )),
            control=PushButtonControl(
                choices=artefacts,
                labels=[f"Version {i + 1}" for i in range(len(artefacts))],
                arrange_vertically=False,
                bot_response=lambda: random.choice(self.targets),
            ),
            time_estimate=30,
        )

    def show_feedback(self, experiment, participant):
        artefacts = self.get_all_targets()

        accuracy = dict()
        for i, artefact in enumerate(artefacts):
            accuracy[str(artefact)] = GridNode.accuracy(
                self.get_target_answer(artefact),
                self.context['original']
            )

        best = max(list(accuracy.values()))
        successful_prediction = (accuracy[str(self.answer)] == best)

        return InfoPage(
            Markup("Congratulations, you chose the most accurate artefact!")
            if successful_prediction == True
            else Markup("This was not the most accurate artefact. Better luck next time!")
        )


class GridTrialMaker(CreateAndRateTrialMakerMixin, ImitationChainTrialMaker):
    """Trial maker specifically for grid experiments"""

    def __init__(self, n_grids: int, grid_size: int = 10, *args, **kwargs):
        super().__init__(start_nodes=[GridNode(size=grid_size, random=True) for _ in range(n_grids)], *args, **kwargs)

    def finalize_trial(self, answer, trial, experiment, participant):
        super().finalize_trial(answer, trial, experiment, participant)

        if isinstance(trial, SelectTrialMixin):
            artefacts = trial.get_all_targets()

            accuracy = dict()
            for i, artefact in enumerate(artefacts):
                accuracy[str(artefact)] = GridNode.accuracy(
                    trial.get_target_answer(artefact),
                    trial.context['original']
                )

            best = max(list(accuracy.values()))
            trial.var.success = bool(accuracy[str(trial.answer)] == best)
            trial.var.accuracy = int(accuracy[str(trial.answer)])

        elif isinstance(trial, CreateTrialMixin):
            trial.var.accuracy = int(GridNode.accuracy(trial.context["original"], trial.answer))


# Experiment setup
trial_maker = GridTrialMaker(
    n_creators=N_CREATORS_PER_GENERATION,
    n_raters=1,
    node_class=GridNode,
    creator_class=GridCreateTrial,
    rater_class=GridSelectTrial,
    include_previous_iteration=False,
    rate_mode="select",
    target_selection_method="all",
    verbose=True,
    id_="grid_trial_maker",
    chain_type="across",
    expected_trials_per_participant=N_GRIDS,
    max_trials_per_participant=N_GRIDS,
    n_grids=N_GRIDS,
    grid_size=10,
    chains_per_experiment=N_GRIDS,
    balance_across_chains=False,
    check_performance_at_end=True,
    check_performance_every_trial=False,
    propagate_failure=False,
    recruit_mode="n_trials",
    target_n_participants=None,
    wait_for_networks=False,
    max_nodes_per_chain=N_GENERATIONS,
    allow_revisiting_networks_in_across_chains=False
)


class Exp(psynet.experiment.Experiment):
    label = "Grid pattern transmission experiment"
    initial_recruitment_size = 1

    timeline = Timeline(
        MainConsent(),
        trial_maker,
        SuccessfulEndPage(),
    )
