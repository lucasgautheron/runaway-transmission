# pylint: disable=unused-import,abstract-method,unused-argument

from markupsafe import Markup
import psynet.experiment
from psynet.consent import NoConsent
from psynet.modular_page import Prompt, ModularPage, PushButtonControl, TextControl, Control
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

import pandas as pd
import random
import numpy as np

from typing import List

logger = get_logger()

N_CREATORS_PER_GENERATION = 2
N_STORIES = 1
N_COLORLISTS = 1
N_GRIDS = 1
N_GENERATIONS = 10

stories = pd.read_csv("stories.csv").sample(n=N_STORIES)["Story"].tolist()


# Utility function for color HTML generation (no database interaction)
def colors_to_html(colors: List[List[int]], width="50px"):
    """Generate HTML representation of colors without creating a node"""
    html = ""
    for color in colors:
        html += f"<div style='background-color: rgb({color[0]}, {color[1]}, {color[2]}); width: {width}; padding: 0px; margin-right: 10px'>&nbsp;</div>"
    return f"<div style='display:flex;'>{html}</div>"


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

        print(trials)
        for trial in trials:
            if not isinstance(trial, SelectTrialMixin):
                print("exclude ", trial)
                continue

            # Find which target matches this trial's answer
            for i, target in enumerate(all_targets):
                print(trial.answer, target.answer)
                if trial.answer == target.answer:
                    count_dict[i] += 1
                    break

        # Get the index of the target with highest count
        winning_index = max(count_dict, key=count_dict.get)
        last_choice = all_targets[winning_index].answer

        definition = self.seed.copy()
        definition["generation"] += 1
        definition["last_choice"] = last_choice

        return definition


class StoryNode(ArtefactNode):
    """Node class specifically for story artefacts"""

    def __init__(self, story_text: str = None, **kwargs):
        super().__init__(artefact=story_text, **kwargs)

    def get_artefact_type(self):
        return "text"

    def validate_artefact(self):
        """Validate that the story is a non-empty string"""
        return isinstance(self.artefact, str) and len(self.artefact.strip()) > 0

    def get_display_text(self):
        """Get formatted text for display"""
        return self.artefact

    def __repr__(self):
        preview = self.artefact[:50] + "..." if len(self.artefact) > 50 else self.artefact
        return f"StoryNode('{preview}')"


class ColorListNode(ArtefactNode):
    """Node class specifically for color list artefacts"""

    def __init__(self, colors: List[List[int]] = None, n: int = 10, **kwargs):
        if colors is None:
            colors = self.random(n)

        # Validate colors during initialization
        self._validate_colors_format(colors)
        super().__init__(artefact=colors, **kwargs)

    @property
    def n_colors(self):
        """Get number of colors"""
        return len(self.artefact) if self.artefact else 0

    def get_artefact_type(self):
        return "colors"

    def random(self, n: int):
        """Generate n random RGB colors"""
        return np.random.randint(0, 256, size=(n, 3)).tolist()

    def _validate_colors_format(self, colors):
        """Validate colors format with assertions (like original ColorArtefact)"""
        assert all([
            (len(color) == 3) for color in colors
        ]), "All colors must have exactly 3 components (RGB)"
        assert all([
            all([(c >= 0) and (c <= 255) for c in color]) for color in colors
        ]), "All color components must be between 0 and 255"

    def validate_artefact(self):
        """Validate that colors are in correct format (non-asserting version)"""
        if not isinstance(self.artefact, list):
            return False

        for color in self.artefact:
            if not isinstance(color, list) or len(color) != 3:
                return False
            if not all(isinstance(c, int) and 0 <= c <= 255 for c in color):
                return False

        return True

    def html(self, width="50px"):
        """Generate HTML representation of colors"""
        return colors_to_html(self.artefact, width)

    def get_html_display(self, width="50px"):
        """Get HTML representation of colors"""
        return self.html(width)

    @classmethod
    def from_color_data(cls, colors: List[List[int]], **kwargs):
        """Create ColorListNode from existing color data"""
        return cls(colors=colors, **kwargs)

    @classmethod
    def generate_random(cls, n: int = 10, **kwargs):
        """Create ColorListNode with random colors"""
        return cls(n=n, **kwargs)

    def __repr__(self):
        return f"ColorListNode({self.n_colors} colors)"


class GridNode(ArtefactNode):
    """Node class specifically for grid artefacts"""

    def __init__(self, grid_data: List[List[int]] = None, size: int = 8, random: bool = False, **kwargs):
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
    def generate_random(cls, size: int = 8, **kwargs):
        """Create GridNode with random pattern"""
        return cls(size=size, random=True, **kwargs)

    def __repr__(self):
        return f"GridNode({self.size}x{self.size})"


class StoryInputPage(ModularPage):
    def __init__(self, label: str, prompt: str, time_estimate: float, bot_response):
        super().__init__(
            label,
            Prompt(prompt),
            control=TextControl(
                block_copy_paste=True,
                bot_response=bot_response,
            ),
            time_estimate=time_estimate,
        )

    def format_answer(self, raw_answer, **kwargs):
        try:
            assert len(raw_answer.strip()) > 0
        except (ValueError, AssertionError):
            return "INVALID_RESPONSE"

        return raw_answer

    def validate(self, response, **kwargs):
        if response.answer == "INVALID_RESPONSE":
            return FailedValidation("Please enter a response.")
        return None


class ColorReproductionControl(Control):
    macro = "color_picker_control"
    external_template = "color-reproduction.html"

    def __init__(self, num_colors=16, picker_type="wheel_sliders", bot_response=None):
        super().__init__(bot_response=bot_response)  # Pass bot_response to parent Control
        self.num_colors = num_colors
        self.picker_type = picker_type

    @property
    def metadata(self):
        return {
            "num_colors": self.num_colors,
            "picker_type": self.picker_type
        }


class ColorsInputPage(ModularPage):
    def __init__(self, label: str, prompt: str, time_estimate: float, bot_response):
        super().__init__(
            label,
            Prompt(prompt),
            control=ColorReproductionControl(
                bot_response=bot_response  # Pass bot_response to the control, not the page
            ),
            time_estimate=time_estimate,
        )

    def format_answer(self, raw_answer, **kwargs):
        if not isinstance(raw_answer, list):
            return "INVALID_RESPONSE"

        colors = raw_answer
        return [self._validate_color(color) for color in colors]

    def _validate_color(self, color):
        # Ensure color format consistency
        if isinstance(color, dict) and all(k in color for k in ['r', 'g', 'b']):
            return [
                max(0, min(255, int(color['r']))),
                max(0, min(255, int(color['g']))),
                max(0, min(255, int(color['b'])))
            ]
        return [128, 128, 128]

    def validate(self, response, **kwargs):
        print(response)
        if response.answer == "INVALID_RESPONSE":
            return FailedValidation("Please enter a response.")
        if not isinstance(response.answer, list):
            return FailedValidation("Invalid response format.")
        colors = response.answer
        if len(colors) == 0:
            return FailedValidation("Please select at least one color.")
        return None


class GridReproductionControl(Control):
    macro = "grid_reproduction_control"
    external_template = "grid-reproduction.html"

    def __init__(self, grid_size=8, bot_response=None):
        super().__init__(bot_response=bot_response)
        self.grid_size = grid_size

    @property
    def metadata(self):
        return {
            "grid_size": self.grid_size
        }


class GridInputPage(ModularPage):
    def __init__(self, label: str, prompt: str, time_estimate: float, bot_response, grid_size: int = 8):
        super().__init__(
            label,
            Prompt(prompt),
            control=GridReproductionControl(
                grid_size=grid_size,
                bot_response=bot_response
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


# ============================================================================
# SEPARATED TRIAL CLASSES - STORIES
# ============================================================================

class StoryCreateTrial(CreateTrialMixin, ImitationChainTrial):
    """Trial class specifically for story creation tasks"""
    time_estimate = 5

    def first_trial(self):
        """First trial - show original story"""
        return InfoPage(
            Markup(
                f"<h3>Story Reproduction</h3>"
                f"<p>Please read and memorize this story:</p>"
                f"<div style='background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50;'>"
                f"<strong>Original Story:</strong><br>{self.context['original']}"
                f"</div>"
            ),
            time_estimate=self.time_estimate,
        )

    def other_trial(self):
        """Subsequent trials - show original + last winning story"""
        generation = self.definition["generation"]
        return InfoPage(
            Markup(
                f"<h3>Story Reproduction - Generation {generation}</h3>"
                f"<p>Please reproduce the original story.</p>"
                f"<div style='background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50;'>"
                f"<strong>Original Story:</strong><br>{self.context['original']}"
                f"</div>"
                f"<div style='background: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107;'>"
                f"<strong>Last winning story (chosen by a participant):</strong><br>{self.definition['last_choice']}"
                f"</div>"
            ),
            time_estimate=self.time_estimate,
        )

    def input_page(self):
        """Input page for story creation"""
        return StoryInputPage(
            "artefact",
            Markup(
                f"<h3>Your Task</h3>"
                f"<p>Please reproduce the story for a peer. They will read multiple proposals and decide which is most likely correct.</p>"
            ),
            time_estimate=120,
            bot_response=lambda: self.context["original"],
        )

    def show_trial(self, experiment, participant):
        """Show the complete story creation trial"""
        generation = self.definition["generation"]

        if generation == 0:
            info_page = self.first_trial()
        else:
            info_page = self.other_trial()

        input_page = self.input_page()
        return [info_page, input_page]


class StorySelectTrial(SelectTrialMixin, ImitationChainTrial):
    """Trial class specifically for story selection tasks"""
    time_estimate = 5

    def show_trial(self, experiment, participant):
        """Show story selection trial"""
        artefacts = self.get_all_targets()

        return ModularPage(
            "choice",
            Prompt(Markup(
                "<h3>Choose the Best Version</h3>"
                "<p>Please choose the story that seems most faithful to the original:</p>"
                "<ul>"
                + "\n".join([
                    f"<li><strong>Version {i + 1}:</strong> {self.get_target_answer(artefact)}</li>"
                    for i, artefact in enumerate(artefacts)
                ])
                + "</ul>"
            )),
            control=PushButtonControl(
                choices=artefacts,
                labels=[f"Version {i + 1}" for i in range(len(artefacts))],
                arrange_vertically=True,
                bot_response=lambda: random.choice(self.targets),
            ),
            time_estimate=30,
        )


# ============================================================================
# SEPARATED TRIAL CLASSES - COLORS
# ============================================================================

class ColorCreateTrial(CreateTrialMixin, ImitationChainTrial):
    """Trial class specifically for color creation tasks"""
    time_estimate = 5

    def first_trial(self):
        """First trial - show original colors"""
        return InfoPage(
            Markup(
                f"<h3>Colors Reproduction</h3>"
                f"<p>Please read and memorize the following colors:</p>"
                f"<div style='padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50;'>"
                f"<strong>Original colors:</strong><br>{colors_to_html(self.context['original'])}"
                f"</div>"
            ),
            time_estimate=self.time_estimate,
        )

    def other_trial(self):
        """Subsequent trials - show original + last winning colors"""
        generation = self.definition["generation"]
        return InfoPage(
            Markup(
                f"<h3>Colors Reproduction - Generation {generation}</h3>"
                f"<p>Please read and memorize the following colors:</p>"
                f"<div style='padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50;'>"
                f"<strong>Original colors:</strong><br>{colors_to_html(self.context['original'])}"
                f"</div>"
                f"<div style='padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107;'>"
                f"<strong>Last winning colors (chosen by a participant):</strong><br>{colors_to_html(self.definition['last_choice'])}"
                f"</div>"
            ),
            time_estimate=self.time_estimate,
        )

    def input_page(self):
        """Input page for color creation"""
        return ColorsInputPage(
            "artefact",
            Markup(
                f"<h3>Your Task</h3>"
                f"<p>Please reproduce the colors for a peer. They will read multiple proposals and decide which is most likely correct.</p>"
            ),
            time_estimate=120,
            bot_response=lambda: self.context["original"]
        )

    def show_trial(self, experiment, participant):
        """Show the complete color creation trial"""
        generation = self.definition["generation"]

        if generation == 0:
            info_page = self.first_trial()
        else:
            info_page = self.other_trial()

        input_page = self.input_page()
        return [info_page, input_page]


class ColorSelectTrial(SelectTrialMixin, ImitationChainTrial):
    """Trial class specifically for color selection tasks"""
    time_estimate = 5

    def show_trial(self, experiment, participant):
        print("Show select trial")
        """Show color selection trial"""
        artefacts = self.get_all_targets()

        for i, artefact in enumerate(artefacts):
            print(f"choice {i}: {self.get_target_answer(artefact)}")

        return ModularPage(
            "choice",
            Prompt(Markup(
                "<h3>Choose the Best Version</h3>"
                "<p>Please choose the colors that seem most faithful to the original:</p>"
                "<ul>"
                + "\n".join([
                    f"<li>{colors_to_html(self.get_target_answer(artefact))}</li>"
                    for i, artefact in enumerate(artefacts)
                ])
                + "</ul>"
            )),
            control=PushButtonControl(
                choices=artefacts,
                labels=[f"Version {i + 1}" for i in range(len(artefacts))],
                arrange_vertically=True,
                bot_response=lambda: random.choice(self.targets),
            ),
            time_estimate=30,
        )


class GridCreateTrial(CreateTrialMixin, ImitationChainTrial):
    """Trial class specifically for grid creation tasks"""
    time_estimate = 5

    def first_trial(self):
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

    def other_trial(self):
        """Subsequent trials - show original + last winning grid"""
        generation = self.definition["generation"]
        return InfoPage(
            Markup(
                f"<h3>Grid Pattern Reproduction - Generation {generation}</h3>"
                f"<p>Please study and memorize the following grid patterns:</p>"
                f"<div style='padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50; text-align: center;'>"
                f"<strong>Original Pattern:</strong><br>{grid_to_html(self.context['original'])}"
                f"</div>"
                f"<div style='padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107; text-align: center;'>"
                f"<strong>Last winning pattern (chosen by a participant):</strong><br>{grid_to_html(self.definition['last_choice'])}"
                f"</div>"
            ),
            time_estimate=self.time_estimate,
        )

    def input_page(self):
        """Input page for grid creation"""
        original_grid = self.context["original"]
        grid_size = len(original_grid)

        return GridInputPage(
            "artefact",
            Markup(
                f"<h3>Your Task</h3>"
                f"<p>Please reproduce the grid pattern for a peer. They will see multiple proposals and decide which is most likely correct.</p>"
                f"<p><strong>Instructions:</strong> Click cells to toggle between black and white.</p>"
            ),
            time_estimate=120,
            bot_response=lambda: self.context["original"],
            grid_size=grid_size
        )

    def show_trial(self, experiment, participant):
        """Show the complete grid creation trial"""
        generation = self.definition["generation"]

        if generation == 0:
            info_page = self.first_trial()
        else:
            info_page = self.other_trial()

        input_page = self.input_page()
        return [info_page, input_page]


class GridSelectTrial(SelectTrialMixin, ImitationChainTrial):
    """Trial class specifically for grid selection tasks"""
    time_estimate = 5

    def show_trial(self, experiment, participant):
        """Show grid selection trial"""
        artefacts = self.get_all_targets()

        return ModularPage(
            "choice",
            Prompt(Markup(
                "<h3>Choose the Best Version</h3>"
                "<p>Please choose the grid pattern that seems most faithful to the original:</p>"
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
                arrange_vertically=True,
                bot_response=lambda: random.choice(self.targets),
            ),
            time_estimate=30,
        )


# ============================================================================
# TRIAL MAKERS
# ============================================================================

class StoryTrialMaker(CreateAndRateTrialMakerMixin, ImitationChainTrialMaker):
    """Trial maker specifically for story experiments"""
    pass


class ColorTrialMaker(CreateAndRateTrialMakerMixin, ImitationChainTrialMaker):
    """Trial maker specifically for color experiments"""
    pass


class GridTrialMaker(CreateAndRateTrialMakerMixin, ImitationChainTrialMaker):
    """Trial maker specifically for grid experiments"""
    pass


# ============================================================================
# FACTORY FUNCTIONS AND SETUP
# ============================================================================

def create_grid_nodes(n_grids: int, grid_size: int = 8) -> List[GridNode]:
    """Create grid nodes"""
    return [GridNode(size=grid_size, random=True) for _ in range(n_grids)]


def create_story_nodes(stories_list: List[str]) -> List[StoryNode]:
    """Create story nodes from a list of story texts"""
    return [StoryNode(story) for story in stories_list]


def create_color_nodes(n_colorlists: int, colors_per_list: int = 10) -> List[ColorListNode]:
    """Create color list nodes"""
    return [ColorListNode(n=colors_per_list) for _ in range(n_colorlists)]


# Create nodes using the new classes
story_nodes = create_story_nodes(stories)
color_nodes = create_color_nodes(N_COLORLISTS, 10)
grid_nodes = create_grid_nodes(N_GRIDS, 8)

# Validation example
for node in story_nodes + color_nodes:
    if not node.validate_artefact():
        logger.warning(f"Invalid artefact in node: {node}")

# ============================================================================
# EXPERIMENT SETUP
# ============================================================================

# Choose which type of experiment to run
EXPERIMENT_TYPE = "grids"  # Change to "stories" or "mixed"

if EXPERIMENT_TYPE == "stories":
    nodes = story_nodes
    trial_maker = StoryTrialMaker(
        n_creators=N_CREATORS_PER_GENERATION,
        n_raters=1,
        node_class=StoryNode,
        creator_class=StoryCreateTrial,
        rater_class=StorySelectTrial,
        include_previous_iteration=False,
        rate_mode="select",
        target_selection_method="all",
        verbose=True,
        id_="story_trial_maker",
        chain_type="across",
        expected_trials_per_participant=len(nodes),
        max_trials_per_participant=len(nodes),
        start_nodes=nodes,
        chains_per_experiment=len(nodes),
        balance_across_chains=False,
        check_performance_at_end=True,
        check_performance_every_trial=False,
        propagate_failure=False,
        recruit_mode="n_trials",
        target_n_participants=None,
        wait_for_networks=False,
        max_nodes_per_chain=N_GENERATIONS,
    )
elif EXPERIMENT_TYPE == "colors":
    nodes = color_nodes
    trial_maker = ColorTrialMaker(
        n_creators=N_CREATORS_PER_GENERATION,
        n_raters=1,
        node_class=ColorListNode,
        creator_class=ColorCreateTrial,
        rater_class=ColorSelectTrial,
        include_previous_iteration=False,
        rate_mode="select",
        target_selection_method="all",
        verbose=True,
        id_="color_trial_maker",
        chain_type="across",
        expected_trials_per_participant=len(nodes),
        max_trials_per_participant=len(nodes),
        start_nodes=nodes,
        chains_per_experiment=len(nodes),
        balance_across_chains=False,
        check_performance_at_end=True,
        check_performance_every_trial=False,
        propagate_failure=False,
        recruit_mode="n_trials",
        target_n_participants=None,
        wait_for_networks=False,
        max_nodes_per_chain=N_GENERATIONS,
    )
elif EXPERIMENT_TYPE == "grids":
    nodes = grid_nodes
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
        expected_trials_per_participant=len(nodes),
        max_trials_per_participant=len(nodes),
        start_nodes=nodes,
        chains_per_experiment=len(nodes),
        balance_across_chains=False,
        check_performance_at_end=True,
        check_performance_every_trial=False,
        propagate_failure=False,
        recruit_mode="n_trials",
        target_n_participants=None,
        wait_for_networks=False,
        max_nodes_per_chain=N_GENERATIONS,
    )
else:  # mixed
    nodes = story_nodes + color_nodes
    # For mixed experiments, you might want to create a more complex setup
    # This is a simplified version - you may need to adjust based on your needs
    trial_maker = StoryTrialMaker(  # You could create a MixedTrialMaker if needed
        n_creators=N_CREATORS_PER_GENERATION,
        n_raters=1,
        node_class=ArtefactNode,
        creator_class=StoryCreateTrial,  # This won't work well for mixed - consider creating unified trials
        rater_class=StorySelectTrial,
        include_previous_iteration=False,
        rate_mode="select",
        target_selection_method="all",
        verbose=True,
        id_="mixed_trial_maker",
        chain_type="across",
        expected_trials_per_participant=len(nodes),
        max_trials_per_participant=len(nodes),
        start_nodes=nodes,
        chains_per_experiment=len(nodes),
        balance_across_chains=False,
        check_performance_at_end=True,
        check_performance_every_trial=False,
        propagate_failure=False,
        recruit_mode="n_trials",
        target_n_participants=None,
        wait_for_networks=False,
        max_nodes_per_chain=N_GENERATIONS,
    )


class Exp(psynet.experiment.Experiment):
    label = "Story transmission experiment"
    initial_recruitment_size = 1

    timeline = Timeline(
        NoConsent(),
        trial_maker,
        SuccessfulEndPage(),
    )