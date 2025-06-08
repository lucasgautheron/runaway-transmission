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
N_GENERATIONS = 10

stories = pd.read_csv("stories.csv").sample(n=N_STORIES)["Story"].tolist()


# Utility function for color HTML generation (no database interaction)
def colors_to_html(colors: List[List[int]], width="50px"):
    """Generate HTML representation of colors without creating a node"""
    html = ""
    for color in colors:
        html += f"<div style='background-color: rgb({color[0]}, {color[1]}, {color[2]}); width: {width}; padding: 0px; margin-right: 10px'>&nbsp;</div>"
    return f"<div style='display:flex;'>{html}</div>"


class ArtefactNode(ChainNode, CreateAndRateNodeMixin):
    def __init__(self, artefact=None, **kwargs):
        self.artefact = artefact
        # Set context before calling super().__init__
        if artefact is not None:
            kwargs['context'] = {"original": artefact, "artefact_type": self.get_artefact_type()}
        super().__init__(**kwargs)

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
        trial_maker = self.trial_maker
        all_rate_trials = trial_maker.rater_class.query.filter_by(
            node_id=self.id, failed=False, finalized=True
        ).all()
        all_targets = all_rate_trials[0].get_all_targets()

        # Use indices as keys - works for any data type
        count_dict = {i: 0 for i in range(len(all_targets))}

        for trial in trials:
            # Find which target matches this trial's answer
            for i, target in enumerate(all_targets):
                if trial.answer == target.answer:
                    count_dict[i] += 1
                    break

        # Get the index of the target with highest count
        winning_index = max(count_dict, key=count_dict.get)
        last_choice = all_targets[winning_index].answer

        definition = self.seed.copy()
        definition["generation"] += 1
        definition["last_choice"] = last_choice

        print(definition)

        return definition


class StoryNode(ArtefactNode):
    """Node class specifically for story artefacts"""

    def __init__(self, story_text: str, **kwargs):
        super().__init__(artefact=story_text, **kwargs)
        self.story_text = story_text

    def get_artefact_type(self):
        return "text"

    def validate_artefact(self):
        """Validate that the story is a non-empty string"""
        return isinstance(self.artefact, str) and len(self.artefact.strip()) > 0

    def get_display_text(self):
        """Get formatted text for display"""
        return self.story_text

    def __repr__(self):
        preview = self.story_text[:50] + "..." if len(self.story_text) > 50 else self.story_text
        return f"StoryNode('{preview}')"


class ColorListNode(ArtefactNode):
    """Node class specifically for color list artefacts"""

    def __init__(self, colors: List[List[int]] = None, n: int = 10, **kwargs):
        if colors is None:
            colors = self._generate_random_colors(n)

        # Validate colors during initialization
        self._validate_colors_format(colors)

        super().__init__(artefact=colors, **kwargs)
        self.colors = colors
        self.n_colors = len(colors)

    def get_artefact_type(self):
        return "colors"

    def _generate_random_colors(self, n: int):
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
        """Generate HTML representation of colors (integrated from ColorArtefact)"""
        html = ""
        for color in self.colors:
            html += f"<div style='background-color: rgb({color[0]}, {color[1]}, {color[2]}); width: {width}; padding: 0px; margin-right: 10px'>&nbsp;</div>"
        return f"<div style='display:flex;'>{html}</div>"

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

    def __init__(self, num_colors=10, picker_type="wheel_sliders", bot_response=None):
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


class ChoicePage(ModularPage):
    def __init__(self, label: str, prompt: str, artefacts: list, bot_response):
        super().__init__(
            label,
            Prompt(prompt),
            PushButtonControl(
                choices=artefacts,
                labels=[f"Version {i + 1}" for i in range(len(artefacts))],
                arrange_vertically=True,
                bot_response=bot_response,
            ),
            time_estimate=30,
        )


class CreateTrial(CreateTrialMixin, ImitationChainTrial):
    time_estimate = 5

    def get_current_node(self):
        """Helper method to get the current node - implement based on your trial structure"""
        # This would depend on how you access the current node in your trial system
        # You might need to adapt this based on your actual implementation
        return getattr(self, 'node', None)

    def get_info_page_for_story(self, story_node: StoryNode):
        """Create info page for story reproduction"""
        generation = self.definition.get("generation", 0)

        if generation == 0:
            return InfoPage(
                Markup(
                    f"<h3>Story Reproduction</h3>"
                    f"<p>Please read and memorize this story:</p>"
                    f"<div style='background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50;'>"
                    f"<strong>Original Story:</strong><br>{story_node.get_display_text()}"
                    f"</div>"
                ),
                time_estimate=self.time_estimate,
            )
        else:
            return InfoPage(
                Markup(
                    f"<h3>Story Reproduction - Generation {generation}</h3>"
                    f"<p>Please reproduce the original story.</p>"
                    f"<div style='background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50;'>"
                    f"<strong>Original Story:</strong><br>{story_node.get_display_text()}"
                    f"</div>"
                    f"<div style='background: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107;'>"
                    f"<strong>Last winning story (chosen by a participant):</strong><br>{self.definition['last_choice']}"
                    f"</div>"
                ),
                time_estimate=self.time_estimate,
            )

    def get_info_page_for_colors(self, color_node: ColorListNode):
        """Create info page for color reproduction"""
        generation = self.definition.get("generation", 0)

        if generation == 0:
            return InfoPage(
                Markup(
                    f"<h3>Colors Reproduction</h3>"
                    f"<p>Please read and memorize the following colors:</p>"
                    f"<div style='padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50;'>"
                    f"<strong>Original colors:</strong><br>{color_node.html()}"
                    f"</div>"
                ),
                time_estimate=self.time_estimate,
            )
        else:
            # Use utility function instead of creating a new node
            return InfoPage(
                Markup(
                    f"<h3>Colors Reproduction - Generation {generation}</h3>"
                    f"<p>Please read and memorize the following colors:</p>"
                    f"<div style='padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50;'>"
                    f"<strong>Original colors:</strong><br>{color_node.html()}"
                    f"</div>"
                    f"<div style='padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107;'>"
                    f"<strong>Last winning colors (chosen by a participant):</strong><br>{colors_to_html(self.definition['last_choice'])}"
                    f"</div>"
                ),
                time_estimate=self.time_estimate,
            )

    def first_trial(self):
        print(self.context)

        if self.context["artefact_type"] == "text":
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
        elif self.context["artefact_type"] == "colors":
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
        return None

    def other_trial(self):
        generation = self.definition["generation"]

        if self.context["artefact_type"] == "text":
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
        elif self.context["artefact_type"] == "colors":
            return InfoPage(
                Markup(
                    f"<h3>Colors Reproduction</h3>"
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

        return None

    def input_page(self):
        if self.context["artefact_type"] == "text":
            return StoryInputPage(
                "artefact",
                Markup(
                    f"<h3>Your Task</h3>"
                    f"<p>Please reproduce the story for a peer. They will read multiple proposals and decide which is most likely correct. </p>"
                ),
                time_estimate=120,
                bot_response=lambda: self.context["original"],
            )
        elif self.context["artefact_type"] == "colors":
            return ColorsInputPage(
                "artefact",
                Markup(
                    f"<h3>Your Task</h3>"
                    f"<p>Please reproduce the colors for a peer. They will read multiple proposals and decide which is most likely correct. </p>"
                ),
                time_estimate=120,
                bot_response=lambda: self.context["original"]
            )

        return None

    def show_trial(self, experiment, participant):
        generation = self.definition["generation"]

        if generation == 0:
            info_page = self.first_trial()
        else:
            info_page = self.other_trial()

        input_page = self.input_page()

        return [info_page, input_page]


class SelectTrial(SelectTrialMixin, ImitationChainTrial):
    time_estimate = 5

    def show_trial(self, experiment, participant):
        artefact_type = self.context["artefact_type"]

        if artefact_type == 'text':
            artefacts = self.get_all_targets()
            return ChoicePage(
                "choice",
                Markup(
                    "<h3>Choose the Best Version</h3>"
                    "<p>Please choose the story that seems most faithful to the original:</p>"
                    "<ul>"
                    + "\n".join(
                        [
                            f"<li><strong>Version {i + 1}:</strong> {self.get_target_answer(artefact)}</li>"
                            for i, artefact in enumerate(artefacts)
                        ]
                    )
                    + "</ul>"
                ),
                artefacts=artefacts,
                bot_response=lambda: random.choice(self.targets),
            )
        elif artefact_type == 'colors':
            artefacts = self.get_all_targets()
            for i, artefact in enumerate(artefacts):
                print(self.get_target_answer(artefact))

            return ChoicePage(
                "choice",
                Markup(
                    "<h3>Choose the Best Version</h3>"
                    "<p>Please choose the colors that seem most faithful to the original:</p>"
                    "<ul>"
                    + "\n".join(
                        [
                            f"<li>{colors_to_html(self.get_target_answer(artefact))}</li>"
                            for i, artefact in enumerate(artefacts)
                        ]
                    )
                    + "</ul>"
                ),
                artefacts=artefacts,
                bot_response=lambda: random.choice(self.targets),
            )

        return None


class CreateAndRateTrialMaker(CreateAndRateTrialMakerMixin, ImitationChainTrialMaker):
    pass


# Factory functions for creating nodes
def create_story_nodes(stories_list: List[str]) -> List[StoryNode]:
    """Create story nodes from a list of story texts"""
    return [StoryNode(story) for story in stories_list]


def create_color_nodes(n_colorlists: int, colors_per_list: int = 10) -> List[ColorListNode]:
    """Create color list nodes"""
    return [ColorListNode(n=colors_per_list) for _ in range(n_colorlists)]


# Create nodes using the new classes
story_nodes = create_story_nodes(stories)
color_nodes = create_color_nodes(N_COLORLISTS, 10)

# You can now use either story_nodes, color_nodes, or both
# nodes = story_nodes + color_nodes
nodes = color_nodes

# Validation example
for node in nodes:
    if not node.validate_artefact():
        logger.warning(f"Invalid artefact in node: {node}")

create_and_rate = CreateAndRateTrialMaker(
    n_creators=N_CREATORS_PER_GENERATION,
    n_raters=1,
    node_class=ArtefactNode,  # Base class - actual nodes will be StoryNode or ColorListNode
    creator_class=CreateTrial,
    rater_class=SelectTrial,
    include_previous_iteration=False,
    rate_mode="select",
    target_selection_method="all",
    verbose=True,
    # trial_maker params
    id_="create_and_rate_trial_maker",
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
        create_and_rate,
        SuccessfulEndPage(),
    )
