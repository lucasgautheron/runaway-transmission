# # pylint: disable=unused-import,abstract-method,unused-argument

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
N_STORIES = 10
N_COLORLISTS = 1
N_GENERATIONS = 1

stories = pd.read_csv("stories.csv").sample(n=N_STORIES)["Story"].tolist()


class ColorArtefact:
    def __init__(self, colors: List[List[int]] = None, n: int = 10):
        if colors is None:
            colors = self.draw(n)

        assert all([
            (len(color) == 3) for color in colors
        ])
        assert all([
            all([(c >= 0) and (c <= 255) for c in color]) for color in colors
        ])

        self.colors = colors

    def draw(self, n: int = 10):
        return np.random.randint(0, 255 + 1, size=(n, 3))

    def html(self, width="50px"):
        html = ""

        for color in self.colors:
            html += f"<div style='background-color: rgb({color[0]}, {color[1]}, {color[2]}); width: {width}; padding: 0px; margin-right: 10px'>&nbsp;</div>"

        return f"<div style='display:flex;'>{html}</div>"


class ArtefactNode(CreateAndRateNodeMixin, ChainNode):
    def __init__(self, artefact=None, artefact_type: str = None, context=None, **kwargs):
        self.artefact = artefact
        self.artefact_type = artefact_type

        super().__init__(
            context={"original": artefact, "artefact_type": self.artefact_type} if artefact is not None else context,
            **kwargs
        )

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
        str2target = {target.answer: target for target in all_targets}
        count_dict = {target_str: 0 for target_str in str2target.keys()}

        for trial in trials:
            if trial.answer in count_dict:
                count_dict[trial.answer] += 1

        target_str_with_highest_count = max(count_dict, key=count_dict.get)

        definition = self.seed.copy()
        definition["generation"] += 1
        definition["last_choice"] = target_str_with_highest_count

        return definition


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
        if not isinstance(raw_answer, dict):
            return "INVALID_RESPONSE"

        colors = raw_answer.get('selected_colors', [])
        interactions = raw_answer.get('interactions', [])

        return {
            'selected_colors': [self._validate_color(c) for c in colors],
            'selection_duration': raw_answer.get('total_time', 0),
            'color_change_events': len(interactions),
            'interaction_log': interactions
        }

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
        if not isinstance(response.answer, dict):
            return FailedValidation("Invalid response format.")
        colors = response.answer.get('selected_colors', [])
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
            color_artefact = ColorArtefact(self.context["original"])

            return InfoPage(
                Markup(
                    f"<h3>Colors Reproduction</h3>"
                    f"<p>Please read and memorize the following colors:</p>"
                    f"<div style='padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50;'>"
                    f"<strong>Original colors:</strong><br>{color_artefact.html()}"
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
            color_artefact = ColorArtefact(self.context["original"])
            last_color_artefact = ColorArtefact(self.definition["last_choice"])

            return InfoPage(
                Markup(
                    f"<h3>Colors Reproduction</h3>"
                    f"<p>Please read and memorize the following colors:</p>"
                    f"<div style='padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50;'>"
                    f"<strong>Original colors:</strong><br>{color_artefact.html()}"
                    f"</div>"
                    f"<div style='padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107;'>"
                    f"<strong>Last winning colors (chosen by a participant):</strong><br>{last_color_artefact.html()}"
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

            return ChoicePage(
                "choice",
                Markup(
                    "<h3>Choose the Best Version</h3>"
                    "<p>Please choose the story that seems most faithful to the original:</p>"
                    "<ul>"
                    + "\n".join(
                        [
                            f"<li>{ColorArtefact(self.get_target_answer(artefact)['selected_colors']).html()}</li>"
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


story_nodes = [ArtefactNode(artefact=story, artefact_type="text") for story in stories]
color_nodes = [
    ArtefactNode(ColorArtefact(n=10).colors, artefact_type="colors") for i in range(N_COLORLISTS)
]

# nodes = story_nodes + color_nodes
nodes = color_nodes

create_and_rate = CreateAndRateTrialMaker(
    n_creators=N_CREATORS_PER_GENERATION,
    n_raters=1,
    node_class=ArtefactNode,
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
