# # pylint: disable=unused-import,abstract-method,unused-argument

from markupsafe import Markup
import psynet.experiment
from psynet.consent import NoConsent
from psynet.modular_page import Prompt, ModularPage, PushButtonControl, TextControl
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

logger = get_logger()

N_STORIES = 10
N_GENERATIONS = 10

stories = pd.read_csv("stories.csv").sample(n=N_STORIES)["Story"].tolist()


class StoryNode(CreateAndRateNodeMixin, ChainNode):
    def __init__(self, story=None, context=None, **kwargs):
        self.story = story

        super().__init__(
            context={"original": story} if story is not None else context, **kwargs
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

        print(definition)

        return definition


class InputPage(ModularPage):
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


class ChoicePage(ModularPage):
    def __init__(self, label: str, prompt: str, stories: list, bot_response):
        super().__init__(
            label,
            Prompt(prompt),
            PushButtonControl(
                choices=stories,
                labels=[f"Version {i+1}" for i in range(len(stories))],
                arrange_vertically=True,
                bot_response=bot_response,
            ),
            time_estimate=30,
        )


class CreateTrial(CreateTrialMixin, ImitationChainTrial):
    time_estimate = 5

    def show_trial(self, experiment, participant):
        generation = self.definition["generation"]

        if generation == 0:
            info_page = InfoPage(
                Markup(
                    f"<h3>Story Reproduction</h3>"
                    f"<p>Please read and memorize this story:</p>"
                    f"<div style='background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50;'>"
                    f"<strong>Original Story:</strong><br>{self.context['original']}"
                    f"</div>"
                ),
                time_estimate=self.time_estimate,
            )
        else:
            info_page = InfoPage(
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
        input_page = InputPage(
            "artefact",
            Markup(
                f"<h3>Your Task</h3>"
                f"<p>Please reproduce the story for a peer. They will read multiple proposals and decide which is most likely correct. </p>"
            ),
            time_estimate=120,
            bot_response=lambda: self.context["original"],
        )

        return [info_page, input_page]


class SelectTrial(SelectTrialMixin, ImitationChainTrial):
    time_estimate = 5

    def show_trial(self, experiment, participant):
        stories = self.get_all_targets()
        return ChoicePage(
            "choice",
            Markup(
                "<h3>Choose the Best Version</h3>"
                "<p>Please choose the story that seems most faithful to the original:</p><ul>"
                + "\n".join(
                    [
                        f"<li><strong>Version {i+1}:</strong> {self.get_target_answer(story)}</li>"
                        for i, story in enumerate(stories)
                    ]
                )
                + "</ul>"
            ),
            stories=stories,
            bot_response=lambda: random.choice(self.targets),
        )


class CreateAndRateTrialMaker(CreateAndRateTrialMakerMixin, ImitationChainTrialMaker):
    pass


start_nodes = [StoryNode(story) for story in stories]

create_and_rate = CreateAndRateTrialMaker(
    n_creators=2,
    n_raters=1,
    node_class=StoryNode,
    creator_class=CreateTrial,
    rater_class=SelectTrial,
    include_previous_iteration=False,
    rate_mode="select",
    target_selection_method="all",
    verbose=True,
    # trial_maker params
    id_="create_and_rate_trial_maker",
    chain_type="across",
    expected_trials_per_participant=len(start_nodes),
    max_trials_per_participant=len(start_nodes),
    start_nodes=start_nodes,
    chains_per_experiment=len(start_nodes),
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
