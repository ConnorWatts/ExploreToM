import os
import time
import copy
import dill
import random
import argparse
import pandas as pd
from utils import ModelCallHandler, ModelAnswerEvaluator
from belief_tracker import FullBeliefTracker
from story_structure_infiller import FullStoryStructureInfiller
from compute_statistics import dump_all_infilled_stories, _get_model_shortname

### GENERATE STORY CONTEXT ###

# python story_context_generator.py --num_elements_by_class 6 --num_contexts_to_generate 100

def generate_story_context():

    from story_context_generator import StoryContextGenerator, ObjectStateUpdatesGenerator

    MODEL_NAME = ...
    MODEL_ACCESS_METHOD = ...
    NUM_ELEMENTS_BY_CLASS = ...
    NUM_CONTEXTS_TO_GENERATE = ...


    assert (
            NUM_ELEMENTS_BY_CLASS <= 100 and NUM_CONTEXTS_TO_GENERATE <= 100
        ), "Llama refuses to generate a longer list so for simplicity we cap to 100."

    model_call_handler = ModelCallHandler(MODEL_NAME, MODEL_ACCESS_METHOD)
    story_context_generator = StoryContextGenerator(model_call_handler)
    state_updates_generator = ObjectStateUpdatesGenerator(model_call_handler)
    filepath = story_context_generator.main(
        num_elements_by_class=NUM_ELEMENTS_BY_CLASS,
        num_requested_contexts=NUM_CONTEXTS_TO_GENERATE,
    )
    filepath = state_updates_generator.main(
        filepath, num_contexts_to_fill=NUM_CONTEXTS_TO_GENERATE
    )
    print("Final generated context is in", filepath)

def generate_search():

    from story_structure_searcher import FullStoryStructureSearcher

    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_ACCESS_METHOD = "huggingface-4bit"

    GROUP_N_NEXT_STEPS = 3
    A_STAR_G_VERSION = "acc"
    A_STAR_H_VERSION = "intstory_w_people"
    A_STAR_H_WEIGHT = 0.1
    A_STAR_NEIGHBOUR_PRIORITY = "weight-goal4"
    MAX_NEIGHBORS_TO_EXPLORE = "1_10-3_0.75-5_0.5"
    MODEL_GENERATED_CONTEXTS_FILE = "logs/model_generated_contexts_gpt-4o_n_100_p_6_m_6_r_2_update_object_state_equiv_class.jsonl"
    NUM_STORIES_BY_CONTEXT = 1
    EXPERIMENT_TO_RUN = "search"
    GENERALIZATION_TO_EXPLORE = "store_true"
    BUDGET_PER_STORY = 50
    NUM_STORIES_TOTAL = 10


    model_call_handler = ModelCallHandler(MODEL_NAME, MODEL_ACCESS_METHOD)

    all_story_types = [
        "tomi",
        "tomi+object-state",
        "tomi-object-state",
        "tomi+room-changes",
        "fantom-private",
        "fantom-public",
        #"tomi+info-exchange",
        #"allbutfantom",
        #"all",
    ]
    all_story_types.extend([a + "+asymmetric" for a in all_story_types])

    VALID_STORY_GENERATION_TYPES = []
    if GENERALIZATION_TO_EXPLORE:
        for story_type in all_story_types:
            for num_people in range(2, 7):
                for num_moves in range(num_people, 11):
                    for num_rooms in (
                        [2] if "room" in story_type or "all" in story_type else [1]
                    ):
                        for max_sentences in [15]:
                            VALID_STORY_GENERATION_TYPES.append(
                                (
                                    story_type,
                                    num_people,
                                    num_moves,
                                    num_rooms,
                                    max_sentences,
                                )
                            )
    else:
        for story_type in all_story_types:
            for num_people in range(2, 5):
                for num_moves in range(2, 5):
                    for num_rooms in (
                        [2] if "room" in story_type or "all" in story_type else [1]
                    ):
                        for max_sentences in [15]:
                            VALID_STORY_GENERATION_TYPES.append(
                                (
                                    story_type,
                                    num_people,
                                    num_moves,
                                    num_rooms,
                                    max_sentences,
                                )
                            )
    random.seed(0)
    random.shuffle(VALID_STORY_GENERATION_TYPES)
    random.seed(0)

    full_searcher = FullStoryStructureSearcher(
        model_call_handler,
        group_n_next_steps=GROUP_N_NEXT_STEPS,
        a_star_g_version=A_STAR_G_VERSION,
        a_star_h_version=A_STAR_H_VERSION,
        a_star_h_weight=A_STAR_H_WEIGHT,
        a_star_neighbor_priority=A_STAR_NEIGHBOUR_PRIORITY,
        max_neighbors_to_explore=MAX_NEIGHBORS_TO_EXPLORE,
        model_generated_contexts_file=MODEL_GENERATED_CONTEXTS_FILE,
        num_stories_by_context=NUM_STORIES_BY_CONTEXT,
        experiment_to_run=EXPERIMENT_TO_RUN,
        experiment_variations=[
            "count_peeking_distracted_as_action"
        ],  # , 'verify_final_story_for_state_updates'],
        dir_to_write="logs_debug_dec4_fixed_restrictions",
    )

    for i, (story_type, num_people, num_moves, num_rooms, max_sentences) in enumerate(
        VALID_STORY_GENERATION_TYPES
    ):
        #if (
            #args.i is not None and i % NUM_PARALLEL != args.i % NUM_PARALLEL
        #):  # DIY parallelization :)
            #continue
        full_searcher.main(
            story_type,
            num_people,
            num_moves,
            num_rooms,
            max_sentences,
            BUDGET_PER_STORY,
            NUM_STORIES_TOTAL,
            i,
        )

from huggingface_hub import login
token = os.getenv("HF_TOKEN")
login(token=token)

generate_search()



"""model_name = "gpt-4o-mini"
model_access_method = "openai-api"
NUM_PARALLEL = 1 
num_stories_total = 5
generate_fantom_like_data = "story_true"
i = 0
NUM_STORIES_CONST = 5
BUDGET = 50
logs_directory = "logs_debug_dec4_fixed_restrictions"
model_shortname = _get_model_shortname(model_name)

infilled_filename = (
    f"analyses/infilled_dec4_all_{model_shortname}_n_{NUM_STORIES_CONST}.csv"
)
if os.path.exists(infilled_filename):
    df_infilled = pd.read_csv(infilled_filename, index_col=0)
else:
    df_infilled = dump_all_infilled_stories(
        ["_v3", model_shortname, f"_n_{NUM_STORIES_CONST}_"]
        if NUM_STORIES_CONST > 0
        else [model_shortname],
        ["_fanlike", "_v2"],
        "infilled_with_llm_judge_w_goals_and_initial_debug_dec4_fixed_restrictions",
        infilled_filename,
    )
    infilled_filename2 = f"analyses/infilled_dec4_all_{model_shortname}_n_{NUM_STORIES_CONST}_fantom_like_data.csv"
    _ = dump_all_infilled_stories(
        ["_fanlike", "_v3", model_shortname, f"_n_{NUM_STORIES_CONST}_"]
        if NUM_STORIES_CONST > 0
        else [model_shortname],
        ["_v2"],
        "infilled_with_llm_judge_w_goals_and_initial_debug_dec4_fixed_restrictions",
        infilled_filename2,
    )"""


"""# Prerequisite: find challenging story structures and dump them in logs_debug_nov7_fixed_restrictions
model_call_handler = ModelCallHandler(model_name, model_access_method)

if generate_fantom_like_data:
    full_infiller = FullStoryStructureInfiller(
        model_call_handler,
        sample_all_next_step_completions_simultaneously=True,
        experiment_variations=["fantom_like_data"],
    )
else:
    full_infiller = FullStoryStructureInfiller(
        model_call_handler, sample_all_next_step_completions_simultaneously=True
    )
full_infiller.main(
    logs_directory="logs_debug_dec4_fixed_restrictions",
    NUM_STORIES=num_stories_total,
    NUM_PARALLEL=NUM_PARALLEL,
    parallel_idx=i,
    )"""