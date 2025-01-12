import os
import json 

from pathlib import Path
from dotenv import load_dotenv

from goodfire import Client, Variant
from inspect_ai import eval_set, Task

from inspect_ai.dataset import Sample
from inspect_evals.mmlu.mmlu import mmlu_0_shot, mmlu_5_shot

import test_time_sae_steering.ember # noqa: F401
from test_time_sae_steering.ember.controller import write_controller_params

curr_dir = Path(__file__).parent
project_dir = curr_dir.parent

load_dotenv(project_dir / ".env")
GOODFIRE_API_KEY = os.getenv("GOODFIRE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = Client(api_key=GOODFIRE_API_KEY)

models = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
]
# Limit in order to avoid rate limits
limit = 50

short_names = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama31_8b_it",
    "meta-llama/Llama-3.3-70B-Instruct": "llama33_70b_it",
}

def evaluate_variant(
    task: Task,
    variant: Variant,
    suffix: str = "default",
    limit: int = limit,
    results_dir: str | Path = curr_dir / "results",
) -> None:
    task_name = task.name.replace("inspect_evals/", "")
    controller_params = variant.controller.json()
    # Set the controller params
    write_controller_params(controller_params)
    log_dir = (results_dir / task_name / suffix / short_names[variant.base_model]).absolute()
    # NOTE: we use eval set because it skips the run if previous results have been completed
    # This saves on API requests
    eval_set(
        model = f"ember/{variant.base_model}",
        tasks = [task],
        log_dir = str(log_dir),
        limit=limit
    )
    # Also save the controller params to a file
    write_controller_params(controller_params, path = log_dir / "controller.json")
    # Also write the metadata
    with open(log_dir / "metadata.json", "w") as f:
        json.dump({
            "suffix": suffix,
            "limit": limit,
        }, f)

def zero_shot(base_model):
    return Variant(base_model = base_model)

def zero_shot_steering(base_model, description: str):
    variant = Variant(base_model = base_model)
    edits = client.features.AutoSteer(
        specification=description,  # Natural language description
        model=variant,  # Model variant to use
    )
    variant.set(edits)
    return variant

def few_shot_steering(base_model, examples: list[str]):
    variant = Variant(base_model = base_model)
    edits = client.features.AutoSteer(
        specification=examples,  # Natural language description
        model=variant,  # Model variant to use
    )
    variant.set(edits)
    return variant

def get_few_shot_examples(task: Task) -> list[Sample]:
    # TODO: check we don't leak any few-shot examples
    return task.dataset.samples[:5]

if __name__ == "__main__":

    for model in models:

        # Zero-shot prompting
        task = mmlu_0_shot(subjects=["high_school_mathematics"])

        # 1. Zero-shot
        variant = zero_shot(model)
        evaluate_variant(task, variant, suffix = "zero-shot", limit = limit)

        # 2. Zero-shot steering
        variant = zero_shot_steering(model, "Solve a high school mathematics question")
        evaluate_variant(task, variant, suffix = "zero-shot-steering", limit = limit)

        # 3. Few-shot steering
        few_shot_examples = get_few_shot_examples(task)
        variant = few_shot_steering(model, few_shot_examples)
        evaluate_variant(task, variant, suffix = "few-shot-steering", limit = limit)

        # 4. Few-shot prompting
        # Few-shot prompting
        task = mmlu_5_shot(subjects=["high_school_mathematics"])
        variant = zero_shot(model)
        evaluate_variant(task, variant, suffix = "few-shot-prompting", limit = limit)

        # 5. Few-shot prompting and steering
        # Few-shot steering 
        variant = few_shot_steering(model, few_shot_examples)
        evaluate_variant(task, variant, suffix = "few-shot-prompting-steering", limit = limit)

