import os
import json 

from pathlib import Path
from dotenv import load_dotenv

from goodfire import Client, Variant
from inspect_ai import eval_set, Task
from inspect_evals.mmlu.mmlu import mmlu_0_shot

import test_time_sae_steering.ember # noqa: F401
from test_time_sae_steering.ember.controller import write_controller_params

curr_dir = Path(__file__).parent
project_dir = curr_dir.parent

load_dotenv(project_dir / ".env")
GOODFIRE_API_KEY = os.getenv("GOODFIRE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

models = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
]

short_names = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama31_8b",
    "meta-llama/Llama-3.3-70B-Instruct": "llama33_70b",
}

# Limit in order to avoid rate limits
limit = 50

def evaluate_variant(
    task: Task,
    variant: Variant,
    suffix: str = "default",
    limit: int = 50,
) -> None:
    controller_params = variant.controller.json()
    # Set the controller params
    write_controller_params(controller_params)
    log_dir = (curr_dir / f"logs/mmlu_0_shot/{suffix}_{short_names[variant.base_model]}").absolute()
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
    

if __name__ == "__main__":

    # Set up the task
    task = mmlu_0_shot(subjects=["high_school_mathematics"])

    for model in models:

        variant = Variant(base_model = model)

        # Evaluate the base model on the task
        evaluate_variant(task, variant, suffix = "default", limit = limit)

        # Get the auto-steering edits
        client = Client(api_key=GOODFIRE_API_KEY)
        edits = client.features.AutoSteer(
            specification="Solve a high school mathematics question",  # Natural language description
            model=variant,  # Model variant to use
        )
        variant.set(edits)

        # Evaluate the autosteered model
        evaluate_variant(task, variant, suffix = "autosteered", limit = limit)
