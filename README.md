# Ember-Inspect

A library for using Goodfire Ember models with Inspect AI.

## Install

```bash
# Install optional dependencies for running examples
pip install ember-inspect[examples]
# Install inspect-evals separately because it's not available on PyPI
pip install "inspect-evals @ git+https://github.com/UKGovernmentBEIS/inspect_evals"
```

## Usage

```python
import os
import dotenv

from inspect_evals.mmlu.mmlu import mmlu_0_shot
from goodfire import Client, Variant
from ember_inspect import eval_variant

# Ensure your GoodFire API key is set
dotenv.load_dotenv(".env")
assert os.getenv("GOODFIRE_API_KEY") is not None, "GOODFIRE_API_KEY is not set"
GOODFIRE_API_KEY = os.getenv("GOODFIRE_API_KEY")

# Create Ember model variant
base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
variant = Variant(base_model = base_model)
client = Client(api_key = GOODFIRE_API_KEY)
edits = client.features.AutoSteer(
    specification = "Think like an expert in college biology",
    model = variant,
)
variant.set(edits)

# Evaluate model variant with Inspect
eval_variant(
    variant = variant,
    tasks = [mmlu_0_shot(subjects=["college_biology"])],
    limit = 5,
)
```

## Examples

See `examples/` for examples of how to use Ember-Inspect.

- `examples/simple`: A simple demo of Ember-Inspect
- `examples/test_time_sae_steering/`: Test-time SAE Steering

## FAQ

Q: My Inspect tasks aren't completing when I use Ember models.
A: This may be due to HTTP rate limits. Wait a minute and try again.
