# Test-time SAE steering

Here we investigate whether we can use SAE feature steering to improve the performance of a model on a task.

1. We consider the MMLU multiiple choice task.
2. We evaluate Llama 3.1 8B and Llama 3.3 70B instruction-tuned models on the task.
3. For each model, we compare the performance of the model before and after steering.

We use the [GoodFire Ember API](https://goodfire.ai) to get the SAE features for a model.
We use the auto-steering API to automatically select SAE features based on the prompt. 
We use the [Inspect AI](https://inspect.ai-safety-institute.org.uk/) library to evaluate the performance of the model on a task.

## Usage

```bash
# Run the steering experiments
python main.py 
# Parse the results and plot figure
python analyse.py
```

## Steering Protocols

Current implemented:

- [x] `zero-shot`: Do nothing, i.e. this is equivalent to running the original model.
- [x] `zero-shot-steering`: Provide a natural language description of the task, and use the Auto-Steering API to find relevant SAE features.
- [x] `few-shot-prompting`: No steering, but give the model few-shot examples when solving the task.

Planned steering protocols:

- [ ] `few-shot-steering`: Give the ICL examples to the Auto-Steering API, which selects SAE features based on the ICL examples.
- [ ] `few-shot-prompting-steering`: ICL examples used for both prompting and steering.

## Results

![plot](plot.png)

At the moment, steering decreases the performance of the model on the task.

It is also possible to look at the completions:

```bash
inspect view --logdir results
```

## Open Questions / Future Work

Why is zero-shot steering worse than no steering? Some hypotheses

- The SAE features selected are not task relevant.
- SAE feature steering interferes with general instruction following.
- SAE feature steering interferes with the specific prompt template used.

It may also be important to optimize the controller hyperparameters.
