# Test-time SAE steering

Here we investigate whether we can use SAE feature steering to improve the performance of a model on a task.

1. We consider the MMLU 0-shot multiiple choice task.
2. We evaluate Llama 3.1 8B and Llama 3.3 70B instruction-tuned models on the task.
3. For each model, we compare the performance of the model before and after steering.

We use the [GoodFire Ember API](https://goodfire.ai) to get the SAE features for a model.
We use the auto-steering API to automatically select SAE features based on the prompt. 
We use the [Inspect AI](https://inspect.ai-safety-institute.org.uk/) library to evaluate the performance of the model on a task.

## Results

![plot](experiments/plot.png)

At the moment, steering decreases the performance of the model on the task.