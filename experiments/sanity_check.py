# TODO: sanity check that querying through ember gets same perf as querying through vLLM / Huggingface
# eval_set(
#     model = f"vllm/{base_model}",
#     tasks = [task],
#     log_dir = f"logs/{base_model}_mmlu_0_shot",
#     limit = 100,
# )