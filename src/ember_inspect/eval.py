from goodfire import Variant
from inspect_ai import (
    eval, 
    eval_set,
)
from inspect_ai.log import EvalLog
from threading import Lock

import ember_inspect.provider as provider # noqa: F401
from ember_inspect.controller import write_controller_params, reset_controller_params

# Mutex for controller.json
# This is to prevent multiple controllers from writing to controller.json at the same time
# This also means that you can only have one eval or eval_set running at a time
ControllerParamsMutex = Lock()

def eval_variant(
    variant: Variant,
    *args,
    **kwargs,
) -> list[EvalLog]:
    with ControllerParamsMutex:
        write_controller_params(variant.controller.json())
        logs = eval(
            model = "ember/" + variant.base_model,
            *args,
            **kwargs,
        )
        reset_controller_params()
    return logs

def eval_set_variant(
    variant: Variant,
    *args,
    **kwargs,
) -> list[EvalLog]:
    with ControllerParamsMutex:
        write_controller_params(variant.controller.json())
        logs = eval_set(
            model = "ember/" + variant.base_model,
            *args,
            **kwargs,
        )
        reset_controller_params()
    return logs