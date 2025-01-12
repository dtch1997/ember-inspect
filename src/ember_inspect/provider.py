from inspect_ai.model import modelapi


@modelapi(name="ember")
def ember():
    from .model_api import EmberAPI

    return EmberAPI
