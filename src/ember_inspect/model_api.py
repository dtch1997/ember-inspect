# ruff: noqa: F401
""" Hacky ModelAPI implementation for Goodfire Ember"""

from typing import Any
import json
import os
from logging import getLogger

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from openai._types import NOT_GIVEN
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params.function_definition import FunctionDefinition
from typing_extensions import override

from inspect_ai._util.constants import DEFAULT_MAX_RETRIES
from inspect_ai._util.content import Content
from inspect_ai._util.error import PrerequisiteError
from inspect_ai._util.images import image_as_data_uri
from inspect_ai._util.logger import warn_once
from inspect_ai._util.url import is_data_uri, is_http_url
from inspect_ai.tool import ToolCall, ToolChoice, ToolFunction, ToolInfo

from inspect_ai.model._chat_message import ChatMessage, ChatMessageAssistant
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._image import image_url_filter
from inspect_ai.model._model import ModelAPI
from inspect_ai.model._model_call import ModelCall
from inspect_ai.model._model_output import (
    ChatCompletionChoice,
    Logprobs,
    ModelOutput,
    ModelUsage,
    StopReason,
)
from inspect_ai.model._providers.util import (
    as_stop_reason,
    environment_prerequisite_error,
    model_base_url,
    parse_tool_call,
)

from inspect_ai.model._providers.openai import OpenAIAPI, as_openai_chat_messages, chat_tools, chat_tool_choice
from .controller import read_controller_params, write_controller_params

# Reference: goodfire.variants.variants; SUPPORTED_MODELS
SUPPORTED_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct"
]

logger = getLogger(__name__)

GOODFIRE_API_KEY = "GOODFIRE_API_KEY"
GOODFIRE_API_KEY_VALUE = os.getenv("GOODFIRE_API_KEY")
assert GOODFIRE_API_KEY_VALUE is not None, "GOODFIRE_API_KEY is not set"

# Changes we need to make
# [X] Change the base URL to Goodfire's URL 
# [X] Change the API key to Goodfire's API key
# [X] Add init arg for setting extra_body (for the controller)
# [ ] Set the extra_body when calling the OAI client

class EmberAPI(OpenAIAPI):
    def __init__(
        self, 
        model_name: str, # can only be one of ember supported models
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any
    ) -> None:
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Model name {model_name} is not supported for Goodfire")
        # Override the base URL and API key
        logger.info("Overriding base URL and API key for Goodfire")
        del base_url
        del api_key
        del api_key_vars
        base_url = "https://api.goodfire.ai/api/inference/v1"
        api_key = GOODFIRE_API_KEY_VALUE
        # Set the controller params
        self.controller_params = read_controller_params()
        super().__init__(model_name, base_url, api_key, config, **model_args)
  
    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput, ModelCall]:
        # short-circuit to call o1- models that are text only
        if self.is_o1_preview() or self.is_o1_mini():
            raise NotImplementedError("O1 models are not supported for Goodfire")

        # setup request and response for ModelCall
        request: dict[str, Any] = {}
        response: dict[str, Any] = {}

        def model_call() -> ModelCall:
            return ModelCall.create(
                request=request,
                response=response,
                filter=image_url_filter,
            )

        # unlike text models, vision models require a max_tokens (and set it to a very low
        # default, see https://community.openai.com/t/gpt-4-vision-preview-finish-details/475911/10)
        OPENAI_IMAGE_DEFAULT_TOKENS = 4096
        if "vision" in self.model_name:
            if isinstance(config.max_tokens, int):
                config.max_tokens = max(config.max_tokens, OPENAI_IMAGE_DEFAULT_TOKENS)
            else:
                config.max_tokens = OPENAI_IMAGE_DEFAULT_TOKENS

        # prepare request (we do this so we can log the ModelCall)
        request = dict(
            messages=await as_openai_chat_messages(input, self.is_o1_full()),
            tools=chat_tools(tools) if len(tools) > 0 else NOT_GIVEN,
            tool_choice=chat_tool_choice(tool_choice) if len(tools) > 0 else NOT_GIVEN,
            extra_body={"controller": self.controller_params},
            **self.completion_params(config, len(tools) > 0),
        )

        try:
            # generate completion
            completion: ChatCompletion = await self.client.chat.completions.create(
                **request
            )

            # save response for model_call
            response = completion.model_dump()

            # parse out choices
            choices = self._chat_choices_from_response(completion, tools)

            # return output and call
            return ModelOutput(
                model=completion.model,
                choices=choices,
                usage=(
                    ModelUsage(
                        input_tokens=completion.usage.prompt_tokens,
                        output_tokens=completion.usage.completion_tokens,
                        total_tokens=completion.usage.total_tokens,
                    )
                    if completion.usage
                    else None
                ),
            ), model_call()
        except BadRequestError as e:
            return self.handle_bad_request(e), model_call()