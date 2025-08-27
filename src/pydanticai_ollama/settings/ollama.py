"""
Ollama model settings and configurations.

This module provides the OllamaModelSettings class which extends ModelSettings to include
Ollama-specific model parameters and configurations for fine-tuning model behavior.

Classes:
    OllamaModelSettings: Settings class for configuring Ollama model parameters.
"""

from typing import Literal, List, Optional, Union
from pydantic_ai.settings import ModelSettings


class OllamaModelSettings(ModelSettings, total=False):
    """
    Settings for Ollama models, encapsulating various parameters
    that control the model's generation process.
    """

    temperature: float
    """The temperature of the model. Increasing the temperature will
    make the model answer more creatively. (Default: ``0.8``)"""

    num_predict: int
    """Maximum number of tokens to predict when generating text.
    (``-1`` = infinite generation, ``-2`` = fill context)"""

    top_k: int
    """Reduces the probability of generating nonsense. A higher value (e.g. ``100``)
    will give more diverse answers, while a lower value (e.g. ``10``)
    will be more conservative. (Default: ``40``)"""

    top_p: float
    """Works together with top-k. A higher value (e.g., ``0.95``) will lead
    to more diverse text, while a lower value (e.g., ``0.5``) will
    generate more focused and conservative text. (Default: ``0.9``)"""

    repeat_last_n: int
    """Sets how far back for the model to look back to prevent
    repetition. (Default: ``64``, ``0`` = disabled, ``-1`` = ``num_ctx``)"""

    repeat_penalty: float
    """Sets how strongly to penalize repetitions. A higher value (e.g., ``1.5``)
    will penalize repetitions more strongly, while a lower value (e.g., ``0.9``)
    will be more lenient. (Default: ``1.1``)"""

    seed: int
    """Sets the random number seed to use for generation. Setting this
    to a specific number will make the model generate the same text for
    the same prompt."""

    stop: Union[str, List[str]]
    """A list of strings to stop generation at. 
    The model will stop generating tokens when it encounters any of these strings."""

    tfs_z: float
    """Tail free sampling is used to reduce the impact of less probable
    tokens from the output. A higher value (e.g., ``2.0``) will reduce the
    impact more, while a value of 1.0 disables this setting. (default: ``1``)"""

    num_gpu: int
    """The number of GPUs to use. 
    On macOS it defaults to ``1`` to enable metal support, ``0`` to disable."""

    main_gpu: int
    """The GPU to use for the main computation. A value of 0 uses the first GPU."""

    num_thread: int
    """Sets the number of threads to use during computation.
    By default, Ollama will detect this for optimal performance.
    It is recommended to set this value to the number of physical
    CPU cores your system has (as opposed to the logical number of cores)."""

    num_ctx: int
    """The size of the context window. 
    A higher value (e.g., 2048) will allow the model to consider more of the past conversation, 
    while a lower value (e.g., 512) will be more focused."""

    mirostat: int
    """Enable Mirostat sampling for controlling perplexity. 
    0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0."""

    mirostat_eta: float
    """Influences how quickly the algorithm responds to feedback
    from the generated text. 
    A lower learning rate will result in
    slower adjustments, while a higher learning rate will make
    the algorithm more responsive. (Default: ``0.1``)"""

    mirostat_tau: float
    """Controls the balance between coherence and diversity
    of the output. A lower value will result in more focused and
    coherent text. (Default: ``5.0``)"""

    think: Optional[Union[bool, Literal["low", "medium", "high"]]]
    """Controls the reasoning/thinking mode for supported models: https://ollama.com/search?c=thinking
    True: Enables reasoning mode. The main response content will not include the reasoning tags.
    False: Disables reasoning mode. The model will not perform any reasoning, and the response will not include any reasoning content.
    None (Default): The model will use its default reasoning behavior. Note however, if the model’s default behavior is to perform reasoning, think tags ()``<think>`` and </think>) will be present within the main response content unless you set reasoning to True.
    str: e.g. 'low', 'medium', 'high'. Enables reasoning with a custom intensity level. Currently, this is only supported gpt-oss. See the Ollama docs for more information."""

    keep_alive: Optional[Union[float, str]]
    """How long the model will stay loaded into memory."""
