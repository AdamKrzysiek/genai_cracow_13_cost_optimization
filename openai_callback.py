"""Callback Handler that prints to std out."""
from __future__ import annotations

from pydantic import BaseModel


class LLMDebug(BaseModel):
    total_tokens: int
    prompt_tokens: int
    write_tokens: int = 0
    read_tokens: int = 0
    completion_tokens: int
    total_cost: str
    model_name: str

    def __str__(self):
        return (
            f"Total Tokens: {self.total_tokens}\nPrompt Tokens: {self.prompt_tokens}\n"
            f"Read Tokens: {self.read_tokens}\nWrite Tokens: {self.write_tokens}\n "
            f"Completion Tokens: {self.completion_tokens}\nTotal Cost: {self.total_cost}\n"
            f"Model name: {self.model_name}"
        )


MODEL_COST_PER_1K_TOKENS_CUSTOM = {
    "gpt-4o-mini-2024-07-18": 0.00015,
    "gpt-4o-mini-2024-07-18-completion": 0.0006,
    "gpt-4o-mini-2024-07-18-cached": 0.000075,
    "gpt-4o-mini-2024-07-18-batch": 0.000075,
    "gpt-4o-mini-2024-07-18-completion-batch": 0.0003,

    "gpt-4o-2024-11-20": 0.0025,
    "gpt-4o-2024-11-20-completion": 0.010,
    "gpt-4o-2024-11-20-cached": 0.00125,
    "gpt-4o-2024-11-20-batch": 0.00125,
    "gpt-4o-2024-11-20-completion-batch": 0.005,

    "claude-3-5-haiku-20241022": 0.0008,
    "claude-3-5-haiku-20241022-completion": 0.0040,
    "claude-3-5-haiku-20241022-write-cached": 0.0010,
    "claude-3-5-haiku-20241022-read-cached": 0.00008,
    "claude-3-5-haiku-20241022-batch": 0.0004,
    "claude-3-5-haiku-20241022-completion-batch": 0.0020,

    # "claude-3-7-sonnet-20250219":

    "anthropic.claude-3-5-haiku-20241022-v1:0": 0.0008,
    "anthropic.claude-3-5-haiku-20241022-v1:0-completion": 0.0040,
    "anthropic.claude-3-5-haiku-20241022-v1:0-write-cached": 0.0010,
    "anthropic.claude-3-5-haiku-20241022-v1:0-read-cached": 0.00008,
    "anthropic.claude-3-5-haiku-20241022-v1:0-batch": 0.0004,
    "anthropic.claude-3-5-haiku-20241022-v1:0-completion-batch": 0.0020,



}


def get_token_cost_for_model(
    model_name: str, num_tokens: int, is_completion: bool = False, is_batch: bool = False
) -> float:
    """
    Get the cost in USD for a given model and number of tokens.

    Args:
        model_name: Name of the model
        num_tokens: Number of tokens.
        is_completion: Whether the model is used for completion or not.
            Defaults to False.

    Returns:
        Cost in USD.
    """
    if is_completion:
        model_name += "-completion"
    if is_batch:
        model_name += "-batch"

    if model_name in MODEL_COST_PER_1K_TOKENS_CUSTOM:
        return MODEL_COST_PER_1K_TOKENS_CUSTOM[model_name] * (num_tokens / 1000)

    raise ValueError(
        f"Unknown model: {model_name}. Please provide a valid OpenAI model name."
        "Known models are: " + ", ".join(MODEL_COST_PER_1K_TOKENS_CUSTOM.keys())
    )



def get_llm_debug_openai(usage, model_name, is_batch=False) -> LLMDebug:
    token_usage = usage
    completion_tokens = token_usage.get("completion_tokens", 0)
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    cached_tokens = token_usage.get("prompt_tokens_details", 0).get("cached_tokens", 0)

    completion_cost = get_token_cost_for_model(
        model_name, completion_tokens, is_completion=True, is_batch=is_batch
    )
    prompt_cost = get_token_cost_for_model(model_name, prompt_tokens - cached_tokens, is_batch=is_batch)
    cached_cost = get_token_cost_for_model(model_name + "-cached", cached_tokens)
    prompt_cost += cached_cost
    total_cost = completion_cost + prompt_cost
    llm_debug = LLMDebug(
        total_tokens=completion_tokens + prompt_tokens + cached_tokens,
        prompt_tokens=prompt_tokens,
        read_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        total_cost=str(total_cost),
        model_name=model_name
    )
    return llm_debug


def get_llm_debug_anthropic(usage, model_name, is_batch=False) -> LLMDebug:
    token_usage = usage
    completion_tokens = token_usage.get("output_tokens", 0)
    prompt_tokens = token_usage.get("input_tokens", 0)
    cached_write_tokens = token_usage.get("cache_creation_input_tokens", 0)
    cached_read_tokens = token_usage.get("cache_read_input_tokens", 0)

    completion_cost = get_token_cost_for_model(
        model_name, completion_tokens, is_completion=True, is_batch=is_batch
    )
    prompt_cost = get_token_cost_for_model(model_name, prompt_tokens, is_batch=is_batch)
    write_cached_cost = get_token_cost_for_model(model_name + "-write-cached", cached_write_tokens)
    read_cached_cost = get_token_cost_for_model(model_name + "-read-cached", cached_read_tokens)
    prompt_cost += write_cached_cost
    prompt_cost += read_cached_cost
    total_cost = completion_cost + prompt_cost
    llm_debug = LLMDebug(
        total_tokens=completion_tokens + prompt_tokens + cached_write_tokens + cached_read_tokens,
        prompt_tokens=prompt_tokens,
        read_tokens=cached_read_tokens,
        write_tokens=cached_write_tokens,
        completion_tokens=completion_tokens,
        total_cost=str(total_cost),
        model_name=model_name
    )
    return llm_debug

def get_llm_debug_bedrock(usage, model_name, is_batch=False) -> LLMDebug:
    token_usage = usage
    completion_tokens = token_usage.get("outputTokens", 0)
    prompt_tokens = token_usage.get("inputTokens", 0)

    completion_cost = get_token_cost_for_model(
        model_name, completion_tokens, is_completion=True, is_batch=is_batch
    )
    prompt_cost = get_token_cost_for_model(model_name, prompt_tokens, is_batch=is_batch)
    total_cost = completion_cost + prompt_cost
    llm_debug = LLMDebug(
        total_tokens=completion_tokens + prompt_tokens,
        prompt_tokens=prompt_tokens,
        read_tokens=0,
        write_tokens=0,
        completion_tokens=completion_tokens,
        total_cost=str(total_cost),
        model_name=model_name
    )
    return llm_debug