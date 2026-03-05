"""
Reference-Guided On-Policy SFT (custom generate function for SLIME).

This implements a hybrid SFT/RL training mode:

1. The model is shown a "reference" solution trace in-context and asked to
   solve the problem on its own, with tool use enabled.
2. Tools are actually executed (as in RL rollout), producing an on-policy
   trajectory that benefits from in-context learning of the reference.
3. After generation, the reference context is **stripped** from the training
   sequence — the model is trained (via SFT loss) to reproduce the on-policy
   response given *only* the original prompt, without the reference crutch.

This distills "in-context learned wisdom" into the model's weights.

Usage:
    --custom-generate-function-path examples.reference_guided_sft.generate.generate

The dataset should have:
    - input_key  : the user problem/question (str)
    - label_key  : the reference solution trace (str)

Optionally, the reference trace can live in a metadata field instead (see
REFERENCE_METADATA_KEY below).
"""

import re
from typing import Any, Dict, List, Optional

try:
    from jinja2 import Template
except ImportError:
    raise ImportError("Jinja2 is required. Please install it with: pip install jinja2")

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# If the reference trace is stored in sample.metadata under a specific key
# instead of sample.label, set this. Otherwise leave as None to use sample.label.
REFERENCE_METADATA_KEY: Optional[str] = None

# Maximum number of multi-turn tool-call rounds.
MAX_TURNS = 16

# Whether to import and use the retool sandbox for tool execution.
# Set to False if you want pure text generation (no tool execution).
USE_TOOL_SANDBOX = True

# When True, mask tool-output tokens from the SFT loss (recommended).
MASK_TOOL_OUTPUTS = True

if USE_TOOL_SANDBOX:
    try:
        from examples.retool.tool_sandbox import SEMAPHORE, tool_registry
    except ImportError:
        # Fallback: try importing from the retool directory directly
        # (in case PYTHONPATH includes the examples/retool directory)
        from tool_sandbox import SEMAPHORE, tool_registry


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Jinja2 template that wraps the original problem with a reference trace and
# asks the model to produce its own solution.  The key design choice: the
# reference is shown *before* the assistant turn so that the model can leverage
# it during generation, but the entire reference preamble is later stripped from
# the training sequence.

REFERENCE_GUIDED_TEMPLATE = """\
<|im_start|>system
You are a helpful assistant that solves problems step by step.\
{% if tools %}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{ tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{% endif -%}
<|im_end|>
<|im_start|>user
{{ problem }}

Here is a reference solution for your information:
<reference>
{{ reference_trace }}
</reference>

Now solve this problem yourself. You may use tools if helpful. \
Show your work step by step and give your final answer as: Answer: \\boxed{answer}<|im_end|>
<|im_start|>assistant
"""

# The "clean" template used to produce the training-time prompt (without the
# reference).  This is what the model will see at inference time.

CLEAN_TEMPLATE = """\
<|im_start|>system
You are a helpful assistant that solves problems step by step.\
{% if tools %}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{ tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{% endif -%}
<|im_end|>
<|im_start|>user
{{ problem }}

Solve this problem step by step. You may use tools if helpful. \
Show your work and give your final answer as: Answer: \\boxed{answer}<|im_end|>
<|im_start|>assistant
"""


def _render(template_str: str, **kwargs) -> str:
    return Template(template_str).render(**kwargs)


# ---------------------------------------------------------------------------
# Tool-call helpers  (mirrors retool logic)
# ---------------------------------------------------------------------------

def postprocess_response(resp: str) -> str:
    """Trim response to the last complete action."""
    # <tool_call>...</tool_call>
    if "<tool_call>" in resp:
        pat = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
        matches = list(re.finditer(pat, resp, re.DOTALL))
        if matches:
            return resp[: matches[-1].end()]

    # Answer: \boxed{...}
    if "Answer:" in resp and "\\boxed{" in resp:
        pat = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
        matches = list(re.finditer(pat, resp, re.DOTALL))
        if matches:
            return resp[: matches[-1].end()]

    return resp


def parse_action(prediction: str):
    """Return (action_type, content) from a model response chunk."""
    # Answer
    answer_pat = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    m = re.search(answer_pat, prediction, re.DOTALL)
    if m:
        return "answer", m.group(1).strip()

    # <tool_call> JSON
    tc_pat = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    m = re.search(tc_pat, prediction, re.DOTALL)
    if m:
        import json as _json
        try:
            obj = _json.loads(m.group(1).replace("\n", "\\n"))
            if obj.get("name") == "code_interpreter":
                code = obj.get("arguments", {}).get("code", "").strip()
                if code:
                    return "code", code
        except (_json.JSONDecodeError, KeyError, AttributeError):
            pass

    # ```python ... ```
    m = re.search(r"```python\s*(.*?)\s*```", prediction, re.DOTALL)
    if m:
        return "code", m.group(1).strip()

    return None, ""


async def execute_code(code: str) -> str:
    """Execute code in the retool sandbox and return the observation string."""
    if not USE_TOOL_SANDBOX:
        return "\n\n<interpreter>\nError: tool execution disabled\n</interpreter>\n\n"

    async with SEMAPHORE:
        result = await tool_registry.execute_tool("code_interpreter", {"code": code})
    return f"\n\n<interpreter>\n{result}\n</interpreter>\n\n"


# ---------------------------------------------------------------------------
# Main generate function  (plugged in via --custom-generate-function-path)
# ---------------------------------------------------------------------------

async def generate(args, sample: Sample, sampling_params) -> Sample:
    """
    Reference-guided on-policy generation with prompt reconstruction.

    1. Build an augmented prompt containing the reference trace.
    2. Generate a multi-turn response (with tool execution).
    3. Reconstruct sample.tokens by replacing the augmented prompt prefix
       with the *clean* prompt (no reference), so that training sees only
       the original problem + the model's own response.
    """
    assert not getattr(args, "partial_rollout", False), (
        "partial_rollout is not supported for reference-guided SFT"
    )

    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # ------------------------------------------------------------------
    # 1.  Extract the reference trace
    # ------------------------------------------------------------------
    if REFERENCE_METADATA_KEY is not None:
        reference_trace = sample.metadata.get(REFERENCE_METADATA_KEY, "")
    else:
        reference_trace = sample.label or ""

    if not reference_trace:
        # No reference available — fall back to vanilla generation.
        # The model just sees the clean prompt.
        reference_trace = "(no reference provided)"

    # ------------------------------------------------------------------
    # 2.  Build both prompt variants
    # ------------------------------------------------------------------
    # Determine available tools (if any).
    tool_specs: List[Dict[str, Any]] = []
    if USE_TOOL_SANDBOX:
        try:
            tool_specs = tool_registry.get_tool_specs()
        except Exception:
            pass

    template_kwargs = dict(
        problem=sample.prompt,
        reference_trace=reference_trace,
        tools=tool_specs,
    )

    augmented_prompt = _render(REFERENCE_GUIDED_TEMPLATE, **template_kwargs)
    clean_prompt = _render(CLEAN_TEMPLATE, **{k: v for k, v in template_kwargs.items() if k != "reference_trace"})

    augmented_prompt_ids = tokenizer(augmented_prompt, add_special_tokens=False)["input_ids"]
    clean_prompt_ids = tokenizer(clean_prompt, add_special_tokens=False)["input_ids"]

    # ------------------------------------------------------------------
    # 3.  Multi-turn generation with the augmented prompt
    # ------------------------------------------------------------------
    response_text = ""
    response_token_ids: list[int] = []
    loss_masks: list[int] = []

    for turn in range(MAX_TURNS):
        payload = {
            "text": augmented_prompt + response_text,
            "sampling_params": sampling_params,
        }
        output = await post(url, payload)

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            sample.reward = 0
            return sample

        cur_response = postprocess_response(output["text"])
        cur_token_ids = tokenizer(cur_response, add_special_tokens=False)["input_ids"]

        response_text += cur_response
        response_token_ids += cur_token_ids
        loss_masks += [1] * len(cur_token_ids)  # model-generated → trainable

        # Length limit
        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        # Parse action
        action_type, content = parse_action(cur_response)

        if action_type == "answer":
            break
        elif action_type == "code":
            observation = await execute_code(content)
            obs_token_ids = tokenizer(observation, add_special_tokens=False)["input_ids"]
            response_text += observation
            response_token_ids += obs_token_ids
            if MASK_TOOL_OUTPUTS:
                loss_masks += [0] * len(obs_token_ids)  # tool output → masked
            else:
                loss_masks += [1] * len(obs_token_ids)
        else:
            # Invalid action — nudge the model
            nudge = (
                "\nMy previous action is invalid. "
                "If I want to execute code, I should use <tool_call> tags. "
                "If I want to give the final answer, I should use "
                "'Answer: \\boxed{answer}'. Let me try again.\n"
            )
            nudge_ids = tokenizer(nudge, add_special_tokens=False)["input_ids"]
            response_text += nudge
            response_token_ids += nudge_ids
            loss_masks += [0] * len(nudge_ids)  # system nudge → masked

    # ------------------------------------------------------------------
    # 4.  Reconstruct training sequence: clean_prompt + response
    # ------------------------------------------------------------------
    # This is the critical step: the training sequence does NOT contain the
    # reference trace.  The model is trained to produce the on-policy
    # response given only the original prompt.
    sample.tokens = clean_prompt_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response_text
    sample.loss_mask = loss_masks

    # No reward — this is SFT, not RL.
    sample.reward = 0

    # Set status
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample
