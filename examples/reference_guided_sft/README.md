# Reference-Guided On-Policy SFT

A hybrid SFT/RL training mode that distills in-context learning into model weights.

## Motivation

Standard SFT trains the model to imitate a fixed supervisor trace. This is
off-policy: the model never produces its own tokens, so it can struggle with
distribution shift at inference time. Standard RL fixes this by generating
on-policy, but requires a reward signal and can be unstable.

**Reference-guided on-policy SFT** sits in between:

1. **Show** the model a reference solution trace in-context (as a "hint").
2. **Generate** the model's own response on-policy, with real tool execution.
3. **Train** using SFT loss on the on-policy trajectory — but **strip the
   reference from the training context**, so the model learns to produce the
   enhanced output *without* needing the reference at inference time.

The intuition: when the model sees the reference in-context, it enters a
"mode" where it produces better solutions (in-context learning). By training
on those improved outputs *without* the reference prompt, we bake that mode
into the model's weights.

## How it works in SLIME

```
┌─────────────────────────────────────────────────────────────────┐
│  GENERATION  (augmented prompt — model sees the reference)      │
│                                                                 │
│  [system + tools] [user: problem + reference trace] [assistant] │
│                          │                                      │
│                          ▼                                      │
│               SGLang inference engine                           │
│               ┌──────────────────────┐                          │
│               │  generate response   │◄──┐                      │
│               │  parse tool calls    │   │ multi-turn loop      │
│               │  execute tools       │───┘                      │
│               └──────────────────────┘                          │
│                          │                                      │
│                          ▼                                      │
│               on-policy response tokens                         │
│               + loss_mask (0 for tool outputs)                  │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  TRAINING  (clean prompt — reference is gone)                   │
│                                                                 │
│  sample.tokens = [clean_prompt_ids] + [response_token_ids]      │
│                                                                 │
│  The model is trained to produce the response given ONLY the    │
│  original problem — no reference. SFT loss with loss_mask.      │
└─────────────────────────────────────────────────────────────────┘
```

The prompt reconstruction happens inside `generate.py`:
- During generation: `augmented_prompt + response` (model sees reference)
- After generation:  `clean_prompt + response` (reference stripped)

Since SLIME's training backend only sees `sample.tokens` and
`sample.response_length`, this swap is transparent to the rest of the system.

## Files

| File | Description |
|---|---|
| `generate.py` | Custom generate function (the core logic) |
| `run_qwen3_4b.sh` | Example launch script for Qwen3-4B on 4×GPU |

## Dataset format

Prepare a `.jsonl` file where each line has:

```json
{
  "prompt": "Find the value of x such that x^2 + 3x - 10 = 0",
  "reference": "Let me solve this step by step...\n<tool_call>\n{\"name\": \"code_interpreter\", \"arguments\": {\"code\": \"import sympy; x = sympy.Symbol('x'); print(sympy.solve(x**2 + 3*x - 10, x))\"}}\n</tool_call>\n\n<interpreter>\n[-5, 2]\n</interpreter>\n\nThe solutions are x = -5 and x = 2.\n\nAnswer: \\boxed{-5, 2}"
}
```

- **`prompt`**: The problem statement (used as `--input-key`)
- **`reference`**: A full solution trace, possibly including tool calls and
  outputs (used as `--label-key`)

## Usage

1. Prepare your dataset:
```bash
# Your dataset should be a .jsonl with "prompt" and "reference" columns
ls /path/to/your/reference_guided_sft_data.jsonl
```

2. Convert model to torch_dist format:
```bash
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen/Qwen3-4B-Instruct-2507 \
    --rotary-base 5000000 \
    --save /root/Qwen/Qwen3-4B-Instruct-2507_torch_dist
```

3. Edit paths in `run_qwen3_4b.sh` and launch:
```bash
bash examples/reference_guided_sft/run_qwen3_4b.sh
```

## Key configuration

The launch script uses **standard SLIME sglang rollout** (not `--debug-train-only`),
because we need the inference engine for on-policy generation. The difference
from RL is:

```bash
# SFT loss — no reward, no advantage computation
--loss-type sft_loss
--calculate-per-token-loss
--disable-compute-advantages-and-returns

# Single sample per prompt (no GRPO-style variance reduction)
--n-samples-per-prompt 1

# Our custom generate function handles reference injection + stripping
--custom-generate-function-path examples.reference_guided_sft.generate.generate
```

## Customization

In `generate.py`, you can adjust:

- **`REFERENCE_METADATA_KEY`**: Set this if the reference trace lives in
  `sample.metadata["some_key"]` instead of `sample.label`.
- **`MAX_TURNS`**: Maximum multi-turn tool-call rounds (default: 16).
- **`USE_TOOL_SANDBOX`**: Set to `False` for pure text generation without
  tool execution.
- **`MASK_TOOL_OUTPUTS`**: Whether to mask tool-output tokens from the SFT
  loss (recommended: `True`).
- **Templates**: Edit `REFERENCE_GUIDED_TEMPLATE` and `CLEAN_TEMPLATE` to
  match your prompt format.

## Comparison with other approaches

| Approach | On-policy? | Uses reference? | Loss | Tool execution? |
|---|---|---|---|---|
| Standard SFT | ❌ | Trains directly on it | NLL | ❌ |
| Standard RL (GRPO) | ✅ | ❌ | Policy gradient | ✅ |
| On-policy distillation | ✅ | Teacher model logprobs | PG with teacher advantage | ❌ |
| **Reference-guided SFT** | **✅** | **In-context only** | **NLL** | **✅** |
