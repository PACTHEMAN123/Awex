# Disaggregated model transfer recipes

Rollout uses vLLM for every model. Training and rollout must occupy different
physical hosts; the GPU limit counts both roles. `models.json` pins each upstream
recipe and the SHA256 of its preserved source script. Those scripts are source
evidence, not launch commands for this repository.

| Model | Training TP/PP/CP/EP/ETP | Training GPUs | vLLM rollout | Rollout GPUs | Total |
|---|---|---:|---|---:|---:|
| Qwen3.5-9B | 2/1/1/1/1 | 4 | 2 instances, TP2 | 4 | 8 |
| Qwen2.5-VL-7B | 2/1/1/1/2 | 4 | 4 instances, TP1 | 4 | 8 |
| GPT-OSS-20B | 4/1/1/8/1 | 8 | 1 instance, TP4 | 4 | 12 |
| GLM-4.7-Flash | 2/2/2/8/1 | 16 | 1 instance, TP1/DP8/EP8 | 8 | 24 |

GLM's upstream script requests eight actor GPUs, but its EP8/PP2 expert layout
requires at least sixteen. Its training topology is retained with sixteen GPUs.
The upstream SGLang TP8/DP-attention8 rollout is replaced, with user approval,
by vLLM TP1/DP8/EP8. Its 20 attention heads cannot use ordinary vLLM TP8.
The trained MTP layer remains part of transfer, along with its draft model's
parameters. MLA absorbed weights are rebuilt after publication.

GPT-OSS requires a dequantized BF16 checkpoint, as in the source recipe. This
branch does not add quantization modes. Its alternating expert gate/up rows,
expert biases, router bias and attention sinks have separate layout rules;
vLLM's expert down bias stays zero outside TP rank zero.

Qwen3.5's selected recipe leaves MTP disabled. Qwen2.5-VL uses the Megatron
Bridge provider's replicated Transformers vision tower and vLLM vision TP1.

Validate a planned placement before launching:

```bash
PYTHONPATH=python:benchmarks python -m shardstream_benchmarks.recipes \
  glm-4.7-flash --training-hosts train0 train1 --rollout-hosts infer0
```

CPU layout and placement checks do not constitute GPU end-to-end qualification.
GPU acceptance requires full weight equality after transfer, generation, and
the selected recipe's actual parallel topology on separate hosts.

Run a recipe against an already prepared Ray cluster, with the source checkout,
container-local runtime and installed ShardStream stage available on each node:

```bash
SHARDSTREAM_RECIPE_REPOSITORY=/path/to/ShardStream \
PYTHONPATH=/path/to/local/stage python -m shardstream_benchmarks.recipe_runner \
  qwen3.5-9b --ray-address HEAD:PORT \
  --training-hosts TRAIN_IP --rollout-hosts ROLLOUT_IP \
  --model-path /path/to/checkpoint --source /path/to/ShardStream \
  --runtime /path/to/local/environment --stage /path/to/local/stage \
  --output /path/to/result.json
```

The runner reserves the recipe GPU counts on separate Ray nodes. It uses the
recipe's training provider settings and vLLM TP/DP/EP configuration, checks all
loaded parameters after three publications, and compares deterministic text
generation before and after publication. It does not submit image requests.
Process groups and logs belong to that invocation; failed runs retain their logs
and a result containing the error. The result qualifies weight transfer and
generation, not an entire RL optimizer/training loop.
