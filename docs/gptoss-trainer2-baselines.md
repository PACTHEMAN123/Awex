# GPT-OSS trainer-two-node baseline layouts

The original GPT-OSS layouts allocate one trainer node and three rollout nodes.
The user's Awex restriction requires two trainer nodes and one or two rollout
nodes. Two additional fixed layouts cover the missing model under that
restriction:

| Layout | Trainer | Rollout | GPUs |
|---|---|---|---:|
| `gptoss16-r2tp4` | 16 GPUs, TP4/EP8 | 2 TP4 instances on one node | 24 |
| `gptoss16-r4tp4` | 16 GPUs, TP4/EP8 | 4 TP4 instances on two nodes | 32 |

Keep the original GPT-OSS BF16 checkpoint, training and inference precision
rules, publication protocols, bucket sizes and network parameters. Run
ShardStream, original Awex one-by-one NCCL, original veRL NCCL, NIXL and
Mooncake on these identical configurations. Each method/layout first needs
three full qualification versions, then three independent launches with five
warmup and thirty measured versions.

These thirty launches supplement the original 225-slot experiment matrix.
They must remain separately identifiable, preserve every full parameter and
generation check and raw node archive, and record actual HCA use. The original
three-rollout-node Awex cases remain deferred under the current user override.
Configuration validation is not evidence that GPU qualification has passed.
