# RDMA paper benchmark integration

Based on official inclusionAI/Awex main
bae159256a768d54465ed693d90d925025032a17 (fetched 2026-10-25).

This branch ports the five-model vLLM native-storage adapters from the existing
benchmark fork 420da10587b53e9e76a558608c905018d1464a53 onto current upstream.
They expose writable canonical tensor views, replicated GQA KV-head coordinates,
expert ownership when EP is disabled, MLA post-update reconstruction and GLM MTP
storage. The source metadata respects globally named PP/MTP layers.

The official Mooncake transport, transfer planner, Mooncake writer and Mooncake
reader are unchanged. Run with comm_backend=mooncake and explicit
AWEX_MOONCAKE_PROTOCOL=rdma. Preserve upstream and adapter revision separately.

Why adapters are necessary: the untouched upstream full Qwen30B probe on
2026-10-25 rejected parameter keys (18867 training vs 18771 inference), including
vLLM routed_experts names. That failed attempt is archived under local
results/figure11-four-backends-20261025/evidence/inter-r1-awex-r0-attempt1.
The native two-host CUDA RDMA transport independently passed three repeated-step
updates and strided destination checks, with native remote-access retries kept
in its logs. No TCP or NCCL replacement is used.
