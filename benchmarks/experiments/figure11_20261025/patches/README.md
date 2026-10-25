The AState patch adds an optional `TRANSFER_ENGINE_RDMA_DEVICES` setting.
The native device selector can otherwise invent `mlx5_bond_*` names after
misparsing an indented NIC legend, and sees host NICs unavailable to the
container. Our deployment supplies two active HCAs selected from the current
process's physical GPU topology. AState's default device count and all buffer
sizes are retained. Transfer, snapshot, registration and completion code is
unchanged. Runtime manifests include the exact patch SHA256.

Apply to official AState 6598daf2f8f493a1538d9cf4b5e2beb3acb80db9 with
`git apply astate-explicit-rdma-devices.patch`. Packaging refuses any other diff.
