"""Read cumulative RDMA port counters without entering CUDA or transport code."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import socket
import subprocess
import time


def node_rdma_counters(root="/sys/class/infiniband"):
    from shardstream.topology import _active_rdma_endpoints

    started_ns = time.time_ns()
    def capture(endpoint):
        device = Path(root) / endpoint.name
        interfaces = list((device / "device/net").iterdir())
        if len(interfaces) != 1:
            raise ValueError(f"Ambiguous HCA/netdev mapping: {endpoint.name}")
        netdev = interfaces[0].name
        before = time.time_ns()
        text = subprocess.check_output(["ethtool", "-S", netdev], text=True, timeout=15)
        raw = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            name, value = line.strip().split(":", 1)
            if name in ("tx_vport_rdma_unicast_bytes", "rx_vport_rdma_unicast_bytes"):
                raw[name] = int(value)
        if len(raw) != 2:
            raise ValueError(f"RDMA byte counters unavailable on {netdev}")
        return endpoint.name + ":" + str(endpoint.port), {
            "tx_bytes": raw["tx_vport_rdma_unicast_bytes"],
            "rx_bytes": raw["rx_vport_rdma_unicast_bytes"],
            "netdev": netdev, "source": "ethtool RDMA unicast bytes",
            "started_ns": before, "ended_ns": time.time_ns(),
        }
    with ThreadPoolExecutor(max_workers=4) as executor:
        ports = dict(executor.map(capture, _active_rdma_endpoints()))
    return {"hostname": socket.gethostname(), "pid": os.getpid(),
            "started_ns": started_ns, "ended_ns": time.time_ns(), "rdma": ports}
