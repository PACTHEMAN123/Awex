# `nccl_device_v2` 与标准 NCCL 的跨机实现差异

本文整理分析中的第 2、3、5 点，对比 Awex `nccl_device_v2` 与标准 `nccl_comm`
所使用的 NCCL P2P 路径，并记录本轮跨机对齐后的状态。不把尚未通过跨机 profiling
验证的推测写成结论。

对比基线：

- Awex：`codex/nccl-device-gin` 分支中的 `nccl_device_v2`，实现以本文所在 commit 为准；
- 标准 NCCL：工作区 `comm/nccl`，commit `fd16832`；
- 场景：跨节点 P2P send/recv，Awex v2 使用 GIN，`nccl_comm` 使用 NCCL 常规 NET transport。

## 差异概览

| 对比项 | Awex `nccl_device_v2` | 标准 NCCL | 主要影响 |
| --- | --- | --- | --- |
| 2. 跨机 channel 分片公式 | GIN 已改为多节点公式：`step / 2` 到 `step` | 多节点使用 `step / 2` 到 `step` | 已对齐 |
| 3. 默认流水线粒度 | LSA 512 KiB；GIN 128 KiB；8 层 FIFO；4 MiB work chunk | 跨机 P2P 默认 128 KiB，并按消息大小和协议调节 | GIN step 和 SIMPLE 小消息调节已对齐；work chunk 仍是 Awex lowering 层概念 |
| 5. 网络执行上下文 | connection 由 active RDMA netdev 探测；context 和 channel 根据实际 connection 数、peer 数及负荷自动确定 | 常规 NET 按 NIC、带宽、channel 和 flow 调度 | 默认路径无需手工指定并行度 |

## 2. 跨机使用多节点 SIMPLE 分片公式

### Awex `nccl_device_v2`

修改前，`v2ChannelsForBytes()` 对所有 transport 固定使用节点内范围：

```text
min_part_bytes = step_bytes / 8
max_part_bytes = step_bytes * 32
```

当前 lowering 会读取每个 peer 的 transport。LSA/NVLink 保留上述范围，GIN 改为：

```text
min_part_bytes = network_step_bytes / 2
max_part_bytes = network_step_bytes
```

默认 `network_step_bytes = 128 KiB`，对应：

```text
min_part_bytes = 64 KiB
max_part_bytes = 128 KiB
```

源码：

- [`device_v2_lowering.h`](../awex/transfer/nccl_device_v2/device_v2_lowering.h)：
  `v2ChannelsForBytes()` 以及调用该函数的 peer stream lowering；
- [`nccl_device_v2.py`](../awex/transfer/nccl_device_v2.py)：默认 `network_step_bytes`。

### 标准 NCCL

NCCL 的 `addP2pToPlan()` 会根据 communicator 是否跨节点选择不同公式：

```text
单节点: minPartSize = step / 8, maxPartSize = step * 32
多节点: minPartSize = step / 2, maxPartSize = step
```

跨机默认 `p2pChunkSize` 为 128 KiB，因此初始跨机范围为：

```text
minPartSize = 64 KiB
maxPartSize = 128 KiB
```

源码：

- [`enqueue.cc`](../../comm/nccl/src/enqueue/enqueue.cc)：`addP2pToPlan()` 中的多节点分支；
- [`init.cc`](../../comm/nccl/src/init.cc)：`P2P_NET_CHUNKSIZE` 默认值与 `p2pChunkSize` 初始化。

### 当前对齐状态

GIN 与标准 NCCL 现在使用相同的默认 `minPartSize=64 KiB`、`maxPartSize=128 KiB`。
实际 channel 数仍受 per-peer channel 上限约束。v2 在 GIN 初始化完成后读取
`ginConnectionCount`，先按 peer payload 是否足以填满一个 FIFO window 分配 1/2/4/8 个
基础 channel（上限为每个 connection 两个），再以 `6 * ginConnectionCount` 作为共享 issue
预算。只有 payload 占比足以覆盖下一档 2 次幂且共享预算仍有余量的 peer 才会升级；单 peer
可以独占向上取整后的预算。这样小控制流不会和 GB 级权重流占用相同资源，多 peer 也不会因
各自向上取整而超出总预算。由于公开 Device API 没有直接暴露标准 NCCL 内部的 NIC 总带宽，
`ceil(netBw / 14 GB/s)` 这一项仍无法直接复刻。
如需使用标准 NCCL 计算出的更大值，可以设置 `NCCL_NCHANNELS_PER_NET_PEER`；v2 会继承它，
`AWEX_NCCL_DEVICE_V2_NET_CHANNELS_PER_PEER` 则提供优先级更高的单后端覆盖。显式值同样会向上取
2 次幂并受总 channel 上限约束。

## 3. 默认 step、FIFO 与 work chunk 不同

### Awex `nccl_device_v2`

当前默认配置为：

```text
max_channels = 64
fifo_depth   = 8
local_step_bytes   = 512 KiB
network_step_bytes = 128 KiB
chunk_bytes  = 4 MiB
```

其中：

- `max_channels` 是总 channel 上限，不代表跨机 peer 实际会得到 64 个 channel；
- `local_step_bytes` 是 LSA/NVLink 的传输 step；
- `network_step_bytes` 是 GIN 的规划 step，也是大消息单次 GIN put 的最大 slice；
- `fifo_depth` 表示每个 `(peer, channel)` 有 8 个循环使用的 slot；
- `chunk_bytes` 是 lowering 生成一个 `V2Work` 的默认上限。一个 4 MiB work 在 LSA 路径包含
  8 个 512 KiB step，在 GIN 大消息路径包含 32 个 128 KiB step。

window 的 slot 按两种 step 的较大值分配，因此默认每个 `(peer, channel)` 的 payload window 仍为：

```text
8 * 512 KiB = 4 MiB
```

源码：

- [`nccl_device_v2.py`](../awex/transfer/nccl_device_v2.py)：`max_channels`、`fifo_depth`、
  local/network step 和 `chunk_bytes` 默认值；
- [`device_v2_gin.cuh`](../awex/transfer/nccl_device_v2/device_v2_gin.cuh)：按 work step 切分，
  执行 GIN put，以及 FIFO credit 控制；
- [`nccl_device_v2_ext.cu`](../awex/transfer/nccl_device_v2/nccl_device_v2_ext.cu)：
  `payload_buffer_bytes` 的实际计算方式。

### 标准 NCCL

标准 NCCL 同样定义 `NCCL_STEPS = 8`，跨机 P2P 默认
`P2P_NET_CHUNKSIZE = 128 KiB`。规划时还会：

- 根据 payload 大小选择 LL 或 SIMPLE protocol；
- 对网络小消息把 chunk 缩小到原来的 `1/4` 或 `1/2`；
- 结合节点数、protocol buffer 和用户配置计算最终 step/chunk。

标准 NCCL 的 SIMPLE buffer 默认总大小为 4 MiB，但它与 v2 的 `chunk_bytes=4 MiB` 不是同一个
概念：前者是 NCCL protocol buffer 配置，后者是 Awex host lowering 生成 `V2Work` 的粒度。

源码：

- [`device.h`](../../comm/nccl/src/include/device.h)：`NCCL_STEPS = 8`；
- [`init.cc`](../../comm/nccl/src/init.cc)：4 MiB SIMPLE buffer 和 128 KiB 跨机 P2P chunk；
- [`enqueue.cc`](../../comm/nccl/src/enqueue/enqueue.cc)：protocol 选择与网络 chunk 调节。

### 当前对齐状态

v2 的 GIN 默认 step 已改为 128 KiB。完成 channel 选择后，它也复刻 SIMPLE 网络小消息调节：
payload 小于一个 step 时使用 `step / 4`，小于八个 step 时使用 `step / 2`。实际 step 写入每个
`V2Work`，lowering 的 step 计数与 kernel 的 put/credit 推进使用同一个值。

`AWEX_NCCL_DEVICE_V2_NET_STEP_BYTES` 可以显式覆盖该值；未设置时会读取
`NCCL_P2P_NET_CHUNKSIZE`，再回退到 128 KiB。

需要特别避免把 `4 MiB` 对 `4 MiB` 当成“配置相同”：v2 的 4 MiB 是 work chunk，标准 NCCL 的
4 MiB 是 SIMPLE protocol buffer，两者处在不同层次。

## 5. GIN context 与网络 connection/channel

### Awex `nccl_device_v2`

创建 device communicator 时，v2 默认请求：

```text
ginContextCount = detectedGinConnectionCount
```

执行时每个 channel 选择：

```text
context = channel % ginContextCount
```

默认结果是每个 connection 一个 context。这里由 Awex 显式传入数量，因为 NCCL 2.30.4 不会把单个
context 请求向上取整到 connection 数；新版 NCCL 即使支持 round-up，也得到相同结果。
connection 数由 Awex 在 communicator 创建前统计 sysfs 中 active 且带 netdev 的 RDMA 设备，最多使用
4 个 GIN connection slot；sysfs 不可见时回退到 4。显式设置为 0 才使用 NCCL 的原生 local-device
discovery。kernel 始终使用返回的 `dev_comm.ginContextCount` 做 channel 取模。两个请求值都可由环境变量覆盖。

源码：

- [`nccl_device_v2_ext.cu`](../awex/transfer/nccl_device_v2/nccl_device_v2_ext.cu)：
  `ncclDevCommRequirements.ginContextCount`；
- [`device_v2_gin.cuh`](../awex/transfer/nccl_device_v2/device_v2_gin.cuh)：send/recv channel 到
  GIN context 的取模映射。

### 标准 NCCL

常规 `nccl_comm` 跨机 P2P 走 NCCL NET transport，不使用 v2 这一层 GIN context。NCCL 会：

- 根据 NIC 数和本地网络带宽计算 remote peer 的 channel 需求，默认至少从 2 个 channel 起步；
- 以 `p2pnChannelsPerPeer` 向网络插件声明每个 peer 可使用的最大 flow 数；
- 按 net device、remote rank 和 channel 建立或复用 send/recv net comm；
- 由 proxy 线程和网络插件推进各 channel 的网络请求。

源码：

- [`paths.cc`](../../comm/nccl/src/graph/paths.cc)：跨机 P2P channel 数计算；
- [`net.cc`](../../comm/nccl/src/transport/net.cc)：`maxFlowsPerPeer` 以及按 channel
  建立/复用 net comm。

### 当前对齐状态与剩余差异

v2 现在区分三层并行度：物理/后端 GIN connection、GIN context、CUDA channel。跨机
`network_channels_per_peer` 根据实际 `ginConnectionCount`、活跃 peer 数和每个 peer 的 payload 字节数决定，
channel 到 context 仍使用 NCCL 设备端示例采用的取模方式。标准 `nccl_comm` 的网络并行度则由
topology、per-peer flow、net device 和 plugin/proxy 共同决定，仍比 v2 的公开信息更完整。

需要注意，GIN context、NCCL channel、network flow 和 RDMA QP 不是一一等价的对象。v2 默认从 active
RDMA netdev 推导 connection 数，并让 NCCL 为每个 connection 创建一个 context；NCCL 返回的实际数量仍可能因
connection 数向上取整。可通过 `AWEX_NCCL_DEVICE_V2_GIN_CONNECTIONS` 和
`AWEX_NCCL_DEVICE_V2_GIN_CONTEXTS` 分别覆盖这两个请求值，并结合以下指标做 sweep：

```text
channel_count
network_channels_per_peer
requested_network_channels_per_peer
gin_connection_count
gin_context_count
requested_gin_context_count
gin_doorbell_batch
gin_reliable_doorbell_mode
gin_type
network_step_bytes
min_work_step_bytes
payload_bytes
backend_execute_time_ms
```

建议重点观察 `gin_connection_count`、`gin_context_count` 和 `network_channels_per_peer` 是否符合
机器 NIC 配置，再比较 connection/context/channel 组合的吞吐变化。

## 结论

本轮已经对齐跨机分片公式、128 KiB 网络 step、SIMPLE 小消息调节，并把默认 GIN 并行度改为
运行时自动规划：active RDMA netdev 决定 connection，每个 connection 一个 context，channel 按 connection 与 peer
payload 负荷分配。GDAKI reliable doorbell 默认使用带普通 DBR 回退的 mode 2。connection、context、
channel、doorbell batch 以及 reliable doorbell mode 都保留独立覆盖项用于诊断。

仍未完全对齐的部分是标准 NCCL 内部基于 NIC 总带宽的 channel 增量、LL protocol 选择，以及
NET plugin/proxy 的动态 flow 调度。因此这次修改需要通过真实双机吞吐测试确认收益，不能仅根据
实现结构认定已经达到标准 NCCL 性能。
