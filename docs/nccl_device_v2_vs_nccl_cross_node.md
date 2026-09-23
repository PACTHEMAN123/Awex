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
| 5. 网络执行上下文 | 默认 4 个 GIN context；跨机 channel 根据实际 GIN connection 数确定 | 常规 NET 按 NIC、带宽、channel 和 flow 调度 | context 保留 NCCL Device API 默认值；channel 已接入实际 connection 数 |

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
`ginConnectionCount`，以 `max(2, 2 * ginConnectionCount)` 作为跨机需求并向上取 2 次幂，最后受总
channel 上限约束。实测每个 connection 使用两个 channel 可以更充分地驱动 200 Gb/s 端口；由于公开 Device API
没有直接暴露标准 NCCL 内部的 NIC 总带宽，`ceil(netBw / 14 GB/s)` 这一项无法自动复刻。
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

创建 device communicator 时，v2 请求：

```text
ginContextCount = min(4, total_channels)
```

执行时每个 channel 选择：

```text
context = channel % ginContextCount
```

默认同时请求 4 个 GIN connection，因此每个 connection 分配一个 context，两个 channel 共享它。NCCL 可能根据
connection 数向上取整实际 context 数，kernel 始终使用返回的 `dev_comm.ginContextCount` 做
channel 取模。两个请求值都可由环境变量覆盖。

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
`network_channels_per_peer` 根据实际 `ginConnectionCount` 的两倍决定，而 channel 到 context 仍使用 NCCL
设备端示例采用的取模方式。标准 `nccl_comm` 的网络并行度则由 topology、per-peer flow、net device
和 plugin/proxy 共同决定，仍比 v2 的公开信息更完整。

需要注意，GIN context、NCCL channel、network flow 和 RDMA QP 不是一一等价的对象。v2 请求最多
4 个 GIN connection 和 4 个 GIN context，使默认的 8 个网络 channel 每两个共享一个 context；NCCL 返回的
实际数量仍可能因 connection 数向上取整。可通过 `AWEX_NCCL_DEVICE_V2_GIN_CONNECTIONS` 和
`AWEX_NCCL_DEVICE_V2_GIN_CONTEXTS` 分别覆盖这两个请求值，并结合以下指标做 sweep：

```text
channel_count
network_channels_per_peer
requested_network_channels_per_peer
gin_connection_count
gin_context_count
requested_gin_context_count
gin_doorbell_batch
gin_skip_credit_check
gin_type
network_step_bytes
min_work_step_bytes
payload_bytes
backend_execute_time_ms
```

建议重点观察 `gin_connection_count`、`gin_context_count` 和 `network_channels_per_peer` 是否符合
机器 NIC 配置，再比较 connection/context/channel 组合的吞吐变化。

## 结论

本轮已经对齐跨机分片公式、128 KiB 网络 step、SIMPLE 小消息调节，并将默认 GIN 并行度提高到
4 个 connection、4 个 context 和每个 remote peer 8 个 channel。connection、context、channel 以及
doorbell batch 和 QP credit check 都保留独立覆盖项，便于针对实际 NIC 数量和消息规模复测。

仍未完全对齐的部分是标准 NCCL 内部基于 NIC 总带宽的 channel 增量、LL protocol 选择，以及
NET plugin/proxy 的动态 flow 调度。因此这次修改需要通过真实双机吞吐测试确认收益，不能仅根据
实现结构认定已经达到标准 NCCL 性能。
