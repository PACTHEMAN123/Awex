# Awex `weights_exchange_vllm_it.py` 原生 NCCL Device API 草案

> 目标：不接入 DeepEP，而是使用 NCCL Device API 和 Awex 自有 CUDA kernel，替换 NCCL 后端中 CPU 逐 operation 构造、提交和同步的部分。首个适配对象是单机、同一个 LSA domain 内的多 GPU、`tp_size=1` 的 `weights_exchange_vllm_it.py`。
>
> 基线：`PACTHEMAN123/Awex` 的 `main`（本次检查到 `4e19c60`）。本草案依赖的 `weights_exchange_vllm_it.py`、`transfer_plan.py` 和 `nccl_comm.py` 在 fork 中与当前上游 checkout 的结构一致。

## 1. 结论

这件事技术上可行，但“完全不经过 CPU”需要定义为：**运行时的数据搬运、operation 匹配和远端进度不经过 CPU**。NCCL Device API 仍需要 host 侧一次性创建 communicator/window/GIN 或 LSA 资源；每轮更新也仍需要把本轮 tensor 地址和任务描述符写入 device 可见内存，并启动 kernel 或更新 doorbell。

实现上已经增加独立的 opt-in backend `comm_backend=nccl_device`，默认的 `comm_backend=nccl` 不变：

```bash
AWEX_NCCL_DEVICE_TIMEOUT_MS=120000 \
python -m awex.tests.weights_exchange_vllm_it \
  --comm_backend nccl_device \
  --nccl-device-chunk-mb 4
```

`AWEX_NCCL_INCLUDE` 和 `AWEX_NCCL_LIB` 可用于指定包含 `nccl_device.h` 的 NCCL 头文件目录和 `libnccl.so` 目录。这样可在同一个 IT case 内直接 A/B legacy NCCL 与 device path。当前实现没有 queue-depth 配置；每次 update 使用单调 sequence 和 rank-indexed counters，避免复用 slot 时覆盖仍在消费的信号。

device task 在现有 `TransferPlan` 降低为 `_DeviceBatch` 时按固定字节数切分，默认 chunk 为 4 MiB。可通过 `AWEX_NCCL_DEVICE_CHUNK_BYTES` 或上述 IT 参数调整；设置为 0 会恢复一 tensor 一 task，便于 A/B。切分只改变 device task 粒度，不改变通用 `CommunicationOperation`，因此不会影响 legacy NCCL 后端。

## 2. 现有路径和替换边界

`weights_exchange_vllm_it.py` 的 reader/writer 生命周期、MetaServer、converter 和 `TransferPlan` 应保持不变。当前 NCCL 执行层大致是：

1. `TransferPlan` 按参数名、rank、shard 和 offset 生成确定顺序的 `CommunicationOperation`。
2. `nccl_comm.py` 把 operation 逐条转换为 `dist.P2POp`，调用 `batch_isend_irecv`/`batch_send_recv`。
3. 非 contiguous view 在 CPU 侧准备 staging tensor，通信后 copy back；colocate 路径还会做分阶段同步和 barrier。

新实现只替换第 2、3 步的提交和执行，不绕过第 1 步。参数匹配、shape/dtype/byte 数校验和版本错误继续由 Awex 负责。

## 3. 目标架构

```text
Awex Python: MetaServer / converter / TransferPlan
              |
              v
Awex C++/CUDA extension:
  ncclDevCommCreate + ncclCommWindowRegister
  tensor pointer table + task table + slot/control state
              |
              v
awex_transfer_kernel<<<...>>>
  read tasks -> pack/gather -> LSA or GIN -> unpack/scatter
              |
              v
destination model weights
```

这里不依赖 DeepEP。Awex 自己维护 task descriptor、slot、sequence、匹配和生命周期。

### 3.1 一次性 host 初始化

在现有 NCCL process group 建立后，每个 transfer rank 初始化一次：

1. 查询 NCCL Device API requirements 和 communicator properties。
2. 创建 `ncclDevComm`，按实际路径准备 LSA/GIN context。
3. 为通信 staging arena、control 和 slot 区域调用 `ncclCommWindowRegister`。对称 window 的注册必须在 communicator 上 collective 完成。
4. 分配 device-resident 的 tensor pointer table、`AwexTransferTask[]`、rank-indexed ready/done counters、error state 和 completion counter。
5. 把 rank、peer、window pointer 和 transport kind 等静态字段写入 `AwexDeviceContext`。

这些动作应放在 `_awex_init` 生命周期内，不是每次更新的 CPU progress。启动时检查 NCCL/CUDA/driver/GPU 是否满足 Device API 要求（GPU 代码通常需要 CUDA 12.2+）；当前显式选择 `nccl_device` 但环境不满足时会报错，使用 `comm_backend=nccl` 可明确切回 legacy NCCL。

### 3.2 每轮更新的 host 工作

每次 `writer.send()` 或 `reader.update()` 只做：

1. 使用现有 plan 的稳定顺序，写入本轮实际 tensor 地址的 `AwexTensorRef[]`。
2. 把 operation 元数据绑定为 `AwexTransferTask[]`。静态 shape/offset/dtype 可缓存，只更新 pointer table 和 sequence。
3. 用 pinned host memory + async copy，或一次小 H2D copy，更新 task count、sequence、role 和 doorbell。
4. 启动一个 `awex_transfer_kernel`，或者唤醒已有 persistent worker。

host 不再逐 operation 调 Python/C10D，不再按 peer 循环等待，也不在 task 之间插入 GPU synchronize。

## 4. Device-side 数据结构

下面是概念 ABI，正式实现时要固定大小、对齐和版本号，不能直接把 Python dataclass memcpy 到 GPU。

```cpp
enum class AwexTransportKind : uint8_t {
  kLsa = 0,       // 同机 peer，优先走 LSA/symmetric window
  kGin = 1,       // 支持 GIN 时走 ncclGin
  kStagedLsa = 2, // 不规则 view 先在 arena 中 pack
};

struct AwexTensorRef {
  uint64_t ptr;
  int64_t  storage_id; // 仅用于 debug/diagnostic
  uint32_t flags;      // contiguous、needs_copyback 等
};

struct AwexTransferTask {
  uint32_t src_tensor_id;
  uint32_t dst_tensor_id;
  uint32_t src_rank;
  uint32_t dst_rank;
  uint32_t peer;
  uint32_t slot;
  uint64_t src_byte_offset;
  uint64_t dst_byte_offset;
  uint64_t nbytes;
  int64_t  src_stride_bytes;
  int64_t  dst_stride_bytes;
  uint32_t dtype_code;
  uint32_t op_kind;    // pack/send/recv/unpack
  uint64_t sequence;
};

struct AwexDeviceContext {
  ncclDevComm dev_comm;
  ncclWindow_t comm_window;
  AwexTransportKind transport;
  AwexTensorRef* tensor_table;
  AwexTransferTask* tasks;
  uint32_t* task_count;
  uint64_t* doorbell;
  uint64_t* completed_sequence;
  uint32_t world_size;
};
```

规则 stride 可以由 kernel 处理；任意非 contiguous view 不应假设能用单个 stride 表达，必须标为 staged 或回退 legacy。

## 5. Kernel 协议

### 5.1 MVP：一次 update 一次 kernel

reader 和 writer 都启动同一个 kernel，带本地 role 和 task table。伪代码如下：

```cpp
__global__ void awex_transfer_kernel(AwexDeviceContext ctx, Role role) {
  uint64_t sequence = wait_for_new_doorbell(ctx);
  publish_local_ready(ctx, sequence);
  wait_remote_ready(ctx, sequence);

  for (uint32_t i = blockIdx.x; i < *ctx.task_count; i += gridDim.x) {
    AwexTransferTask task = ctx.tasks[i];
    if (!task_belongs_to_role(task, role)) continue;

    Slot slot = acquire_slot(ctx, task.slot, sequence);
    if (needs_pack(task)) pack_to_slot(task, ctx.tensor_table, slot);

    if (is_sender(task, role)) {
      device_put_or_lsa_copy(ctx, slot, task);
      device_signal_remote(ctx, slot, sequence);
    } else {
      device_wait_remote(ctx, slot, sequence);
      unpack_from_slot(task, ctx.tensor_table, slot);
    }
    release_slot(ctx, slot, sequence);
  }
  publish_completed(ctx, sequence);
}
```

真实实现还需要 block/warp 协作、memory fence、GIN context 轮转和错误状态传播；上面只定义 task engine 的协议。

### 5.2 reader 先启动时的死锁规则

当前 case 是 vLLM reader 先启动、Megatron writer 后发送。如果 sender kernel 在 receiver kernel 启动前等待远端 signal，可能导致 sender 占满 SM 而 receiver 无法调度。因此必须：

1. 两边先快照本地 `ready_count/done_count` 到 base 数组，再发布本轮 `epoch`；kernel 只处理已经看到对端 epoch 的 peer。
2. sender 的 copy block 写入对端 symmetric window；本地 task-complete 表由 block 0 按 peer 内 ordinal 顺序转成远端 `ready_count` 增量。
3. receiver 按来源 peer 的 `ready_base + ordinal + 1` 等待，完成 copyback 后递增 sender 的 `done_count`；sender 等待所有 peer 的 expected done count。
4. 任一侧 timeout 会设置 device error flag，让本地 kernel 退出；host 将 kernel error 转成 Python 异常。

当前实现使用一个对称 window，按来源 rank 分配数据区域；每个 task 携带 peer 和 peer 内 ordinal。sender 会在 task 完成后按 ordinal 顺序发布 ready counter，receiver 按来源 peer 的 counter 等待并回写 done counter。这样可以在同一个 LSA domain 内支持多对多，且不依赖 CPU 逐 task matching。

### 5.3 Persistent worker（第二阶段）

如果 launch/submit 仍是主要开销，再把 kernel 改成常驻 worker：host 只更新 task table 并递增 doorbell，kernel 按 sequence 处理并写 completion counter。常驻 kernel 会长期占用 SM，因此在当前 stop-the-world 权重更新窗口内，MVP 的一次 launch 更适合验证；persistent worker 不应作为首版默认值。

## 6. LSA 与 GIN

### 6.1 单机多 GPU：优先 LSA

`weights_exchange_vllm_it.py` 默认是 Megatron 和 vLLM 同机不同 GPU、`tp_size=1`。首阶段应探测 CUDA peer access 和 NCCL LSA 能力：

- contiguous tensor 走 symmetric window/LSA 的 device path；
- view 或不规则 slice 先 pack 到 registered arena，再传输连续字节；
- 不把 GIN 强行用于本机 P2P，先证明 task executor 和同步协议正确。

### 6.2 GIN

跨节点或 LSA 不可用时，替换同一个 task protocol 的 transport adapter，使用 `ncclGin` 的 device-side `put/get/signal/flush`。GIN 是 one-sided device communication，但仍需要 host 侧预先创建 `ncclDevComm`、注册 window 和建立连接；它消除的是运行时 CPU progress/matching，不是初始化。

LSA 和 GIN 共享 `AwexTransferTask`、slot、sequence、credit 和 completion，只替换：

```text
device_put_or_lsa_copy()
device_signal_remote()
device_wait_remote()
```

## 7. Awex 最小改造面

建议在 fork 中增加：

```text
awex/transfer/nccl_device.py
awex/csrc/nccl_device/
  bindings.cpp
  setup.cu
  transfer_kernel.cu
  device_protocol.cuh
```

- `setup.cu`：`ncclDevCommCreate`、window register、context 创建/销毁。
- `transfer_kernel.cu`：task executor、slot protocol、pack/unpack、LSA/GIN adapter。
- `bindings.cpp`：只暴露 `init/update_table/launch/wait/destroy`，不暴露逐 operation send/recv。
- `nccl_device.py`：把现有 `TransferPlan` 变成静态 task 模板和本轮 pointer table。

不要把 Device API 调用重新塞回 `nccl_comm.py` 的 Python 循环；legacy 路径要保留为 baseline。

建议在 `transfer_plan.py` 增加纯数据接口：

```python
class DeviceTransferPlan:
    static_tasks: list[StaticTask]
    tensor_ids: dict[str, int]
    requires_staging: bool

    def bind(self, src_tensors, dst_tensors, sequence) -> DeviceBatch:
        """只生成 pointer table、sequence 和动态字段。"""
```

`bind()` 不调用 `dist.isend/irecv`，也不触发逐 tensor synchronize。初始化时在每个 peer group 上验证两边 static task 的数量、顺序、dtype 和 byte length 完全一致。

writer/reader 只增加一个内部分支：

```python
if self.device_api_enabled:
    self._device_transport.submit(plan, tensors, sequence)
else:
    self._legacy_nccl_transport.submit(plan, tensors)
```

异常时同时检查 device error state 和 host NCCL error，避免 kernel 失败后 Python 永远等待。

## 8. 分阶段计划

### P0：baseline

- 记录 parameter 数、operation 数、总字节数、CPU submit 时间、GPU idle 时间和端到端 update latency。
- 单独统计 staging/copyback 与 NCCL transport。
- 确认 contiguous operation 占比，避免把额外 pack 误算成通信收益。

### P1：device task engine

- 完成 task、pointer table、sequence/slot、completion 和 abort protocol。
- kernel 先使用单机可见 copy path，验证 reader 先启动、writer 延迟启动、多 peer 交错以及逐字节结果。

### P2：LSA/symmetric window（当前实现）

- 加入 `ncclDevCommCreate`、collective window registration 和真实 LSA device access。
- 第一版只支持 contiguous FP16/BF16/FP32；其他 operation staged 或回退 legacy。

### P3：staged fusion（当前实现的受限版本）

- 把现有 `slice_tensor` 的 contiguous staging 变成 device-side pack/unpack task。
- 融合多个小 task，加入 slot ring 和背压，禁止 producer 覆盖未完成 slot。

### P4：GIN/persistent queue（可选）

- 有跨节点需求或 LSA 不可用时再实现 GIN。
- 先保持一次 update 一次 kernel，证明 GIN 协议正确后再评估常驻 worker。

## 9. 收益预估

这是工程估计，不是承诺：

| 场景 | 预期变化 | 主要原因 |
| --- | ---: | --- |
| 少量大块、几乎 contiguous | `0.8x–1.2x` | 链路/内存带宽是主瓶颈，device path 只能减少少量 CPU overhead |
| 数十到数百个小块 | `1.2x–1.8x` | 一次 task table + 一次 kernel，减少 Python/C10D 构造和同步 |
| 大量碎片 view | `1.5x–2.5x` | fused device pack/unpack 减少 staging launch 和 copyback |
| baseline 主要被 CPU progress/同步阻塞 | 局部 `2x–4x` | 只有链路没有成为主瓶颈时才可能达到 |

对当前 Qwen3-0.6B、单机多 GPU、`tp_size=1` case，第一目标应是：结果完全一致、CPU submit/sync 时间降低 70% 以上、端到端 update latency 不回退；operation 足够多或 view 足够碎时，再争取 `1.2x–2x`。如果大部分参数已经是少量大 contiguous tensor，收益接近零是合理结果。

## 10. 验收和风险

每轮记录 `plan_build_us`、`pointer_bind_us`、`task_upload_us`、kernel first/last completion、pack/transport/unpack、SM occupancy、P2P/NIC throughput、CPU 阻塞时间和 Python/C10D 调用次数，并对每个 parameter 做 checksum/shape/dtype/version 校验。

至少覆盖：冷启动和连续 100 次更新；reader 先启动、writer 延迟启动；contiguous、规则 stride 和 staged view；task count 为 0/1/很多；2/3/多 peer；一侧提前退出；Device API 不可用时显式切回 legacy backend。

主要风险是 Device API 版本/硬件前置条件、window 和 tensor storage 生命周期、非 contiguous view 的错误寻址、远端 wait 死锁，以及 kernel 与模型计算的 SM 争用。常驻 kernel 只应在独立实验中评估。

## 11. 推荐首个边界

- single node、同一 LSA domain 内的多个 transfer ranks、多 peer；
- `tp_size=1`、`pp=1`、`dp=1`、`ep=1`；
- contiguous FP16/BF16/FP32；
- 一次 update 一次 kernel，不做 persistent worker；
- LSA/symmetric window 优先，GIN 使用同一 task protocol 作为第二 transport；
- 非 contiguous 明确 staged 或回退 legacy；
- legacy `nccl_comm.py` 保留，device path 必须可关闭。

当前 fork 已实现 P1、P2 的最小版本和 P3 的 Python staging 回退：同一 LSA domain 内多 rank、多 peer、一次 update 一次 kernel；不规则 view 会先转成 contiguous staging buffer。跨节点 GIN、常驻 device queue 和 device-side pack/unpack 仍未实现。这个边界可以先回答最关键的问题：去掉 Awex 的 CPU 逐 operation NCCL 调度是否真的降低 update latency。

## 12. 官方接口

- NCCL Device API：<https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html>
- Device setup / `ncclDevComm`：<https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html>
- GIN device API：<https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html>
- Communicator window registration：<https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html>
