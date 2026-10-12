# Matched host RMA control

This experimental control preserves r23's original `ncclCommInitRank` call and
reports the effective `hostRmaSupport` property in the retained preparation
metrics. It uses the same observation code as the host RMA candidate.

Compile both against the pinned NCCL 2.31.2 headers with identical compiler flags.
The candidate's runtime difference is its private communicator's `numRmaCtx=0`;
the control retains default host RMA settings. Neither changes GIN geometry,
channels, FIFO, routes, or the live matrix stages.

The control must establish whether host RMA is actually enabled. If its reported
property is already zero, the proposed host RMA mechanism does not explain a
preparation speedup. Native compilation and GPU validation are pending; this
source change is not a measured optimization or a default change.
