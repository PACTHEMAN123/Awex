# Ring routing

Rollout engine IDs can alternate across nodes. Ring order follows actual node
placement, collected once before preparing the immutable host plan. Engines on
the same node stay consecutive, ordered by engine ID; nodes are ordered by their
first engine. A TP instance must fit on one node.

For engines 0/2 on node A and 1/3 on node B, naive sends every source's stream to
engine 0 and relays through **0 → 2 → 1 → 3**. Each TP shard follows its matching
worker in those instances. The last instance terminates the chain. Naive never
rotates the entry instance by source.

Swizzle uses the same actual node groups, rotating the entry node and member by
source to distribute first-hop traffic. Off sends directly to each destination.
GIN connections/contexts, channel limits, FIFO depths, step/chunk sizes and
signal settings are unchanged. Route-dependent active peers and traffic change
with the requested ring path.

With `SHARDSTREAM_PROFILE=1`, each worker emits one `SHARDSTREAM_RING_ROUTES`
record with its actual prepared source/forward edges and ring IDs. Subsequent
cached launches must retain those descriptors. This supports checking every
ring against the real node placement, separately from weight equality and
publication timing.
