# Measured execution plan; science unchanged — 5 November

The first GPU preflight measured .9376s/update (conservative p90), projecting
8626s training plus 2400s terminal reserve. With 50% margin this needs 16560s,
exceeding the assistant's initial four-hour resource ceiling. It correctly
stopped before any real training. Preserve `timing/report.json` and its guard.

The user authorized setting the budget from measured throughput, not an
unmeasured one-hour assumption. This addendum changes execution planning only:
the same 9200 updates and ALL scientific settings/gates remain fixed. A separate
wrapper tests GPU-resident input caching, as used in the earlier tiny run,
instead of copying many small tensors to GPU every update. It requires EXACT
16-step streamed loss/gradient prefix equality and unchanged batch contents.
No timing trajectory is adopted. A fresh real run checks its first16 updates
again and verifies immutable resident batches after all training.

The complete 460-batch cache is measured for VRAM and throughput. Set the final
wall budget to 1.5*(p90_update_time*9200+2400s), rounded up to a minute. If the
resident loader restores throughput, this may fit the original four hours.
If not, the measured budget may exceed that initial self-imposed ceiling,
bounded by SIX hours overall. This is disclosed resource planning under the
user's explicit broad-run authorization, not a new accuracy experiment, an
extension of trained updates, or a retry after a failed training verdict.
Report the actual plan before launching; stop if it exceeds this upper bound.

Keep the original config, sources, labels and failed preflight immutable.
The new execution config binds them; reuse the exact preparation manifest.
The wrapper has no sampling, label, model, optimizer, loss, selection, graph,
comparison or gate change. Terminal CPU scoring uses the original broad
evaluation function and records the execution binding in guard/checkpoints.
