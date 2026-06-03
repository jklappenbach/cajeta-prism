# Cajeta Prism

A new ML / scientific-compute framework written in **Cajeta**. Prism is a
*consumer* of the Cajeta foundation specs — it sits on top of **cajeta-gpu**
(value types, math, textures, memory, ray query, cooperative matrix) and
**cajeta-xpu** (the kernel/dispatch execution model), the way the
numpy/scipy/keras/torch ports do. It is **not** a foundation spec.

The roadmap lives in `cajeta/plans/cajeta-prism-plan.md`. Two threads:

- **P1 — RT-as-compute spatial primitive** (research-grounded): hardware ray
  tracing repurposed as a general-purpose spatial-index accelerator
  (kNN / radius search, clustering, Monte-Carlo transport, point-in-mesh,
  range queries) — the verified RTNN pattern, built on the cajeta-gpu ray-query
  primitive.
- **P2 — SPELA forward training** (primary focus): single-forward-pass,
  per-layer local-loss training with no global backprop (canonical impl:
  `ml/spela-training`).

## Layout

- `src/prism/spatial/SpatialIndex.cajeta` — **P1.0**, the first real Prism code.
  A `SpatialIndex` over 3-D points with a fixed-radius `countWithin` verb. The
  index is a bottom-level acceleration structure (a BVH over per-point AABBs);
  each query is a `RayQuery` walk inside a compute kernel. The "ray tracing"
  (degenerate rays, AABB-as-index-entry) is library-internal and never surfaced —
  the user-facing model is a spatial index, not rays.

## Status

Seed / early. Prism has no standalone build/test harness yet; P1.0 is
exec-verified through the cajeta compiler's JIT test harness
(`cajeta/test/xpu/PrismSpatialIndexDeviceTests.cpp`), which compiles
`SpatialIndex.cajeta` alongside a driver and runs it on a real ray-query-capable
Vulkan device. A dedicated Prism build lands once the foundation's Part C items
(ray query ✓, cooperative matrix) are real.
