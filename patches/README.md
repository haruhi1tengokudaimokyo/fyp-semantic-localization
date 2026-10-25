# Pinned upstream patch audit

Upstream: `072b7a3d6056c44ad95d08b269bdd296fc3eadab`.

The old aggregate `official_fixes.patch` remains unchanged as the v1 artifact. These three files partition its hunks for review; apply either the aggregate OR the three components, never both.

- `01_runtime_compat.patch`: Python configuration access, Linux path and GPU count, S3D processed path. Runtime/IO fixes, no network architecture change.
- `02_semantic_dispatch.patch`: checkpoint constructor argument fix and explicit `localize_type='semantic'`. The latter changes released executable behavior: it restores the intended semantic mismatch branch rather than numerical L1 over class IDs. It is a documented behavioral repair, not a claim that the unpatched code produced our numbers.
- `03_dataset_storage.patch`: pose copy, mask shape, depth float and ray API signature. Depth integer→float changes supervision values; it does NOT prove that a released checkpoint was trained with float or axial targets. Door-through ray semantics and radial/axial conventions are not silently modified.

Reproduction tests in `tests/test_audit.py` exercise the current API, semantic energy distinction, frozen checkpoint state, fast-versus-original ray casts, and exact reuse of the official candidate ray function. The official literal fine comparison remains selectable even where it mixes network depth interpretation and radial candidate distances. Consistency changes belong only to `configs/integration`.
