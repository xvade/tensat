# TENSAT fork — local modifications

Fork of [xvade/tensat](https://github.com/xvade/tensat) (itself a fork of
`uwplse/tensat`) with the verifiability-project changes. This is the **index of
the delta**; per-function specs live as `///` doc-comments in the source
(optimize.rs and rewrites.rs are heavily doc-commented already), the deep
rationale is in the top-level `../BUGS.md` / `../PROGRESS.md`, and untestable
pieces are flagged in `../PROBLEMATIC.md`.

Upstream tensat's own overview is `README.md` and `../TENSAT_SUMMARY.md`.

## New CLI modes (`-m <mode>`, dispatched in `src/main.rs`)

| Mode | Function | What it does |
|---|---|---|
| `verify` | `prove_taso_rules` (main.rs:1348) | GPU-free axiom saturation: proves each candidate rewrite from the `rules()` axiom set (conv/matmul axioms + activation unfolding + the added min/max lattice axioms). The soundness oracle for rule generation. Complementary to Z3 (`../NNs/z3_verify_egg.py`). |
| `redundancy` | `prune_redundant` (main.rs:1431) | Drops rules re-derivable from the others via bounded saturation. Tunable budget: `--redundancy_iters`, `--n_nodes`, `--n_sec`. Grounds elementwise/PWL rules; non-groundable rules are kept as-is. |
| `parse_check` | `parse_check` (main.rs:1317) | Authoritative oracle for whether a rule string parses under the *current* `Mdl` arities. The definitive drift check (the `Mdl` comments and `converted_full*.txt` are stale). |

`optimize` (default), `test`, `convert`, `ilp`, `greedy` are upstream.

## New extraction behavior (`src/optimize.rs`)

| Feature | Flag | Notes |
|---|---|---|
| VerifCost — verifiability-aware extraction | `--verif_cost` | Extracts to minimize an IBP interval-gap-area cost instead of runtime. Steers maxout to cert_ub 9.65 vs 12.03 at the input form. **The right extraction for a verifiability win** — deterministic, immune to the `--n_diverse` collapse (see `../PROBLEMATIC.md`). |
| Optional CROWN sensitivity weighting | `--sensitivity_file` | Weights the gap cost by backward-CROWN sensitivities (`../NNs/gen_sensitivity.py`). |
| ArchDiverseCost — architecture-diverse extraction | (extraction mode) | Uses rewrite provenance to diversify *structure*. |
| `favor_fusion` continuous strength + axis-0 Concat/Split penalty | `--favor_fusion` | Actively penalizes axis-0 Concat/Split (BUGS #11: auto_LiRPA refuses to bound axis-0 Concat). |
| `--query_chain` diagnostic | `--query_chain` | Probe whether a target association is materialized in the saturated e-graph. |
| `--n_random` sampling | `--n_random` | Random-sample extraction forms. |

## Weight provenance (`src/model.rs`, `src/rewrites.rs`, `src/main.rs`)

`ValTnsr.weight_names` (a `BTreeSet<String>`, model.rs:106) tags every
weight-derived e-class with its originating ONNX weight name(s). Set in
`rewrites.rs::apply_match_pair`, propagated through `TensorAnalysis::make()` and
`merge()` (unioned across every op — model.rs:185-405), and emitted by
`save_model_with_provenance` (main.rs:1293) as a `<model_file>.weight_names.json`
sidecar. This is what lets `../NNs/reconstruct_generic.py` resolve real weights
for **any** extraction without a hand-traced GUID dict.

## Language / parser (`src/model.rs`, `src/parse.rs`, `src/rewrites.rs`)

Added `ewsub`/`ewmax`/`ewmin` to the egg language, the cost function, the graph
builders, `parse_model` ingestion, and `CheckApply`; plus parser guards for
empty-param and newline edge cases. This is the end-to-end min/max support the
maxout/lattice experiments depend on.

## Build (`build.rs`, `Cargo.toml`, `wrapper.h`)

TASO/protobuf link paths are now env-overridable (`TASO_LIB_DIR`,
`TASO_INCLUDE_DIR`, `PROTOBUF_LIB_DIR`; default to a sibling `../taso/build`
checkout) instead of hardcoded to one Docker layout. `--no_runtime_report`
added; bindgen bumped.

## Tests

- **CLI integration (runnable, green):** `../NNs/tests/run_tests.sh` exercises
  `parse_check` (test 2), `verify` soundness + liveness against negative
  canaries and the min/max axioms (test 4), and `redundancy` grounding + pruning
  (test 8). These run against the **prebuilt** `target/debug/tensat` binary and
  need no rebuild.
- **Rust unit tests:** `tests/parse.rs` (`parse_model`). ⚠ Building these
  requires network access to resolve the crates.io registry, which the offline
  research sandbox lacks — see `../PROBLEMATIC.md`. They are expected to build
  and pass in the original (networked) container build environment.
