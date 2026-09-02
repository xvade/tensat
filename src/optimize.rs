use crate::{model::*, rewrites::*};
use egg::*;
use root::taso::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::HashSet;
use std::convert::TryInto;
use std::time::{Duration, Instant};

/// Wrapper class for egg's cost function
pub struct TensorCost<'a> {
    pub egraph: &'a EGraph<Mdl, TensorAnalysis>,
    pub cost_model: &'a CostModel,
}

impl CostFunction<Mdl> for TensorCost<'_> {
    type Cost = f32;
    /// Getting total cost for the subtree rooted at enode. See egg::CostFunction
    /// trait for more information on interface.
    fn cost<C: FnMut(Id) -> Self::Cost>(&mut self, enode: &Mdl, mut costs: C) -> Self::Cost {
        let self_cost = self.cost_model.get_self_cost(self.egraph, enode);
        enode.fold(self_cost, |sum, id| sum + costs(id))
    }
}

/// Cost function that samples a different-but-plausible extraction by
/// jittering the *real* cost model's per-node cost by a random multiplier,
/// then summing over children same as TensorCost -- rather than assigning
/// pure random costs.
///
/// Two prior versions were tried and both failed in practice:
/// - Pure random cost, summed over children: strongly biases selection
///   towards trees with fewer nodes (a multi-node equivalent subtree, e.g.
///   a relu replaced by concat+relu+split, accumulates several random
///   draws summed together, so it almost never beats a single-node
///   alternative on expectation alone). Confirmed on resnet2b: 10 samples
///   all landed on the exact same structure despite thousands of valid
///   multi-pattern-rule-derived alternatives existing in the egraph.
/// - Pure random cost, NOT summed over children (each class's
///   representative chosen independently of its children): removes that
///   bias entirely, but also removes the real cost model's preference for
///   *small* structures, so extraction can pick unboundedly larger
///   equivalent forms compounding across many e-classes at once --
///   observed as an actual OOM kill on resnet2b's ~20k-node/12k-class
///   saturated egraph when combined with building+evaluating 10 samples.
///
/// Jittering the real cost keeps its bounded-size behavior (still
/// generally favors cheap/small structures, so it won't blow up the way
/// pure independent-per-node randomness did) while letting close
/// alternatives occasionally win because of the noise.
pub struct RandomCost<'a> {
    pub egraph: &'a EGraph<Mdl, TensorAnalysis>,
    pub cost_model: &'a CostModel,
    rng: rand::rngs::StdRng,
    memo: HashMap<Mdl, f32>,
    /// Multiplier range applied to each node's real self-cost.
    jitter: (f32, f32),
}

impl<'a> RandomCost<'a> {
    pub fn new(
        egraph: &'a EGraph<Mdl, TensorAnalysis>,
        cost_model: &'a CostModel,
        seed: u64,
    ) -> Self {
        use rand::SeedableRng;
        RandomCost {
            egraph,
            cost_model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            memo: HashMap::new(),
            jitter: (0.1, 3.0),
        }
    }
}

impl CostFunction<Mdl> for RandomCost<'_> {
    type Cost = f32;
    fn cost<C: FnMut(Id) -> Self::Cost>(&mut self, enode: &Mdl, mut costs: C) -> Self::Cost {
        use rand::Rng;
        let real_self_cost = self.cost_model.get_self_cost(self.egraph, enode);
        let (lo, hi) = self.jitter;
        let rng = &mut self.rng;
        let self_cost = *self
            .memo
            .entry(enode.clone())
            .or_insert_with(|| real_self_cost * rng.gen_range(lo, hi));
        enode.fold(self_cost, |sum, id| sum + costs(id))
    }
}

/// A structure-agnostic random baseline, for comparison against
/// `RandomCost`'s jittered-real-cost sampling (which is still fundamentally
/// shaped by the real cost model, just noisy). Each enode gets an i.i.d.
/// draw independent of what op it actually is or what it costs to run --
/// still summed over children (not per-class-independent) to stay bounded,
/// since the un-summed version is exactly the "OOM on resnet2b" failure
/// mode documented on `RandomCost` above. This keeps the same *known,
/// reported* small-tree bias as the first (summed) attempt described
/// there -- a multi-node equivalent subtree accumulates more random draws
/// than a single-node alternative almost regardless of the draws -- which
/// is why this exists as an explicit, separate mode rather than replacing
/// `RandomCost`'s default: it's a deliberately weaker, unbiased-by-op-type
/// baseline for the structural-diversity-vs-verifiability campaign, not a
/// claim that it samples the equivalence class uniformly.
pub struct UniformRandomCost<'a> {
    pub egraph: &'a EGraph<Mdl, TensorAnalysis>,
    rng: rand::rngs::StdRng,
    memo: HashMap<Mdl, f32>,
    /// Range each enode's i.i.d. self-cost is drawn from.
    range: (f32, f32),
}

impl<'a> UniformRandomCost<'a> {
    pub fn new(egraph: &'a EGraph<Mdl, TensorAnalysis>, seed: u64) -> Self {
        use rand::SeedableRng;
        UniformRandomCost {
            egraph,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            memo: HashMap::new(),
            range: (1.0, 10.0),
        }
    }
}

impl CostFunction<Mdl> for UniformRandomCost<'_> {
    type Cost = f32;
    fn cost<C: FnMut(Id) -> Self::Cost>(&mut self, enode: &Mdl, mut costs: C) -> Self::Cost {
        use rand::Rng;
        let (lo, hi) = self.range;
        let rng = &mut self.rng;
        let self_cost = *self
            .memo
            .entry(enode.clone())
            .or_insert_with(|| rng.gen_range(lo, hi));
        enode.fold(self_cost, |sum, id| sum + costs(id))
    }
}

/// Cost function for the diversity-seeking sampler (`--n_diverse`): drawing
/// several samples in sequence, each one penalized against re-selecting any
/// enode a *previous* sample already used, so successive samples are
/// pushed toward covering structurally distinct regions of the egraph
/// rather than just noisily perturbing the same near-optimal tree
/// (`RandomCost`'s failure mode when alternatives are far from cost-
/// competitive). Cost = jittered real cost (reusing `RandomCost`'s own
/// jitter range, so once the single top choice at some e-class is
/// exhausted, extraction doesn't just deterministically repick the same
/// "obvious" runner-up every subsequent round) plus a large additive
/// penalty if the enode is in `used`. Additive, not multiplicative, so a
/// zero-or-near-zero-real-cost node (e.g. a `Weight`/`Input` leaf) is still
/// meaningfully discouraged from reuse instead of the penalty vanishing
/// along with the base cost.
pub struct DiverseCost<'a> {
    pub egraph: &'a EGraph<Mdl, TensorAnalysis>,
    pub cost_model: &'a CostModel,
    pub used: &'a HashSet<Mdl>,
    rng: rand::rngs::StdRng,
    memo: HashMap<Mdl, f32>,
    jitter: (f32, f32),
    penalty: f32,
}

impl<'a> DiverseCost<'a> {
    pub fn new(
        egraph: &'a EGraph<Mdl, TensorAnalysis>,
        cost_model: &'a CostModel,
        used: &'a HashSet<Mdl>,
        seed: u64,
    ) -> Self {
        use rand::SeedableRng;
        DiverseCost {
            egraph,
            cost_model,
            used,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            memo: HashMap::new(),
            jitter: (0.1, 3.0),
            penalty: 1e6,
        }
    }
}

impl CostFunction<Mdl> for DiverseCost<'_> {
    type Cost = f32;
    fn cost<C: FnMut(Id) -> Self::Cost>(&mut self, enode: &Mdl, mut costs: C) -> Self::Cost {
        use rand::Rng;
        let real_self_cost = self.cost_model.get_self_cost(self.egraph, enode);
        let (lo, hi) = self.jitter;
        let rng = &mut self.rng;
        let jittered = *self
            .memo
            .entry(enode.clone())
            .or_insert_with(|| real_self_cost * rng.gen_range(lo, hi));
        let penalty = if self.used.contains(enode) { self.penalty } else { 0.0 };
        enode.fold(jittered + penalty, |sum, id| sum + costs(id))
    }
}

/// Cost function for ARCHITECTURE-diverse extraction. DiverseCost only ever
/// *penalizes* reused enodes, so a fused subtree's higher intrinsic cost always
/// wins out and the extractor never crosses the cost cliff onto it -- which is
/// why it only ever produced the (unfused) baseline structure. ArchDiverseCost
/// instead *rewards* selecting the rewrite-witness enodes of a chosen
/// `target_rule` (making that architecture family cheaper so it actually wins
/// its e-class), and *penalizes* witnesses of already-`covered_rules` families.
/// Rotating `target_rule` across a sequence of samples yields (at most) one
/// extraction per reachable rewrite family -- diverse by construction, O(#rules),
/// never enumerating the extraction space. Reads the per-enode provenance
/// recorded during saturation (`TensorAnalysis::rewrite_witness`, model.rs).
pub struct ArchDiverseCost<'a> {
    pub egraph: &'a EGraph<Mdl, TensorAnalysis>,
    pub cost_model: &'a CostModel,
    /// Rewrite family (index into MultiPatterns.rules) to reward this sample.
    pub target_rule: Option<usize>,
    /// Families prior samples already covered -- penalize their witnesses.
    pub covered_rules: &'a HashSet<usize>,
    rng: rand::rngs::StdRng,
    memo: HashMap<Mdl, f32>,
    jitter: (f32, f32),
    reward: f32,
    penalty: f32,
}

impl<'a> ArchDiverseCost<'a> {
    pub fn new(
        egraph: &'a EGraph<Mdl, TensorAnalysis>,
        cost_model: &'a CostModel,
        target_rule: Option<usize>,
        covered_rules: &'a HashSet<usize>,
        seed: u64,
        reward: f32,
        penalty: f32,
    ) -> Self {
        use rand::SeedableRng;
        ArchDiverseCost {
            egraph,
            cost_model,
            target_rule,
            covered_rules,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            memo: HashMap::new(),
            jitter: (0.1, 3.0),
            reward,
            penalty,
        }
    }
}

impl CostFunction<Mdl> for ArchDiverseCost<'_> {
    type Cost = f32;
    fn cost<C: FnMut(Id) -> Self::Cost>(&mut self, enode: &Mdl, mut costs: C) -> Self::Cost {
        use rand::Rng;
        let real_self_cost = self.cost_model.get_self_cost(self.egraph, enode);
        let (lo, hi) = self.jitter;
        let rng = &mut self.rng;
        let jittered = *self
            .memo
            .entry(enode.clone())
            .or_insert_with(|| real_self_cost * rng.gen_range(lo, hi));
        // Provenance-driven bias: reward the target family's witnesses (drive
        // their self-cost to 0 so the fused representative wins its e-class),
        // penalize already-covered families. Clamp at 0 -- never negative, so
        // extraction can't be driven to prefer unboundedly larger trees.
        let bias = match self.egraph.analysis.rewrite_witness.borrow().get(enode) {
            Some(w) if Some(w.rule_index) == self.target_rule => -self.reward,
            Some(w) if self.covered_rules.contains(&w.rule_index) => self.penalty,
            _ => 0.0,
        };
        enode.fold((jittered + bias).max(0.0), |sum, id| sum + costs(id))
    }
}

/// Verifiability-aware extraction cost. Steers extraction toward the ReLU
/// topology with the least total relaxation slack. Each `ewmax`/`ewmin` enode
/// lowers (at reconstruction) to one ReLU -- `max(u,v)=u+relu(v-u)`,
/// `min(u,v)=u-relu(u-v)` -- and we score it by the TRIANGLE RELAXATION AREA of
/// that ReLU's pre-activation interval over the input box: a stable ReLU
/// (pre-activation doesn't straddle 0) costs 0; an unstable one costs
/// `-l*u/(u-l)` per element. Minimizing the summed gap-area => fewer/smaller
/// unstable ReLUs => tighter certified bound (the mechanism measured on the
/// maxout PoC: chain tighter <=> fewer unstable ReLUs).
///
/// DESIGN CHOICES (the "how"):
/// - Intervals are computed LAZILY here at extraction time via a memoized
///   recursive walk over the e-graph, NOT as an egg Analysis. Extraction runs
///   after saturation so every enode exists; this sidesteps make/merge
///   re-propagation entirely and lets us choose the per-class interval
///   deliberately.
/// - A class's interval is the INTERSECTION (tightest) over its enodes'
///   computed intervals -- an *optimistic* score (the tightest any equivalent
///   form achieves). For a steering heuristic this can only mis-rank, never
///   break soundness (the numeric gate is independent).
/// - Affine LEAF intervals are INJECTED from a sidecar keyed by the leaf's
///   weight-name set: each affine map `W_k x + b_k` has a unique {weight,bias}
///   name pair, and `ValTnsr.weight_names` already carries that provenance.
///   Interior nodes get interval arithmetic from their children.
/// - A None (unscoreable) child reads as WORST (large penalty), never free, so
///   the extractor is not *attracted* to nodes it cannot score.
/// - A tiny per-op epsilon tie-breaks so op-bloating bridge forms
///   (`max=(a+b)-min` adds ewadd/ewsub) don't come out cost-free.
pub type Interval = (Vec<f32>, Vec<f32>);

pub struct VerifCost<'a> {
    pub egraph: &'a EGraph<Mdl, TensorAnalysis>,
    /// sorted leaf weight-name set -> element-wise (lo, hi) over the input box
    pub leaf_intervals: HashMap<Vec<String>, Interval>,
    /// Optional per-node backward-CROWN sensitivity |lambda|, keyed by the node's
    /// OUTPUT weight-name set (union of its operands' names -- identifies which
    /// sub-function's ReLU this is). Weights the gap so the cost targets ReLUs whose
    /// relaxation slack actually reaches the output (critical-path), not all equally.
    /// Empty => all 1.0 (the unweighted VerifCost).
    pub sensitivities: HashMap<Vec<String>, f32>,
    pub scale: f32,
    memo: std::cell::RefCell<HashMap<Id, Option<Interval>>>,
    visiting: std::cell::RefCell<HashSet<Id>>,
}

fn iv_binop(a: &Interval, b: &Interval, f: impl Fn(f32, f32, f32, f32) -> (f32, f32)) -> Interval {
    let n = a.0.len().min(b.0.len());
    let (mut lo, mut hi) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let (l, u) = f(a.0[i], a.1[i], b.0[i], b.1[i]);
        lo.push(l);
        hi.push(u);
    }
    (lo, hi)
}
fn iv_intersect(a: &Interval, b: &Interval) -> Interval {
    let n = a.0.len().min(b.0.len());
    let (mut lo, mut hi) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        lo.push(a.0[i].max(b.0[i]));
        hi.push(a.1[i].min(b.1[i]));
    }
    (lo, hi)
}

impl<'a> VerifCost<'a> {
    pub fn new(
        egraph: &'a EGraph<Mdl, TensorAnalysis>,
        leaf_intervals: HashMap<Vec<String>, Interval>,
        sensitivities: HashMap<Vec<String>, f32>,
        scale: f32,
    ) -> Self {
        VerifCost {
            egraph,
            leaf_intervals,
            sensitivities,
            scale,
            memo: std::cell::RefCell::new(HashMap::new()),
            visiting: std::cell::RefCell::new(HashSet::new()),
        }
    }

    /// Backward-CROWN sensitivity for the ReLU inside `ewmax/ewmin(u,v)`, keyed by
    /// the node's OUTPUT weight-name set (= union of operands' names). 1.0 if no
    /// sensitivity map is loaded or this node is unmeasured (falls back to unweighted).
    fn node_sensitivity(&self, u: Id, v: Id) -> f32 {
        if self.sensitivities.is_empty() {
            return 1.0;
        }
        let mut wn: Vec<String> = self.egraph[self.egraph.find(u)]
            .data
            .weight_names
            .union(&self.egraph[self.egraph.find(v)].data.weight_names)
            .cloned()
            .collect();
        wn.sort();
        self.sensitivities.get(&wn).copied().unwrap_or(1.0)
    }

    /// Tightest-over-forms interval of the function an e-class computes.
    fn class_interval(&self, id: Id) -> Option<Interval> {
        let cid = self.egraph.find(id);
        if let Some(v) = self.memo.borrow().get(&cid) {
            return v.clone();
        }
        if self.visiting.borrow().contains(&cid) {
            return None; // cycle guard
        }
        // Affine leaf? key by the class's weight-name set.
        let mut wn: Vec<String> = self.egraph[cid].data.weight_names.iter().cloned().collect();
        wn.sort();
        if let Some(iv) = self.leaf_intervals.get(&wn) {
            self.memo.borrow_mut().insert(cid, Some(iv.clone()));
            return Some(iv.clone());
        }
        self.visiting.borrow_mut().insert(cid);
        let nodes: Vec<Mdl> = self.egraph[cid].nodes.clone();
        let mut best: Option<Interval> = None;
        for enode in nodes.iter() {
            if let Some(iv) = self.enode_interval(enode) {
                best = Some(match best {
                    None => iv,
                    Some(b) => iv_intersect(&b, &iv),
                });
            }
        }
        self.visiting.borrow_mut().remove(&cid);
        self.memo.borrow_mut().insert(cid, best.clone());
        best
    }

    fn enode_interval(&self, enode: &Mdl) -> Option<Interval> {
        match enode {
            Mdl::Ewadd([a, b]) => Some(iv_binop(&self.class_interval(*a)?, &self.class_interval(*b)?, |la, ua, lb, ub| (la + lb, ua + ub))),
            Mdl::Ewsub([a, b]) => Some(iv_binop(&self.class_interval(*a)?, &self.class_interval(*b)?, |la, ua, lb, ub| (la - ub, ua - lb))),
            Mdl::Ewmax([a, b]) => Some(iv_binop(&self.class_interval(*a)?, &self.class_interval(*b)?, |la, ua, lb, ub| (la.max(lb), ua.max(ub)))),
            Mdl::Ewmin([a, b]) => Some(iv_binop(&self.class_interval(*a)?, &self.class_interval(*b)?, |la, ua, lb, ub| (la.min(lb), ua.min(ub)))),
            Mdl::Relu(a) => {
                let (l, u) = self.class_interval(*a)?;
                Some((l.iter().map(|x| x.max(0.0)).collect(), u.iter().map(|x| x.max(0.0)).collect()))
            }
            _ => None, // matmul/smul/input/etc. without a leaf key -> unknown
        }
    }

    /// Summed triangle relaxation area of `relu(hi_operand - lo_operand)`.
    /// Unscoreable (None child) reads as WORST so it can't attract the extractor.
    fn relu_gap(&self, hi_operand: Id, lo_operand: Id) -> f32 {
        match (self.class_interval(hi_operand), self.class_interval(lo_operand)) {
            (Some((lh, uh)), Some((ll, ul))) => {
                let mut g = 0.0f64;
                for i in 0..lh.len().min(ll.len()) {
                    let l = (lh[i] - ul[i]) as f64; // pre-activation lower
                    let u = (uh[i] - ll[i]) as f64; // pre-activation upper
                    if l < 0.0 && u > 0.0 {
                        g += (-l * u) / (u - l);
                    }
                }
                g as f32
            }
            _ => 1.0e6,
        }
    }

    /// How many injected leaf keys actually match an e-class (the 16/16 check).
    pub fn leaves_bound(&self) -> (usize, usize) {
        let present: HashSet<Vec<String>> = self
            .egraph
            .classes()
            .map(|c| {
                let mut w: Vec<String> = c.data.weight_names.iter().cloned().collect();
                w.sort();
                w
            })
            .collect();
        let hit = self.leaf_intervals.keys().filter(|k| present.contains(*k)).count();
        (hit, self.leaf_intervals.len())
    }
}

impl CostFunction<Mdl> for VerifCost<'_> {
    type Cost = f32;
    fn cost<C: FnMut(Id) -> Self::Cost>(&mut self, enode: &Mdl, mut costs: C) -> Self::Cost {
        const EPS: f32 = 1.0e-3; // op-count tie-breaker floor
        let self_cost = match enode {
            // gap-area weighted by the ReLU's backward-CROWN sensitivity (1.0 if
            // no sensitivity map loaded => the plain unweighted VerifCost).
            Mdl::Ewmax([u, v]) => self.node_sensitivity(*u, *v) * self.relu_gap(*v, *u) * self.scale + EPS,
            Mdl::Ewmin([u, v]) => self.node_sensitivity(*u, *v) * self.relu_gap(*u, *v) * self.scale + EPS,
            _ => EPS,
        };
        enode.fold(self_cost, |sum, id| sum + costs(id))
    }
}

/// Class for our cost model
pub struct CostModel {
    /// To have zero cost for all weight op only
    ignore_all_weight_only: bool,
    /// Discount factor for all weight ops
    all_weight_discount: f32,
    /// Continuous discount applied to axis!=0 Concat/Split/Enlarge's real
    /// measured cost: 1.0 is neutral (no bias at all -- byte-identical to
    /// the old boolean favor_fusion=false path), and progressively smaller
    /// values make a legally-available multi-pattern-rule fusion (e.g. the
    /// relu-merge rule: two relus -> concat+relu+split) win more and more
    /// readily under deterministic extraction, even though it's more real
    /// ops than what it replaces and so never wins under the unmodified
    /// cost model on its own. Not a claim that fusion is actually free --
    /// a knob to make an already-proven-valid equivalence observable for
    /// comparison.
    ///
    /// Originally a plain bool (on = discount by a fixed 0.05x). Made
    /// continuous because the fixed discount's cost advantage turned out
    /// to be marginal rather than decisive -- deterministic extraction
    /// with it enabled sometimes still didn't find the known InceptionMNIST
    /// fusion (run-to-run noise in TASO's own real-cost measurement, from
    /// the random dummy weight *values* `Mdl::Weight`'s `make()` fills in,
    /// was enough to flip the winner). That marginality is exactly why
    /// `--n_random`/`--n_diverse` sampling (which layers jitter or a
    /// diversity penalty on TOP of this already-close gap) essentially
    /// never stumbled into the fused structure on its own -- see
    /// `PROGRESS.md`'s structural-diversity-vs-verifiability campaign
    /// entry. Sampling across a RANGE of strengths (e.g. 1.0, 0.5, 0.2,
    /// 0.05, 0.01) is how that campaign gets samples that reliably span
    /// clearly-unfused through clearly-fused, instead of relying on noise
    /// to land in the interesting region.
    favor_fusion_strength: f32,
}

impl CostModel {
    pub fn with_setting(ignore_all_weight_only: bool) -> Self {
        CostModel {
            ignore_all_weight_only: ignore_all_weight_only,
            all_weight_discount: 1.0,
            favor_fusion_strength: 1.0,
        }
    }

    pub fn with_favor_fusion_strength(ignore_all_weight_only: bool, favor_fusion_strength: f32) -> Self {
        CostModel {
            ignore_all_weight_only: ignore_all_weight_only,
            all_weight_discount: 1.0,
            favor_fusion_strength: favor_fusion_strength,
        }
    }

    /// Gets cost for the enode itself.
    ///
    /// This function gets the cost by calling TASO's get_or_create_{some_op}()
    /// functions with the tensor information stored in metadata. TASO side stores
    /// hashmaps for OpBase objects. So here TASO side will simply lookup previously
    /// created ops (with previously measured runtime).
    ///
    /// # Parameters
    ///
    /// - `egraph`: E-graph of interest
    /// - `enode`: enode to get cost for
    ///
    /// # Returns
    ///
    /// Cost for this enode.
    pub fn get_self_cost(&self, egraph: &EGraph<Mdl, TensorAnalysis>, enode: &Mdl) -> f32 {
        let x = |i: &Id| &egraph[*i].data;
        let mut g = egraph.analysis.graph.borrow_mut();
        let cost = match enode {
            Mdl::Num(_)
            | Mdl::Var(_)
            | Mdl::Input(_)
            | Mdl::Weight(_)
            | Mdl::Merge(_)
            | Mdl::Split0(_)
            | Mdl::Split1(_)
            | Mdl::Reshape(_)
            | Mdl::Transpose(_)
            | Mdl::Dropout(_)
            | Mdl::Iewmul            // const-tensor markers: compile-time, zero cost
            | Mdl::Imatmul
            | Mdl::Iconv(_)
            | Mdl::Noop(_) => 0.0,

            Mdl::Relu(_a) => {
                // Check types
                let a_t_data = x(_a);
                assert!(a_t_data.dtype == DataKind::Tnsr);

                let runtime = unsafe {
                    // Get op
                    let op =
                        (*g.model).get_or_create_activation(*a_t_data.meta, OpType_OP_RELU, true);
                    assert!(op != Op_INVALID_OP);
                    (*op.ptr).runtime.clone()
                };

                if self.ignore_all_weight_only && x(_a).all_weights {
                    self.all_weight_discount * runtime
                } else {
                    runtime
                }
            }

            Mdl::Tanh(_a) => {
                // Check types
                let a_t_data = x(_a);
                assert!(a_t_data.dtype == DataKind::Tnsr);

                let runtime = unsafe {
                    // Get op
                    let op =
                        (*g.model).get_or_create_activation(*a_t_data.meta, OpType_OP_TANH, true);
                    assert!(op != Op_INVALID_OP);
                    (*op.ptr).runtime.clone()
                };

                if self.ignore_all_weight_only && x(_a).all_weights {
                    self.all_weight_discount * runtime
                } else {
                    runtime
                }
            }

            Mdl::Sigmoid(_a) => {
                // Check types
                let a_t_data = x(_a);
                assert!(a_t_data.dtype == DataKind::Tnsr);

                let runtime = unsafe {
                    // Get op
                    let op = (*g.model).get_or_create_activation(
                        *a_t_data.meta,
                        OpType_OP_SIGMOID,
                        true,
                    );
                    assert!(op != Op_INVALID_OP);
                    (*op.ptr).runtime.clone()
                };

                if self.ignore_all_weight_only && x(_a).all_weights {
                    self.all_weight_discount * runtime
                } else {
                    runtime
                }
            }

            Mdl::Conv2d([_stride_h, _stride_w, _pad, _act, _inpt, _wght]) => {
                // conv2d(1,1,SAME,NONE, x, Iconv-const) resolves to x; nominal positive
                // cost (no taso op) so extraction prefers the bare x.
                if x(_wght).dtype == DataKind::Const {
                    return 1.0;
                }
                // Check types
                let _stride_h_data = x(_stride_h);
                let _stride_w_data = x(_stride_w);
                let _pad_data = x(_pad);
                let _act_data = x(_act);
                let _inpt_data = x(_inpt);
                let _wght_data = x(_wght);
                assert!(_stride_h_data.dtype == DataKind::Scalar);
                assert!(_stride_w_data.dtype == DataKind::Scalar);
                assert!(_pad_data.dtype == DataKind::Scalar);
                assert!(_act_data.dtype == DataKind::Scalar);
                assert!(_inpt_data.dtype == DataKind::Tnsr);
                assert!(_wght_data.dtype == DataKind::Tnsr);

                // Get arguments
                let stride_h = _stride_h_data.val;
                let stride_w = _stride_w_data.val;
                let padding: PaddingMode = _pad_data.val.try_into().unwrap();
                let activation: ActiMode = _act_data.val.try_into().unwrap();
                let runtime = unsafe {
                    let t_inpt = *_inpt_data.meta;
                    let t_wght = *_wght_data.meta;
                    // Get op
                    let op = (*g.model).get_or_create_conv2d(
                        t_inpt, t_wght, stride_h, stride_w, padding, activation,
                    );
                    assert!(op != Op_INVALID_OP);
                    (*op.ptr).runtime.clone()
                };

                if self.ignore_all_weight_only && x(_inpt).all_weights && x(_wght).all_weights {
                    self.all_weight_discount * runtime
                } else {
                    runtime
                }
            }

            Mdl::Ewadd([_a, _b]) => {
                // Check types
                let _a_data = x(_a);
                let _b_data = x(_b);
                assert!(_a_data.dtype == DataKind::Tnsr);
                assert!(_b_data.dtype == DataKind::Tnsr);

                // Get arguments
                let t_a = _a_data.meta;
                let t_b = _b_data.meta;
                let runtime = unsafe {
                    // Get op
                    let op = (*g.model).get_or_create_element(OpType_OP_EW_ADD, t_a, t_b);
                    assert!(op != Op_INVALID_OP);
                    (*op.ptr).runtime.clone()
                };

                if self.ignore_all_weight_only && x(_a).all_weights && x(_b).all_weights {
                    self.all_weight_discount * runtime
                } else {
                    runtime
                }
            }

            // Elementwise sub/max/min: same cost handling as ewadd, dispatched on op.
            Mdl::Ewsub([_a, _b]) | Mdl::Ewmax([_a, _b]) | Mdl::Ewmin([_a, _b]) => {
                let _a_data = x(_a);
                let _b_data = x(_b);
                assert!(_a_data.dtype == DataKind::Tnsr);
                assert!(_b_data.dtype == DataKind::Tnsr);
                let t_a = _a_data.meta;
                let t_b = _b_data.meta;
                let op_ty = match enode {
                    Mdl::Ewsub(_) => OpType_OP_EW_SUB,
                    Mdl::Ewmax(_) => OpType_OP_EW_MAX,
                    _ => OpType_OP_EW_MIN,
                };
                let runtime = unsafe {
                    let op = (*g.model).get_or_create_element(op_ty, t_a, t_b);
                    assert!(op != Op_INVALID_OP);
                    (*op.ptr).runtime.clone()
                };
                if self.ignore_all_weight_only && x(_a).all_weights && x(_b).all_weights {
                    self.all_weight_discount * runtime
                } else {
                    runtime
                }
            }

            Mdl::Ewmul([_a, _b]) => {
                // ewmul(x, Iewmul-const) is a no-op wrapper (resolves to x). Charge a
                // small positive self-cost -- no taso op, its Const child has no tensor --
                // so extraction strictly prefers the bare x over this wrapper.
                if x(_a).dtype == DataKind::Const || x(_b).dtype == DataKind::Const {
                    1.0
                } else {
                // Check types
                let _a_data = x(_a);
                let _b_data = x(_b);
                assert!(_a_data.dtype == DataKind::Tnsr);
                assert!(_b_data.dtype == DataKind::Tnsr);

                // Get arguments
                let t_a = _a_data.meta;
                let t_b = _b_data.meta;
                let runtime = unsafe {
                    // Get op
                    let op = (*g.model).get_or_create_element(OpType_OP_EW_MUL, t_a, t_b);
                    assert!(op != Op_INVALID_OP);
                    (*op.ptr).runtime.clone()
                };

                if self.ignore_all_weight_only && x(_a).all_weights && x(_b).all_weights {
                    self.all_weight_discount * runtime
                } else {
                    runtime
                }
                }
            }

            Mdl::Matmul([_act, _a, _b]) => {
                // matmul(x, Imatmul-const) resolves to x; nominal positive cost (no taso
                // op, its Const child has no tensor) so extraction prefers the bare x.
                if x(_a).dtype == DataKind::Const || x(_b).dtype == DataKind::Const {
                    return 1.0;
                }
                // Check types
                let _act_data = x(_act);
                let _a_data = x(_a);
                let _b_data = x(_b);
                assert!(_act_data.dtype == DataKind::Scalar);
                assert!(_a_data.dtype == DataKind::Tnsr);
                assert!(_b_data.dtype == DataKind::Tnsr);

                // Get arguments
                let activation: ActiMode = _act_data.val.try_into().unwrap();
                let runtime = unsafe {
                    let t_a = *_a_data.meta;
                    let t_b = *_b_data.meta;
                    // Get op
                    let op = (*g.model).get_or_create_matmul(t_a, t_b, activation);
                    assert!(op != Op_INVALID_OP);
                    (*op.ptr).runtime.clone()
                };

                if self.ignore_all_weight_only && x(_a).all_weights && x(_b).all_weights {
                    self.all_weight_discount * runtime
                } else {
                    runtime
                }
            }

            Mdl::Concat([_axis, _ndim, _a, _b]) => {
                // Check types
                let _axis_data = x(_axis);
                let _ndim_data = x(_ndim);
                let _a_data = x(_a);
                let _b_data = x(_b);
                assert!(_axis_data.dtype == DataKind::Scalar);
                assert!(_ndim_data.dtype == DataKind::Scalar);
                assert!(_a_data.dtype == DataKind::Tnsr);
                assert!(_b_data.dtype == DataKind::Tnsr);

                // Get arguments
                let axis = _axis_data.val;
                let ndim = _ndim_data.val;
                unsafe {
                    let t_a = *_a_data.meta;
                    let t_b = *_b_data.meta;

                    // Pass ownership to C++
                    let mut inputs = vec![t_a, t_b];
                    inputs.shrink_to_fit();
                    assert!(inputs.len() == inputs.capacity());
                    let ptr = inputs.as_mut_ptr();
                    std::mem::forget(inputs);

                    // Get op
                    let mut need_copy = if self.ignore_all_weight_only
                        && !(x(_a).all_weights && x(_b).all_weights)
                    {
                        [true, true]
                    } else {
                        [false, false]
                    };
                    let op = (*g.model).get_or_create_concat(axis, 2, ptr, need_copy.as_mut_ptr());
                    assert!(op != Op_INVALID_OP);
                    (*op.ptr).runtime.clone()
                }
            }

            Mdl::Concat3([_axis, _ndim, _input1, _input2, _input3]) => {
                // Check types
                let _axis_data = x(_axis);
                let _ndim_data = x(_ndim);
                let _input1_data = x(_input1);
                let _input2_data = x(_input2);
                let _input3_data = x(_input3);
                assert!(_axis_data.dtype == DataKind::Scalar);
                assert!(_ndim_data.dtype == DataKind::Scalar);
                assert!(_input1_data.dtype == DataKind::Tnsr);
                assert!(_input2_data.dtype == DataKind::Tnsr);
                assert!(_input3_data.dtype == DataKind::Tnsr);

                // Get arguments
                let axis = _axis_data.val;
                let ndim = _ndim_data.val;
                unsafe {
                    let t_1 = *_input1_data.meta;
                    let t_2 = *_input2_data.meta;
                    let t_3 = *_input3_data.meta;

                    // Pass ownership to C++
                    let mut inputs = vec![t_1, t_2, t_3];
                    inputs.shrink_to_fit();
                    assert!(inputs.len() == inputs.capacity());
                    let ptr = inputs.as_mut_ptr();
                    std::mem::forget(inputs);

                    // Get op
                    let mut need_copy = if self.ignore_all_weight_only
                        && !(x(_input1).all_weights
                            && x(_input2).all_weights
                            && x(_input3).all_weights)
                    {
                        [true, true, true]
                    } else {
                        [false, false, false]
                    };
                    let op = (*g.model).get_or_create_concat(axis, 3, ptr, need_copy.as_mut_ptr());
                    assert!(op != Op_INVALID_OP);
                    (*op.ptr).runtime.clone()
                }
            }

            Mdl::Concat4([_axis, _ndim, _input1, _input2, _input3, _input4]) => {
                // Check types
                let _axis_data = x(_axis);
                let _ndim_data = x(_ndim);
                let _input1_data = x(_input1);
                let _input2_data = x(_input2);
                let _input3_data = x(_input3);
                let _input4_data = x(_input4);
                assert!(_axis_data.dtype == DataKind::Scalar);
                assert!(_ndim_data.dtype == DataKind::Scalar);
                assert!(_input1_data.dtype == DataKind::Tnsr);
                assert!(_input2_data.dtype == DataKind::Tnsr);
                assert!(_input3_data.dtype == DataKind::Tnsr);
                assert!(_input4_data.dtype == DataKind::Tnsr);

                // Get arguments
                let axis = _axis_data.val;
                let ndim = _ndim_data.val;
                unsafe {
                    let t_1 = *_input1_data.meta;
                    let t_2 = *_input2_data.meta;
                    let t_3 = *_input3_data.meta;
                    let t_4 = *_input4_data.meta;

                    // Pass ownership to C++
                    let mut inputs = vec![t_1, t_2, t_3, t_4];
                    inputs.shrink_to_fit();
                    assert!(inputs.len() == inputs.capacity());
                    let ptr = inputs.as_mut_ptr();
                    std::mem::forget(inputs);

                    // Get op
                    let mut need_copy = if self.ignore_all_weight_only
                        && !(x(_input1).all_weights
                            && x(_input2).all_weights
                            && x(_input3).all_weights
                            && x(_input4).all_weights)
                    {
                        [true, true, true, true]
                    } else {
                        [false, false, false, false]
                    };
                    let op = (*g.model).get_or_create_concat(axis, 4, ptr, need_copy.as_mut_ptr());
                    assert!(op != Op_INVALID_OP);
                    (*op.ptr).runtime.clone()
                }
            }

            Mdl::Concat5([_axis, _ndim, _input1, _input2, _input3, _input4, _input5]) => {
                // Check types
                let _axis_data = x(_axis);
                let _ndim_data = x(_ndim);
                let _input1_data = x(_input1);
                let _input2_data = x(_input2);
                let _input3_data = x(_input3);
                let _input4_data = x(_input4);
                let _input5_data = x(_input5);
                assert!(_axis_data.dtype == DataKind::Scalar);
                assert!(_ndim_data.dtype == DataKind::Scalar);
                assert!(_input1_data.dtype == DataKind::Tnsr);
                assert!(_input2_data.dtype == DataKind::Tnsr);
                assert!(_input3_data.dtype == DataKind::Tnsr);
                assert!(_input4_data.dtype == DataKind::Tnsr);
                assert!(_input5_data.dtype == DataKind::Tnsr);

                // Get arguments
                let axis = _axis_data.val;
                let ndim = _ndim_data.val;
                unsafe {
                    let t_1 = *_input1_data.meta;
                    let t_2 = *_input2_data.meta;
                    let t_3 = *_input3_data.meta;
                    let t_4 = *_input4_data.meta;
                    let t_5 = *_input5_data.meta;

                    // Pass ownership to C++
                    let mut inputs = vec![t_1, t_2, t_3, t_4, t_5];
                    inputs.shrink_to_fit();
                    assert!(inputs.len() == inputs.capacity());
                    let ptr = inputs.as_mut_ptr();
                    std::mem::forget(inputs);

                    // Get op
                    let mut need_copy = if self.ignore_all_weight_only
                        && !(x(_input1).all_weights
                            && x(_input2).all_weights
                            && x(_input3).all_weights
                            && x(_input4).all_weights
                            && x(_input5).all_weights)
                    {
                        [true, true, true, true, true]
                    } else {
                        [false, false, false, false, false]
                    };
                    let op = (*g.model).get_or_create_concat(axis, 5, ptr, need_copy.as_mut_ptr());
                    assert!(op != Op_INVALID_OP);
                    (*op.ptr).runtime.clone()
                }
            }

            Mdl::Poolmax([_inpt, _kernel_h, _kernel_w, _stride_h, _stride_w, _pad, _act]) => {
                // Check types
                let _kernel_h_data = x(_kernel_h);
                let _kernel_w_data = x(_kernel_w);
                let _stride_h_data = x(_stride_h);
                let _stride_w_data = x(_stride_w);
                let _pad_data = x(_pad);
                let _act_data = x(_act);
                let _inpt_data = x(_inpt);
                assert!(_kernel_h_data.dtype == DataKind::Scalar);
                assert!(_kernel_w_data.dtype == DataKind::Scalar);
                assert!(_stride_h_data.dtype == DataKind::Scalar);
                assert!(_stride_w_data.dtype == DataKind::Scalar);
                assert!(_pad_data.dtype == DataKind::Scalar);
                assert!(_act_data.dtype == DataKind::Scalar);
                assert!(_inpt_data.dtype == DataKind::Tnsr);

                // Get arguments
                let kernel_h = _kernel_h_data.val;
                let kernel_w = _kernel_w_data.val;
                let stride_h = _stride_h_data.val;
                let stride_w = _stride_w_data.val;
                let padding: PaddingMode = _pad_data.val.try_into().unwrap();
                let activation: ActiMode = _act_data.val.try_into().unwrap();
                let runtime = unsafe {
                    let t_inpt = *_inpt_data.meta;
                    let t_wght = t_inpt.clone(); // Just a placeholder, t_wght won't be used in get_or_create_pool2d here

                    // Get op
                    let op = (*g.model).get_or_create_pool2d(
                        t_inpt,
                        t_wght,
                        OpType_OP_POOL2D_MAX,
                        kernel_h,
                        kernel_w,
                        stride_h,
                        stride_w,
                        padding,
                        activation,
                    );
                    assert!(op != Op_INVALID_OP);
                    (*op.ptr).runtime.clone()
                };

                if self.ignore_all_weight_only && x(_inpt).all_weights {
                    self.all_weight_discount * runtime
                } else {
                    runtime
                }
            }

            Mdl::Poolavg([_inpt, _kernel_h, _kernel_w, _stride_h, _stride_w, _pad, _act]) => {
                // Check types
                let _kernel_h_data = x(_kernel_h);
                let _kernel_w_data = x(_kernel_w);
                let _stride_h_data = x(_stride_h);
                let _stride_w_data = x(_stride_w);
                let _pad_data = x(_pad);
                let _act_data = x(_act);
                let _inpt_data = x(_inpt);
                assert!(_kernel_h_data.dtype == DataKind::Scalar);
                assert!(_kernel_w_data.dtype == DataKind::Scalar);
                assert!(_stride_h_data.dtype == DataKind::Scalar);
                assert!(_stride_w_data.dtype == DataKind::Scalar);
                assert!(_pad_data.dtype == DataKind::Scalar);
                assert!(_act_data.dtype == DataKind::Scalar);
                assert!(_inpt_data.dtype == DataKind::Tnsr);

                // Get arguments
                let kernel_h = _kernel_h_data.val;
                let kernel_w = _kernel_w_data.val;
                let stride_h = _stride_h_data.val;
                let stride_w = _stride_w_data.val;
                let padding: PaddingMode = _pad_data.val.try_into().unwrap();
                let activation: ActiMode = _act_data.val.try_into().unwrap();
                let runtime = unsafe {
                    let t_inpt = *_inpt_data.meta;
                    let t_wght = t_inpt.clone(); // Just a placeholder, t_wght won't be used in get_or_create_pool2d here

                    // Get op
                    let op = (*g.model).get_or_create_pool2d(
                        t_inpt,
                        t_wght,
                        OpType_OP_POOL2D_AVG,
                        kernel_h,
                        kernel_w,
                        stride_h,
                        stride_w,
                        padding,
                        activation,
                    );
                    assert!(op != Op_INVALID_OP);
                    (*op.ptr).runtime.clone()
                };

                if self.ignore_all_weight_only && x(_inpt).all_weights {
                    self.all_weight_discount * runtime
                } else {
                    runtime
                }
            }

            Mdl::Split([_axis, _inpt]) => {
                // Check types
                let _axis_data = x(_axis);
                let _inpt_data = x(_inpt);
                assert!(_axis_data.dtype == DataKind::Scalar);
                assert!(_inpt_data.dtype == DataKind::Tnsr);

                // Get arguments
                let t_inpt = _inpt_data.meta;
                let axis = _axis_data.val;
                let runtime = unsafe {
                    // Get op
                    let op = (*g.model).get_or_create_split1(t_inpt, axis, 2);
                    assert!(op != Op_INVALID_OP);
                    (*op.ptr).runtime.clone()
                };

                if self.ignore_all_weight_only && x(_inpt).all_weights {
                    self.all_weight_discount * runtime
                } else {
                    runtime
                }
            }

            Mdl::Enlarge([_a, _b]) => {
                // Check types
                let _a_data = x(_a);
                let _b_data = x(_b);
                assert!(_a_data.dtype == DataKind::Tnsr);
                assert!(_b_data.dtype == DataKind::Tnsr);

                // Get arguments
                let runtime = unsafe {
                    let t_a = *_a_data.meta;
                    let t_b = *_b_data.meta;
                    // Get op
                    let op = (*g.model).get_or_create_enlarge(t_a, t_b);
                    assert!(op != Op_INVALID_OP);
                    (*op.ptr).runtime.clone()
                };

                if self.ignore_all_weight_only && x(_a).all_weights && x(_b).all_weights {
                    self.all_weight_discount * runtime
                } else {
                    runtime
                }
            }

            Mdl::BatchNorm([_inpt, _scale, _bias, _mean, _var]) => {
                // Check types
                let _inpt_data = x(_inpt);
                let _scale_data = x(_scale);
                let _bias_data = x(_bias);
                let _mean_data = x(_mean);
                let _var_data = x(_var);
                assert!(_inpt_data.dtype == DataKind::Tnsr);
                assert!(_scale_data.dtype == DataKind::Tnsr);
                assert!(_bias_data.dtype == DataKind::Tnsr);
                assert!(_mean_data.dtype == DataKind::Tnsr);
                assert!(_var_data.dtype == DataKind::Tnsr);

                // Get arguments
                let runtime = unsafe {
                    let t_inpt = *_inpt_data.meta;
                    let t_scale = *_scale_data.meta;
                    let t_bias = *_bias_data.meta;
                    let t_mean = *_mean_data.meta;
                    let t_var = *_var_data.meta;
                    // Get op
                    let op = (*g.model).get_or_create_batchnorm(t_inpt, t_scale, t_bias, t_mean, t_var);
                    assert!(op != Op_INVALID_OP);
                    (*op.ptr).runtime.clone()
                };

                if self.ignore_all_weight_only && x(_inpt).all_weights && x(_scale).all_weights && x(_bias).all_weights && x(_mean).all_weights && x(_var).all_weights {
                    self.all_weight_discount * runtime
                } else {
                    runtime
                }
            }

            other => {
                println!("Get cost not implemented for: {:?}", other);
                0.0
            }
        };
        // Two structurally different multi-pattern rewrites both produce
        // Concat/Split nodes: the parallel-conv-fusion rule (enlarge two
        // weight kernels, concat them on the *output-channel* axis, one
        // wider conv, split the channels back apart -- always safe, never
        // touches axis 0 of any real activation) and a separate
        // relu-merge rule that has both a channel-axis variant and a
        // batch-axis variant (concat two *activations* together on axis 0,
        // relu once, split back apart). The batch-axis variant is exactly
        // what BUGS.md #11 found unverifiable by alpha-beta-CROWN (and by
        // plain batch>1 ONNX inference).
        //
        // Merely *not* discounting axis-0 Concat/Split (an earlier version
        // of this code) isn't enough to avoid it: confirmed by testing
        // that greedy extraction picks the batch-axis relu-merge even with
        // --favor_fusion off entirely, i.e. under the real, undiscounted
        // cost model -- it's a genuine reuse win there (the batched relu
        // subsumes a relu that's also needed directly elsewhere), not an
        // artifact of any discount. So axis-0 Concat/Split needs an active
        // penalty, not just neutral treatment, to be reliably excluded.
        // Enlarge only ever operates on weight kernels' spatial dims (see
        // enlarge.cc), never a batch axis, so it's always safe to favor.
        let axis0_concat_or_split = match enode {
            Mdl::Concat([axis, ..])
            | Mdl::Concat3([axis, ..])
            | Mdl::Concat4([axis, ..])
            | Mdl::Concat5([axis, ..])
            | Mdl::Split([axis, ..]) => x(axis).val == 0,
            _ => false,
        };
        let is_favored_fusion_op = match enode {
            Mdl::Concat([axis, ..])
            | Mdl::Concat3([axis, ..])
            | Mdl::Concat4([axis, ..])
            | Mdl::Concat5([axis, ..])
            | Mdl::Split([axis, ..]) => x(axis).val != 0,
            Mdl::Enlarge(_) => true,
            _ => false,
        };
        let neutral = self.favor_fusion_strength == 1.0;
        if !neutral && axis0_concat_or_split {
            // Always a fixed 1000x penalty regardless of strength: axis-0
            // Concat/Split is unverifiable by design (BUGS.md #11), so
            // there's never a reason to bias sampling *toward* it, no
            // matter how strongly we're favoring fusion in general.
            cost * 1000.0
        } else if !neutral && is_favored_fusion_op {
            cost * self.favor_fusion_strength
        } else {
            cost
        }
    }
}

/// Prepare the data for formulation ILP
///
/// # Returns
///
/// - `m_id_map`: list of EClass Id's each index m refers to
/// - `e_m`: each entry is the list of nodes i within eclass m
/// - `h_i`: each entry is the list of children EClass indices for node i
/// - `cost_i`: self cost for each node i
/// - `g_i`: which EClass index does node i belong to
/// - `root_m`: EClass index of the root eclass
/// - `i_to_nodes: Vector of enodes, ordered by index i
/// - `blacklist_i: Vector of indices of nodes that are blacklisted
pub fn prep_ilp_data(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    root: Id,
    cost_model: &CostModel,
) -> (
    Vec<Id>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    Vec<f32>,
    Vec<usize>,
    usize,
    Vec<Mdl>,
    Vec<usize>,
) {
    let m_id_map: Vec<Id> = egraph.classes().map(|c| egraph.find(c.id)).collect();
    assert!(m_id_map.len() == egraph.number_of_classes());
    let id_m_map: HashMap<Id, usize> = m_id_map
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    let num_classes = egraph.number_of_classes();
    let num_nodes = egraph.total_size();
    let mut i_to_nodes: Vec<Mdl> = Vec::with_capacity(num_nodes);
    let mut e_m: Vec<Vec<usize>> = vec![Vec::new(); num_classes];
    let mut h_i: Vec<Vec<usize>> = Vec::with_capacity(num_nodes);
    let mut cost_i: Vec<f32> = Vec::with_capacity(num_nodes);
    let mut g_i: Vec<usize> = Vec::with_capacity(num_nodes);
    let mut blacklist_i: Vec<usize> = Vec::new();

    let mut i = 0;
    for class in egraph.classes() {
        let m = *id_m_map.get(&egraph.find(class.id)).unwrap();
        for node in class.iter() {
            i_to_nodes.push(node.clone());
            if egraph.analysis.blacklist_nodes.contains(node) {
                blacklist_i.push(i);
            }
            e_m[m].push(i);
            h_i.push(
                node.children()
                    .iter()
                    .map(|id| *id_m_map.get(&egraph.find(*id)).unwrap())
                    .collect(),
            );
            cost_i.push(cost_model.get_self_cost(egraph, node));
            g_i.push(m);
            i += 1;
        }
    }

    let root_m = *id_m_map.get(&egraph.find(root)).unwrap();

    (
        m_id_map,
        e_m,
        h_i,
        cost_i,
        g_i,
        root_m,
        i_to_nodes,
        blacklist_i,
    )
}

/// Struct for storing the solved results from ILP
#[derive(Debug, Serialize, Deserialize)]
pub struct SolvedResults {
    /// The solved values for the variables associated with each node
    pub solved_x: Vec<i32>,
    /// The minimum total cost found
    pub cost: f32,
    /// Time for solver
    pub time: f32,
}

/// Construct the RecExpr of the optimized graph extracted
///
/// This function does the construction recursively with memoization. Call it with eclass=root
/// will construct the whole extracted graph
///
/// # Parameters
///
/// - `node_picked`: hashmap storing which node is picked for each EClass ID
/// - `eclass`: The EClass ID that we aim to construct as root
/// - `added_memo`: Map from EClass ID to RecExpr ID. Storing the eclasses that were already added
/// - `egraph`: E-graph of interest
/// - `expr`: the RecExpr storing the optimized graph, it is constructed within this function
///
/// # Returns
///
/// - The ID (index) in the output RecExpr for the eclass passed in as argument
pub fn construct_best_rec(
    node_picked: &HashMap<Id, Mdl>,
    eclass: Id,
    added_memo: &mut HashMap<Id, Id>,
    egraph: &EGraph<Mdl, TensorAnalysis>,
    expr: &mut RecExpr<Mdl>,
) -> Id {
    let id = egraph.find(eclass);

    match added_memo.get(&id) {
        Some(id_expr) => *id_expr,
        None => {
            let node = node_picked.get(&id).unwrap().clone().map_children(|child| {
                construct_best_rec(node_picked, child, added_memo, egraph, expr)
            });
            let id_expr = expr.add(node);
            assert!(added_memo.insert(id, id_expr).is_none());
            id_expr
        }
    }
}

/// Get the initial solution for ILP using the greedy extraction
///
/// This function does the construction recursively with memoization. Call it with eclass=root
/// will construct the whole extracted graph
///
/// # Parameters
///
/// - `egraph`: egraph of interest
/// - `root`: root eclass
/// - `costs`: Map from eclass ID to the node with the lowest subtree cost (cost, node).
///         Constructed by egg's Extractor
/// - `g_i`: which EClass index does node i belong to
/// - `nodes_to_i`: map from node to index i
///
/// # Returns
///
/// A tuple of (i_list, m_list), where
///
/// - `i_list`: list of i picked by greedy extraction
/// - `m_list`: list of eclass index m that i_list belongs to
pub fn get_init_solution(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    root: Id,
    costs: &HashMap<Id, (f32, Mdl)>,
    g_i: &[usize],
    nodes_to_i: &HashMap<Mdl, usize>,
) -> (Vec<usize>, Vec<usize>) {
    let mut nodes: Vec<Mdl> = Vec::new();
    // added_memo maps eclass id to id in expr
    let mut added_memo: HashSet<Id> = Default::default();
    get_init_rec(egraph, root, &mut added_memo, costs, &mut nodes);

    let i_list: Vec<usize> = nodes
        .iter()
        .map(|node| *nodes_to_i.get(node).unwrap())
        .collect();
    let m_list: Vec<usize> = i_list.iter().map(|i| g_i[*i]).collect();

    (i_list, m_list)
}

/// Recursively get the initial solution for ILP using the greedy extraction, results stored in nodes
///
/// # Parameters
///
/// - `egraph`: egraph of interest
/// - `eclass`: get solution rooted from here
/// - `added_memo`: Stores the set of eclasses that has already been processed
/// - `costs`: Map from eclass ID to the node with the lowest subtree cost (cost, node).
///         Constructed by egg's Extractor
/// - `nodes`: List of nodes picked by greedy extraction. Constructed within this function
fn get_init_rec(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    eclass: Id,
    added_memo: &mut HashSet<Id>,
    costs: &HashMap<Id, (f32, Mdl)>,
    nodes: &mut Vec<Mdl>,
) {
    let id = egraph.find(eclass);

    if !added_memo.contains(&id) {
        let (_, best_node) = match costs.get(&id) {
            Some(result) => result.clone(),
            None => panic!("Failed to extract from eclass {}", id),
        };
        best_node.for_each(|child| get_init_rec(egraph, child, added_memo, costs, nodes));
        nodes.push(best_node);
        added_memo.insert(id);
    }
}
