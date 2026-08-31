use clap::{App, Arg};
use egg::*;
use std::collections::{HashMap, HashSet};
use std::env::*;
use std::fs::*;
use std::time::*;
use std::time::{Duration, Instant};
use tensat::bert;
use tensat::model::*;
use tensat::nasneta;
use tensat::nasrnn;
use tensat::optimize::*;
use tensat::resnet50;
use tensat::resnext50;
use tensat::rewrites::*;
use tensat::inceptionv3;
use tensat::mobilenetv2;
use tensat::vgg;
use tensat::squeezenet;
use tensat::{parse::*, verify::*};

use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::Error;
use std::process::{Command, Stdio};
use std::thread;

use std::ffi::CString;

fn main() {
    // Parse arguments
    let matches = App::new("Tamago")
        .arg(
            Arg::with_name("mode")
                .short("m")
                .long("mode")
                .takes_value(true)
                .default_value("optimize")
                .help("Mode to run, can be verify, optimize, test, convert"),
        )
        .arg(
            Arg::with_name("model")
                .short("d")
                .long("model")
                .takes_value(true)
                .help("Specify a pre-defined model to optimize"),
        )
        .arg(
            Arg::with_name("rules")
                .short("r")
                .long("rules")
                .takes_value(true)
                .help("Provide a file with rewrite rules"),
        )
        .arg(
            Arg::with_name("out_file")
                .short("o")
                .long("out_file")
                .takes_value(true)
                .help("Provide a output file name. For mode convert, it's for converted rules; for mode optimize, it's for measured runtime"),
        )
        .arg(
            Arg::with_name("export_model")
                .short("x")
                .long("export_model")
                .takes_value(true)
                .help("Provide a file name to store optimized model"),
        )
        .arg(
            Arg::with_name("model_file")
                .short("f")
                .long("model_file")
                .takes_value(true)
                .help("Provide a file with the input model"),
        )
        .arg(
            Arg::with_name("multi_rules")
                .short("t")
                .long("multi_rules")
                .takes_value(true)
                .help("File with multi-pattern rules. Every two lines belong to one multi-pattern rule"),
        )
        .arg(
            Arg::with_name("save_graph")
                .short("s")
                .long("save_graph")
                .takes_value(true)
                .default_value("io")
                .help("Whether to save graphs as dot files. Can be: all, io, none"),
        )
        .arg(
            Arg::with_name("use_multi")
                .short("u")
                .long("use_multi")
                .help("Set this flag will enable use of multi-pattern rules"),
        )
        .arg(
            Arg::with_name("n_iter")
                .long("n_iter")
                .takes_value(true)
                .default_value("3")
                .help("Max number of iterations for egg to run"),
        )
        .arg(
            Arg::with_name("n_sec")
                .long("n_sec")
                .takes_value(true)
                .default_value("10")
                .help("Max number of seconds for egg to run"),
        )
        .arg(
            Arg::with_name("n_nodes")
                .long("n_nodes")
                .takes_value(true)
                .default_value("100000")
                .help("Max number of nodes for egraph"),
        )
        .arg(
            Arg::with_name("extract")
                .short("e")
                .long("extract")
                .takes_value(true)
                .default_value("greedy")
                .help("Extraction method, can be greedy, ilp"),
        )
        .arg(
            Arg::with_name("order_var_int")
                .long("order_var_int")
                .help("Set this flag will let ILP use integer var for ordering"),
        )
        .arg(
            Arg::with_name("class_constraint")
                .long("class_constraint")
                .help("Add constraint in ILP that each eclass sum to 1"),
        )
        .arg(
            Arg::with_name("no_order")
                .long("no_order")
                .help("No ordering constraints in ILP"),
        )
        .arg(
            Arg::with_name("initial_with_greedy")
                .long("initial_with_greedy")
                .help("Initialize ILP with greedy solution"),
        )
        .arg(
            Arg::with_name("ilp_time_sec")
                .long("ilp_time_sec")
                .takes_value(true)
                .help("Time limit for ILP solver (seconds)"),
        )
        .arg(
            Arg::with_name("ilp_num_threads")
                .long("ilp_num_threads")
                .takes_value(true)
                .help("Number of threads for ILP solver"),
        )
        .arg(
            Arg::with_name("iter_multi")
                .long("iter_multi")
                .takes_value(true)
                .default_value("1")
                .help("Max number of iterations to apply multi-pattern rules"),
        )
        .arg(
            Arg::with_name("node_multi")
                .long("node_multi")
                .takes_value(true)
                .default_value("3000000")
                .help("Max number of nodes added by multi-pattern rules"),
        )
        .arg(
            Arg::with_name("no_cycle")
                .long("no_cycle")
                .help("Not allowing cycles in EGraph"),
        )
        .arg(
            Arg::with_name("filter_before")
                .long("filter_before")
                .help("Filter cycles before applying rules"),
        )
        .arg(
            Arg::with_name("all_weight_only")
                .long("all_weight_only")
                .help("Treat zero cost for all weight concat only"),
        )
        .arg(
            Arg::with_name("saturation_only")
                .long("saturation_only")
                .help("Run saturation only"),
        )
        .arg(
            Arg::with_name("favor_fusion")
                .long("favor_fusion")
                .help("Deliberately discount Concat/Split/Enlarge's real measured cost \
                       during extraction, so a legally-available multi-pattern-rule \
                       fusion gets picked even though it's more real ops than what it \
                       replaces (and so never wins under the unmodified cost model). \
                       For pulling out an already-proven-valid equivalence to compare \
                       against the unfused baseline, not a real cost-model claim. \
                       Discount amount is --favor_fusion_strength (default 0.05 if this \
                       flag is present but --favor_fusion_strength isn't given)."),
        )
        .arg(
            Arg::with_name("favor_fusion_strength")
                .long("favor_fusion_strength")
                .takes_value(true)
                .help("Only used with --favor_fusion. Continuous discount factor for \
                       axis!=0 Concat/Split/Enlarge's real cost (CostModel::with_favor_fusion_strength \
                       in optimize.rs) -- smaller values favor fusion more strongly. \
                       Sampling across a range of values (rather than relying on a single \
                       fixed discount plus jitter/diversity-penalty noise, which in \
                       practice essentially never stumbles into a fused structure on its \
                       own) is how the structural-diversity-vs-verifiability campaign \
                       gets samples that reliably span unfused through fused. Default \
                       0.05 (the old fixed --favor_fusion discount) when --favor_fusion \
                       is present without this flag."),
        )
        .arg(
            Arg::with_name("n_random")
                .long("n_random")
                .takes_value(true)
                .help("Instead of extracting a single best graph, sample this many \
                       different (but egraph-equivalent) graphs at random from the \
                       saturated egraph -- for comparing equivalent graphs against each \
                       other rather than finding the single cheapest one. Writes \
                       <export_model>_random0.model, _random1.model, etc."),
        )
        .arg(
            Arg::with_name("random_seed")
                .long("random_seed")
                .takes_value(true)
                .default_value("0")
                .help("Base seed for --n_random/--n_diverse sampling (sample i uses seed base+i)"),
        )
        .arg(
            Arg::with_name("random_mode")
                .long("random_mode")
                .takes_value(true)
                .default_value("jitter")
                .possible_values(&["jitter", "uniform"])
                .help("Only used with --n_random. 'jitter' (default, unchanged prior \
                       behavior): RandomCost, jitters the real per-node cost by a random \
                       multiplier -- still fundamentally shaped by the real cost model. \
                       'uniform': UniformRandomCost, i.i.d. random cost per enode \
                       independent of the real cost model entirely -- a structure-agnostic \
                       baseline for comparison, at the cost of a known bias toward smaller \
                       trees (see UniformRandomCost's doc-comment in optimize.rs)."),
        )
        .arg(
            Arg::with_name("n_diverse")
                .long("n_diverse")
                .takes_value(true)
                .help("Instead of extracting a single best graph, sample this many \
                       graphs in sequence, each penalized against re-using any enode a \
                       *previous* sample in this sequence already used -- pushes \
                       successive samples toward structurally distinct regions of the \
                       egraph rather than noisy perturbations of the same near-optimal \
                       tree (DiverseCost in optimize.rs). Writes \
                       <export_model>_diverse0.model, _diverse1.model, etc."),
        )
        .arg(
            Arg::with_name("n_arch_diverse")
                .long("n_arch_diverse")
                .takes_value(true)
                .help("Architecture-diverse sampling: emit one extraction per \
                       multi-pattern rewrite FAMILY reachable in the egraph (plus a \
                       baseline sample 0 with no target). Unlike --n_diverse (which \
                       only penalizes reused enodes and so only ever re-emits the \
                       unfused baseline), each sample REWARDS a target rule's \
                       rewrite-witness enodes so that fusion actually wins its \
                       e-class -- see ArchDiverseCost in optimize.rs. N caps the \
                       number of samples. Writes <export_model>_arch0.model, etc."),
        )
        .arg(
            Arg::with_name("arch_reward")
                .long("arch_reward")
                .takes_value(true)
                .default_value("100000")
                .help("Only with --n_arch_diverse. Additive discount applied to the \
                       target rule's witness enodes (clamped so self-cost >= 0), to \
                       flip extraction onto that fusion family."),
        )
        .arg(
            Arg::with_name("arch_penalty")
                .long("arch_penalty")
                .takes_value(true)
                .default_value("1000000")
                .help("Only with --n_arch_diverse. Additive penalty on witness enodes \
                       of already-covered rule families, to push each new sample onto \
                       an uncovered family."),
        )
        .arg(
            Arg::with_name("verif_cost")
                .long("verif_cost")
                .takes_value(false)
                .help("Verifiability-aware extraction: one extraction minimizing the \
                       summed ReLU relaxation gap-area (VerifCost in optimize.rs), \
                       steering toward the ReLU topology with fewest/smallest unstable \
                       ReLUs. Requires --interval_file. Exports {export}_verif.model."),
        )
        .arg(
            Arg::with_name("interval_file")
                .long("interval_file")
                .takes_value(true)
                .help("JSON for --verif_cost: {\"w_0,w_1\": {\"lo\":[..],\"hi\":[..]}, ..} \
                       mapping each affine leaf's sorted weight-name set to its \
                       element-wise IBP interval over the input box."),
        )
        .arg(
            Arg::with_name("sensitivity_file")
                .long("sensitivity_file")
                .takes_value(true)
                .help("Optional JSON for --verif_cost: {\"w_0,..,w_15\": 0.83, ..} mapping a \
                       node's OUTPUT weight-name set to its backward-CROWN sensitivity \
                       |lambda|, weighting that ReLU's gap. Omit => unweighted."),
        )
        .arg(
            Arg::with_name("query_chain")
                .long("query_chain")
                .takes_value(false)
                .help("Diagnostic: after saturation, query whether the left-deep CHAIN \
                       association of the (min-of-max) start graph is present in the \
                       e-graph. Reports natural-order break depth, an order-independent \
                       subset closure, blacklist membership, and root-equivalence. Read \
                       the printed Stopped: reason for the budget verdict."),
        )
        .arg(
            Arg::with_name("redundancy_iters")
                .long("redundancy_iters")
                .takes_value(true)
                .default_value("4")
                .help("For --mode redundancy: the reachability/budget knob. A rule is \
                       pruned only if the OTHER rules re-derive its LHS=RHS within this \
                       many e-graph iterations. Small => prune only short-derivation \
                       redundancies (keeps shortcuts, preserves reachability); large => \
                       prune aggressively (smaller set, worse budget-reachability)."),
        )
        .arg(
            Arg::with_name("weight_names_json")
                .long("weight_names_json")
                .takes_value(true)
                .help("Only used with -f/--model_file. JSON object mapping guid \
                       (as it appears in that .model file, string-encoded) to a \
                       real weight name, e.g. {\"101\": \"stem.weight\"}. Seeds \
                       real weight identity at parse time instead of the default \
                       synthetic \"w_N\" naming, so it propagates through \
                       TensorAnalysis's weight_names provenance field for every \
                       later rewrite/extraction of this graph."),
        )
        .arg(
            Arg::with_name("no_runtime_report")
                .long("no_runtime_report")
                .help("Skip evaluating full graph runtime (before/after) after extraction. \
                       This calls TASO's real op execution (Graph::run/preprocess_weights), \
                       which a CPU-only TASO build (USE_CUDA=OFF) cannot do -- pass this flag \
                       to stop after extraction instead of hitting that."),
        )
        .get_matches();

    let run_mode = matches.value_of("mode").unwrap();
    println!("Running mode is: {}", run_mode);

    match run_mode {
        "optimize" => optimize(matches),
        "verify" => prove_taso_rules(matches),
        "test" => test(matches),
        "convert" => convert_learned_rules(matches),
        "redundancy" => prune_redundant(matches),
        "parse_check" => parse_check(matches),
        _ => panic!("Running mode not supported"),
    }
}

fn convert_learned_rules(matches: clap::ArgMatches) {
    env_logger::init();

    let file = matches
        .value_of("rules")
        .expect("Pls supply taso rules file.");
    let outf = matches.value_of("out_file").unwrap_or("converted.txt");
    let taso_rules = read_to_string(file).expect("Something went wrong reading the file");

    let converted = parse_and_convert(&taso_rules);

    write(outf, converted).expect("Unable to write file");
}

fn test(matches: clap::ArgMatches) {}

/// Resolves --favor_fusion/--favor_fusion_strength into the continuous
/// strength CostModel::with_favor_fusion_strength expects: 1.0 (neutral,
/// no bias) if --favor_fusion wasn't passed at all; otherwise
/// --favor_fusion_strength's value, or 0.05 (the old fixed discount) if
/// --favor_fusion was passed without an explicit strength.
fn favor_fusion_strength_from_matches(matches: &clap::ArgMatches) -> f32 {
    if !matches.is_present("favor_fusion") {
        return 1.0;
    }
    matches
        .value_of("favor_fusion_strength")
        .unwrap_or("0.05")
        .parse()
        .expect("--favor_fusion_strength must be a float")
}

/// Main procedure to run optimization
///
/// Gets input graph and rewrite rules; runs saturation with TensorAnalysis dealing with metadata; runs
/// greedy extraction with TensorCost getting the cost per node/op; evaluates
/// full graph runtime of the starting graph and extracted graph.
fn optimize(matches: clap::ArgMatches) {
    env_logger::init();

    // Read settings from args
    let rule_file = matches
        .value_of("rules")
        .expect("Pls supply rewrite rules file.");
    let save_graph = matches.value_of("save_graph").unwrap();
    let use_multi = matches.is_present("use_multi");
    let no_cycle = matches.is_present("no_cycle");
    let filter_after = !matches.is_present("filter_before");
    let no_runtime_report = matches.is_present("no_runtime_report");

    // Get input graph and rules
    // learned_rules are the learned rules from TASO, pre_defined_rules are the hand-specified rules from TASO
    let learned_rules =
        read_to_string(rule_file).expect("Something went wrong reading the rule file");
    let pre_defined_rules = PRE_DEFINED_RULES.iter().map(|&x| x);
    // filter empty lines (e.g. a trailing newline) -- they would panic "".parse()
    let split_rules: Vec<&str> = learned_rules
        .split("\n")
        .filter(|l| !l.trim().is_empty())
        .chain(pre_defined_rules)
        .collect();
    let do_filter_after = no_cycle && filter_after;
    let rules = rules_from_str(split_rules, do_filter_after);

    let start = match matches.value_of("model") {
        Some("resnet50") => resnet50::get_resnet50(),
        Some("nasrnn") => nasrnn::get_nasrnn(),
        Some("resnext50") => resnext50::get_resnext50(),
        Some("bert") => bert::get_bert(),
        Some("nasneta") => nasneta::get_nasneta(),
        Some("inceptionv3") => inceptionv3::get_inceptionv3(),
        Some("mobilenetv2") => mobilenetv2::get_mobilenetv2(),
        Some("vgg") => vgg::get_vgg(),
        Some("squeezenet") => squeezenet::get_squeezenet(),
        Some(_) => panic!("The model name is not supported"),
        None => {
            let model_file = matches
                .value_of("model_file")
                .expect("Pls supply input graph file.");
            let input_graph =
                read_to_string(model_file).expect("Something went wrong reading the model file");
            // Was: input_graph.parse().unwrap(), i.e. generic RecExpr::from_str
            // (egg's own S-expression grammar). That's a different format than
            // what taso.export_to_file() actually writes (see tests/parse.rs),
            // so -f silently produced a degenerate one-node graph on any real
            // exported model instead of erroring. parse_model() is the parser
            // built for that format; route through it instead.
            match matches.value_of("weight_names_json") {
                Some(names_path) => {
                    let names_json = read_to_string(names_path)
                        .expect("Something went wrong reading --weight_names_json");
                    let guid_names: HashMap<usize, String> = serde_json::from_str(&names_json)
                        .expect("--weight_names_json must be a JSON object of {\"guid\": \"name\"}");
                    parse_model_with_names(&input_graph, &guid_names).rec_expr()
                }
                None => parse_model(&input_graph).rec_expr(),
            }
        }
    };

    // Get multi-pattern rules. learned_rules are the learned rules from TASO,
    // pre_defined_multi are the hand-specified rules from TASO
    let n_sec = matches.value_of("n_sec").unwrap().parse::<u64>().unwrap();
    let iter_multi = matches
        .value_of("iter_multi")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let node_multi = matches
        .value_of("node_multi")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let mut multi_patterns = if let Some(rule_file) = matches.value_of("multi_rules") {
        let learned_rules =
            read_to_string(rule_file).expect("Something went wrong reading the rule file");
        let pre_defined_multi = PRE_DEFINED_MULTI.iter().map(|&x| (x, /*symmetric=*/ false));
        // The learned rules we have are symmetric. Predefined ones are not
        let multi_rules: Vec<(&str, bool)> = learned_rules
            .split("\n")
            .filter(|l| !l.trim().is_empty())
            .map(|x| (x, /*symmetric=*/ true))
            .chain(pre_defined_multi)
            .collect();
        MultiPatterns::with_rules(multi_rules, no_cycle, iter_multi, filter_after, node_multi, n_sec)
    } else {
        let multi_rules: Vec<(&str, bool)> = PRE_DEFINED_MULTI
            .iter()
            .map(|&x| (x, /*symmetric=*/ false))
            .collect();
        MultiPatterns::with_rules(multi_rules, no_cycle, iter_multi, filter_after, node_multi, n_sec)
    };

    // Run saturation
    let time_limit_sec = Duration::new(n_sec, 0);
    let iter_limit = matches
        .value_of("n_iter")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let node_limit = matches
        .value_of("n_nodes")
        .unwrap()
        .parse::<usize>()
        .unwrap();

    let runner = if use_multi {
        // This hook function (which applies the multi-pattern rules) will be called at the
        // beginning of each iteration in equality saturation
        Runner::<Mdl, TensorAnalysis, ()>::default()
            .with_node_limit(node_limit)
            .with_time_limit(time_limit_sec)
            .with_iter_limit(iter_limit)
            .with_expr(&start)
            .with_hook(move |runner| multi_patterns.run_one(runner))
    } else {
        Runner::<Mdl, TensorAnalysis, ()>::default()
            .with_node_limit(node_limit)
            .with_time_limit(time_limit_sec)
            .with_iter_limit(iter_limit)
            .with_expr(&start)
    };

    let start_time = Instant::now();
    let mut runner = runner.run(&rules[..]);
    if do_filter_after {
        // Do cycle removal after the final iteration
        remove_cycle_by_order(&mut runner);
    }
    let sat_duration = start_time.elapsed();
    let num_iter_sat = runner.iterations.len() - 1;

    println!("Runner complete!");
    println!("  Nodes: {}", runner.egraph.total_size());
    println!("  Classes: {}", runner.egraph.number_of_classes());
    println!("  Stopped: {:?}", runner.stop_reason.unwrap());
    println!("  Time taken: {:?}", sat_duration);
    println!("  Number of iterations: {:?}", num_iter_sat);

    let (num_enodes, num_classes, avg_nodes_per_class, num_edges, num_programs) =
        get_stats(&runner.egraph);
    println!("  Average nodes per class: {}", avg_nodes_per_class);
    println!("  Number of edges: {}", num_edges);
    println!("  Number of programs: {}", num_programs);

    // Save egraph
    let (mut egraph, root) = (runner.egraph, runner.roots[0]);
    if save_graph == "all" {
        egraph.dot().to_svg("target/tensat.svg").unwrap();
    }

    if matches.is_present("saturation_only") {
        if let Some(outf) = matches.value_of("out_file") {
            let mut file = OpenOptions::new()
                .append(true)
                .create(true)
                .open(outf)
                .unwrap();

            // Stats to write: original runtime, optimized runtime, saturation time, extraction time,
            // number of nodes, number of eclasses, number of possible programs
            let data = json!({
                "original": 0.0,
                "optimized": 0.0,
                "saturation": sat_duration.as_secs_f32(),
                "extraction": 0.0,
                "nodes": num_enodes,
                "classes": num_classes,
                "programs": num_programs,
                "iter": num_iter_sat,
            });
            let sol_data_str = serde_json::to_string(&data).expect("Fail to convert json to string");

            if let Err(e) = writeln!(file, "{}", sol_data_str) {
                eprintln!("Couldn't write to file: {}", e);
            }
        }
    } else if matches.is_present("query_chain") {
        // DIAGNOSTIC (reachability): is the left-deep CHAIN association of the
        // min-of-max start graph present in the SATURATED e-graph? This separates
        // four verdicts for why the tight (chain) lattice form is never extracted:
        //   chain in memo, not blacklisted        -> cost function's fault
        //   chain in memo, blacklisted             -> cycle filter, not budget/cost
        //   chain absent, Stopped = *Limit         -> saturation budget
        //   chain absent, Stopped = Saturated      -> rule gap (assoc didn't fire)
        // The "Stopped:" reason was already printed above; do NOT grep it away.
        let nodes: &[Mdl] = start.as_ref();
        let n = nodes.len();

        // (1) Map every RecExpr index -> its canonical e-graph Id. All present:
        //     `start` was added via with_expr, and children precede parents.
        let mut id_of: Vec<Id> = vec![Id::from(0usize); n];
        for i in 0..n {
            let mut node = nodes[i].clone();
            node.update_children(|c| id_of[usize::from(c)]);
            id_of[i] = egraph
                .lookup(node)
                .expect("start node absent from egraph (invariant violated)");
        }

        // (2) Descend the outer ewmin tree from the root; its non-ewmin children are
        //     the per-group ewmax-tree roots. Collect each group's ewmax leaves.
        let root_idx = n - 1;
        let mut group_roots: Vec<usize> = Vec::new();
        let mut stack = vec![root_idx];
        while let Some(i) = stack.pop() {
            if matches!(nodes[i], Mdl::Ewmin(_)) {
                for c in nodes[i].children() {
                    stack.push(usize::from(*c));
                }
            } else {
                group_roots.push(i);
            }
        }
        group_roots.sort_unstable();
        fn collect_max_leaves(nodes: &[Mdl], i: usize, out: &mut Vec<usize>) {
            if let Mdl::Ewmax([a, b]) = nodes[i] {
                collect_max_leaves(nodes, usize::from(a), out);
                collect_max_leaves(nodes, usize::from(b), out);
            } else {
                out.push(i);
            }
        }
        let groups: Vec<Vec<usize>> = group_roots
            .iter()
            .map(|&r| {
                let mut v = Vec::new();
                collect_max_leaves(nodes, r, &mut v);
                v
            })
            .collect();
        println!("query_chain: {} group(s)", groups.len());
        for (gi, g) in groups.iter().enumerate() {
            println!("query_chain:   group {} has {} ewmax leaves", gi, g.len());
        }

        // is this (canonical-children) enode on the cycle-filter blacklist?
        let is_blacklisted = |eg: &EGraph<Mdl, TensorAnalysis>, mut node: Mdl| -> bool {
            node.update_children(|id| eg.find(id));
            eg.analysis.blacklist_nodes.contains(&node)
        };

        // (3a) natural-order chain per group: break depth = budget frontier.
        let mut group_chain_id: Vec<Option<Id>> = Vec::new();
        for (gi, g) in groups.iter().enumerate() {
            let mut acc = id_of[g[0]];
            let mut ok = true;
            let mut bl = false;
            for k in 1..g.len() {
                let node = Mdl::Ewmax([acc, id_of[g[k]]]);
                if is_blacklisted(&egraph, node.clone()) {
                    bl = true;
                }
                match egraph.lookup(node) {
                    Some(id) => acc = id,
                    None => {
                        println!(
                            "query_chain: group {} natural-order chain BREAKS at depth {}/{}",
                            gi,
                            k,
                            g.len() - 1
                        );
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                println!(
                    "query_chain: group {} natural-order chain PRESENT (depth {}){}",
                    gi,
                    g.len() - 1,
                    if bl { " [contains BLACKLISTED node]" } else { "" }
                );
                group_chain_id.push(Some(acc));
            } else {
                group_chain_id.push(None);
            }
        }

        // (3b) order-INDEPENDENT subset closure per group: reach[mask] = set of class
        //      ids realizable as SOME left-deep ewmax chain over exactly that leaf
        //      subset. Full mask non-empty => a tight chain over all leaves EXISTS.
        for (gi, g) in groups.iter().enumerate() {
            let m = g.len();
            if m == 0 || m > 16 {
                println!("query_chain: group {} size {} not closed (skip)", gi, m);
                continue;
            }
            let full = (1usize << m) - 1;
            let mut reach: HashMap<usize, HashSet<Id>> = HashMap::new();
            for j in 0..m {
                let mut s = HashSet::new();
                s.insert(egraph.find(id_of[g[j]]));
                reach.insert(1 << j, s);
            }
            let mut masks: Vec<usize> = (1..=full).collect();
            masks.sort_by_key(|mask| mask.count_ones());
            for &mask in &masks {
                if mask.count_ones() < 2 {
                    continue;
                }
                let mut acc: HashSet<Id> = HashSet::new();
                for j in 0..m {
                    if mask & (1 << j) == 0 {
                        continue;
                    }
                    let sub = mask & !(1 << j);
                    if let Some(ids) = reach.get(&sub) {
                        let leaf = egraph.find(id_of[g[j]]);
                        let ids: Vec<Id> = ids.iter().copied().collect();
                        for id in ids {
                            for node in [Mdl::Ewmax([id, leaf]), Mdl::Ewmax([leaf, id])] {
                                if let Some(r) = egraph.lookup(node) {
                                    acc.insert(r);
                                }
                            }
                        }
                    }
                }
                if !acc.is_empty() {
                    reach.insert(mask, acc);
                }
            }
            match reach.get(&full) {
                Some(ids) if !ids.is_empty() => println!(
                    "query_chain: group {} -- SOME left-deep chain over all {} leaves EXISTS ({} class(es))",
                    gi,
                    m,
                    ids.len()
                ),
                _ => println!(
                    "query_chain: group {} -- NO left-deep chain over all {} leaves in e-graph",
                    gi, m
                ),
            }
        }

        // (4) outer min over the natural-order group chains (both operand orders).
        if group_chain_id.len() >= 2 && group_chain_id.iter().all(|x| x.is_some()) {
            let a = group_chain_id[0].unwrap();
            let b = group_chain_id[1].unwrap();
            let mut found = None;
            for node in [Mdl::Ewmin([a, b]), Mdl::Ewmin([b, a])] {
                if let Some(id) = egraph.lookup(node) {
                    found = Some(id);
                    break;
                }
            }
            match found {
                Some(id) => println!(
                    "query_chain: FULL natural-order chain lattice PRESENT, root-equivalent = {}",
                    egraph.find(id) == egraph.find(root)
                ),
                None => println!("query_chain: group chains present but outer MIN node ABSENT"),
            }
        }
        println!("query_chain: DONE (read the 'Stopped:' reason above for the budget verdict)");
    } else if let Some(n_random_str) = matches.value_of("n_random") {
        // Sample N egraph-equivalent graphs at random instead of extracting a
        // single best one -- see RandomCost/UniformRandomCost in optimize.rs
        // for what "random" means under each --random_mode and their caveats.
        let n_random: u32 = n_random_str.parse().expect("--n_random must be an integer");
        let base_seed: u64 = matches
            .value_of("random_seed")
            .unwrap()
            .parse()
            .expect("--random_seed must be an integer");
        let random_mode = matches.value_of("random_mode").unwrap();
        let cost_model = CostModel::with_favor_fusion_strength(
            /*ignore_all_weight_only=*/ matches.is_present("all_weight_only"),
            /*favor_fusion_strength=*/ favor_fusion_strength_from_matches(&matches),
        );
        for i in 0..n_random {
            let seed = base_seed + i as u64;
            let (best_cost, best, duration) = if random_mode == "uniform" {
                let cost_fn = UniformRandomCost::new(&egraph, seed);
                let start_time = Instant::now();
                let mut extractor = Extractor::new(&egraph, cost_fn);
                let (best_cost, best) = extractor.find_best(root);
                (best_cost, best, start_time.elapsed())
            } else {
                let cost_fn = RandomCost::new(&egraph, &cost_model, seed);
                let start_time = Instant::now();
                let mut extractor = Extractor::new(&egraph, cost_fn);
                let (best_cost, best) = extractor.find_best(root);
                (best_cost, best, start_time.elapsed())
            };
            println!(
                "Random sample {} (mode {}, seed {}): cost {:?}, extraction took {:?}",
                i, random_mode, seed, best_cost, duration
            );

            let runner_ext = Runner::<Mdl, TensorAnalysis, ()>::default().with_expr(&best);
            if !no_runtime_report {
                let time_ext = get_full_graph_runtime(&runner_ext, true);
                println!("  Sample {} graph runtime: {}", i, time_ext);
            }
            if let Some(exportf) = matches.value_of("export_model") {
                save_model_with_provenance(&runner_ext, &format!("{}_random{}.model", exportf, i));
            }
        }
    } else if let Some(n_diverse_str) = matches.value_of("n_diverse") {
        // Sample N graphs in sequence, each pushed away from enodes already
        // used by a previous sample -- see DiverseCost in optimize.rs.
        let n_diverse: u32 = n_diverse_str.parse().expect("--n_diverse must be an integer");
        let base_seed: u64 = matches
            .value_of("random_seed")
            .unwrap()
            .parse()
            .expect("--random_seed must be an integer");
        let cost_model = CostModel::with_favor_fusion_strength(
            /*ignore_all_weight_only=*/ matches.is_present("all_weight_only"),
            /*favor_fusion_strength=*/ favor_fusion_strength_from_matches(&matches),
        );
        let mut used: HashSet<Mdl> = HashSet::new();
        for i in 0..n_diverse {
            let seed = base_seed + i as u64;
            let diverse_cost = DiverseCost::new(&egraph, &cost_model, &used, seed);
            let start_time = Instant::now();
            let mut extractor = Extractor::new(&egraph, diverse_cost);
            let (best_cost, best) = extractor.find_best(root);
            let duration = start_time.elapsed();
            let n_used_before = used.len();
            for node in best.as_ref() {
                used.insert(node.clone());
            }
            println!(
                "Diverse sample {} (seed {}): cost {:?}, extraction took {:?}, \
                 {} new enodes added to the used set (total {})",
                i, seed, best_cost, duration, used.len() - n_used_before, used.len()
            );

            let runner_ext = Runner::<Mdl, TensorAnalysis, ()>::default().with_expr(&best);
            if !no_runtime_report {
                let time_ext = get_full_graph_runtime(&runner_ext, true);
                println!("  Sample {} graph runtime: {}", i, time_ext);
            }
            if let Some(exportf) = matches.value_of("export_model") {
                save_model_with_provenance(&runner_ext, &format!("{}_diverse{}.model", exportf, i));
            }
        }
    } else if let Some(n_arch_str) = matches.value_of("n_arch_diverse") {
        // Architecture-diverse sampling: one extraction per reachable
        // multi-pattern rewrite family, by REWARDING that family's
        // rewrite-witness enodes so the fused representative wins its e-class.
        // See ArchDiverseCost in optimize.rs and the design rationale there.
        let n_arch: u32 = n_arch_str.parse().expect("--n_arch_diverse must be an integer");
        let base_seed: u64 = matches
            .value_of("random_seed")
            .unwrap()
            .parse()
            .expect("--random_seed must be an integer");
        let reward: f32 = matches
            .value_of("arch_reward")
            .unwrap()
            .parse()
            .expect("--arch_reward must be a float");
        let penalty: f32 = matches
            .value_of("arch_penalty")
            .unwrap()
            .parse()
            .expect("--arch_penalty must be a float");
        let cost_model = CostModel::with_favor_fusion_strength(
            /*ignore_all_weight_only=*/ matches.is_present("all_weight_only"),
            /*favor_fusion_strength=*/ favor_fusion_strength_from_matches(&matches),
        );
        // Re-canonicalize witness keys to match the enodes the extractor sees.
        canonicalize_rewrite_witness(&mut egraph);
        // Which rule families have witnesses reachable in this egraph, and how
        // many each -- order targets by witness count DESCENDING so the
        // structurally-dominant fusion families (many witnesses) are targeted
        // first rather than cut off by the sample cap.
        let mut rule_counts: std::collections::BTreeMap<usize, usize> =
            std::collections::BTreeMap::new();
        for w in egraph.analysis.rewrite_witness.borrow().values() {
            *rule_counts.entry(w.rule_index).or_insert(0) += 1;
        }
        let mut ranked: Vec<(usize, usize)> = rule_counts.iter().map(|(&r, &c)| (r, c)).collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        println!(
            "Arch-diverse: {} rule families have witnesses (rule:count, by count desc): {:?}",
            ranked.len(),
            ranked
        );
        // Sample 0 = baseline (no target); then one sample per available family,
        // most-populous first.
        let targets: Vec<Option<usize>> = std::iter::once(None)
            .chain(ranked.iter().map(|&(r, _)| Some(r)))
            .take(n_arch as usize)
            .collect();
        let mut covered: HashSet<usize> = HashSet::new();
        for (i, target_rule) in targets.iter().enumerate() {
            let seed = base_seed + i as u64;
            let arch_cost = ArchDiverseCost::new(
                &egraph,
                &cost_model,
                *target_rule,
                &covered,
                seed,
                reward,
                penalty,
            );
            let start_time = Instant::now();
            let mut extractor = Extractor::new(&egraph, arch_cost);
            let (best_cost, best) = extractor.find_best(root);
            let duration = start_time.elapsed();
            // Detect fusion STRUCTURALLY: Concat/Split enodes only ever enter
            // the egraph via a multi-pattern fusion rule, so their presence in
            // the extracted tree is a reliable "this extraction is fused" signal
            // (the witness map is keyed by egraph enodes with canonical child
            // Ids, which don't match the RecExpr's local Ids, so we can't look
            // it up directly). If a targeted sample came out fused, mark that
            // family covered so later samples are pushed off it.
            let n_concat = best
                .as_ref()
                .iter()
                .filter(|n| matches!(n, Mdl::Concat(_)))
                .count();
            let n_split = best
                .as_ref()
                .iter()
                .filter(|n| matches!(n, Mdl::Split(_) | Mdl::Split0(_) | Mdl::Split1(_)))
                .count();
            let is_fused = n_concat > 0 || n_split > 0;
            if is_fused {
                if let Some(r) = target_rule {
                    covered.insert(*r);
                }
            }
            println!(
                "Arch-diverse sample {} (seed {}, target rule {:?}): cost {:?}, took {:?} -- \
                 {} ({} nodes, concat={}, split={})",
                i,
                seed,
                target_rule,
                best_cost,
                duration,
                if is_fused { "FUSED" } else { "unfused" },
                best.as_ref().len(),
                n_concat,
                n_split
            );

            let runner_ext = Runner::<Mdl, TensorAnalysis, ()>::default().with_expr(&best);
            if !no_runtime_report {
                let time_ext = get_full_graph_runtime(&runner_ext, true);
                println!("  Sample {} graph runtime: {}", i, time_ext);
            }
            if let Some(exportf) = matches.value_of("export_model") {
                save_model_with_provenance(&runner_ext, &format!("{}_arch{}.model", exportf, i));
            }
        }
    } else if matches.is_present("verif_cost") {
        // Verifiability-aware extraction: one extraction minimizing the summed
        // ReLU relaxation gap-area (VerifCost in optimize.rs).
        let interval_file = matches
            .value_of("interval_file")
            .expect("--verif_cost requires --interval_file");
        #[derive(serde::Deserialize)]
        struct IvJson {
            lo: Vec<f32>,
            hi: Vec<f32>,
        }
        let raw: HashMap<String, IvJson> =
            serde_json::from_str(&read_to_string(interval_file).expect("read interval_file"))
                .expect("parse interval_file");
        let leaf_intervals: HashMap<Vec<String>, (Vec<f32>, Vec<f32>)> = raw
            .into_iter()
            .map(|(k, v)| {
                let mut names: Vec<String> = k.split(',').map(|s| s.to_string()).collect();
                names.sort();
                (names, (v.lo, v.hi))
            })
            .collect();
        // Optional per-node backward-CROWN sensitivity weights (critical-path).
        let sensitivities: HashMap<Vec<String>, f32> = match matches.value_of("sensitivity_file") {
            Some(sf) => {
                let raw: HashMap<String, f32> =
                    serde_json::from_str(&read_to_string(sf).expect("read sensitivity_file"))
                        .expect("parse sensitivity_file");
                raw.into_iter()
                    .map(|(k, w)| {
                        let mut names: Vec<String> = k.split(',').map(|s| s.to_string()).collect();
                        names.sort();
                        (names, w)
                    })
                    .collect()
            }
            None => HashMap::new(),
        };
        println!("verif-cost: {} sensitivity weights loaded", sensitivities.len());
        let scale: f32 = 1.0e6; // one unstable ReLU dominates the op-count epsilon
        let verif_cost = VerifCost::new(&egraph, leaf_intervals, sensitivities, scale);
        let (hit, total) = verif_cost.leaves_bound();
        println!(
            "verif-cost: {}/{} leaf intervals matched an e-class weight-name set",
            hit, total
        );
        let start_time = Instant::now();
        let mut extractor = Extractor::new(&egraph, verif_cost);
        let (best_cost, best) = extractor.find_best(root);
        println!(
            "verif-cost extraction: gap-cost {:?}, {} nodes, took {:?}",
            best_cost,
            best.as_ref().len(),
            start_time.elapsed()
        );
        let runner_ext = Runner::<Mdl, TensorAnalysis, ()>::default().with_expr(&best);
        if let Some(exportf) = matches.value_of("export_model") {
            save_model_with_provenance(&runner_ext, &format!("{}_verif.model", exportf));
        }
    } else {
        // Run extraction
        let extract_mode = matches.value_of("extract").unwrap();
        let cost_model = CostModel::with_favor_fusion_strength(
            /*ignore_all_weight_only=*/ matches.is_present("all_weight_only"),
            /*favor_fusion_strength=*/ favor_fusion_strength_from_matches(&matches),
        );
        let (best, ext_secs) = match extract_mode {
            "ilp" => extract_by_ilp(&egraph, root, &matches, &cost_model),
            "greedy" => {
                let tnsr_cost = TensorCost {
                    egraph: &egraph,
                    cost_model: &cost_model,
                };
                let start_time = Instant::now();
                let mut extractor = Extractor::new(&egraph, tnsr_cost);
                let (best_cost, best) = extractor.find_best(root);
                let duration = start_time.elapsed();

                println!("Extractor complete!");
                println!("  Time taken: {:?}", duration);
                println!("  Best cost: {:?}", best_cost);
                (best, duration.as_secs_f32())
            }
            _ => panic!("Extracting mode not supported"),
        };

        // Evaluation starting and extracted graph runtime, save graphs
        let runner_start = Runner::<Mdl, TensorAnalysis, ()>::default().with_expr(&start);
        let runner_ext = Runner::<Mdl, TensorAnalysis, ()>::default().with_expr(&best);

        if save_graph != "none" {
            runner_start
                .egraph
                .dot()
                .to_svg("target/start.svg")
                .unwrap();
            runner_ext.egraph.dot().to_svg("target/ext.svg").unwrap();
        }

        let (time_start, time_ext): (Option<f32>, Option<f32>) = if no_runtime_report {
            println!("Skipping full graph runtime evaluation (--no_runtime_report)");
            (None, None)
        } else {
            let time_start = get_full_graph_runtime(&runner_start, false);
            println!("Start graph runtime: {}", time_start);

            let time_ext = get_full_graph_runtime(&runner_ext, true);
            println!("Extracted graph runtime: {}", time_ext);
            (Some(time_start), Some(time_ext))
        };

        if let Some(exportf) = matches.value_of("export_model") {
            save_model(&runner_start, &(exportf.to_owned()+"_start.model"));
        }

        if let Some(exportf) = matches.value_of("export_model") {
            save_model_with_provenance(&runner_ext, &(exportf.to_owned()+"_optimized.model"));
        }

        if let Some(outf) = matches.value_of("out_file") {
            let mut file = OpenOptions::new()
                .append(true)
                .create(true)
                .open(outf)
                .unwrap();

            // Stats to write: original runtime, optimized runtime, saturation time, extraction time,
            // number of nodes, number of eclasses, number of possible programs
            let data = json!({
                "original": time_start,
                "optimized": time_ext,
                "saturation": sat_duration.as_secs_f32(),
                "extraction": ext_secs,
                "nodes": num_enodes,
                "classes": num_classes,
                "programs": num_programs,
                "iter": num_iter_sat,
            });
            let sol_data_str = serde_json::to_string(&data).expect("Fail to convert json to string");

            if let Err(e) = writeln!(file, "{}", sol_data_str) {
                eprintln!("Couldn't write to file: {}", e);
            }
        }
    }
}

/// Extract the optimal graph from EGraph by ILP
///
/// This function prepares the data for the ILP formulation, save it as json, call the python
/// script to read the data + solve ILP + save the solved results. After the python script
/// finishes, it reads back the solved result and construct the RecExpr for the optimized graph.
fn extract_by_ilp(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    root: Id,
    matches: &clap::ArgMatches,
    cost_model: &CostModel,
) -> (RecExpr<Mdl>, f32) {
    // Prepare data for ILP formulation, save to json
    let (m_id_map, e_m, h_i, cost_i, g_i, root_m, i_to_nodes, blacklist_i) =
        prep_ilp_data(egraph, root, cost_model);

    let data = json!({
        "e_m": e_m,
        "h_i": h_i,
        "cost_i": cost_i,
        "g_i": g_i,
        "root_m": root_m,
        "blacklist_i": blacklist_i,
    });
    let data_str = serde_json::to_string(&data).expect("Fail to convert json to string");
    create_dir_all("./tmp");
    write("./tmp/ilp_data.json", data_str).expect("Unable to write file");

    let initialize = matches.is_present("initial_with_greedy");
    if initialize {
        // Get node_to_i map
        let node_to_i: HashMap<Mdl, usize> = (&i_to_nodes)
            .iter()
            .enumerate()
            .map(|(i, node)| (node.clone(), i))
            .collect();

        let tnsr_cost = TensorCost {
            egraph: egraph,
            cost_model: cost_model,
        };
        let mut extractor = Extractor::new(egraph, tnsr_cost);
        let (i_list, m_list) = get_init_solution(egraph, root, &extractor.costs, &g_i, &node_to_i);

        // Store initial solution
        let solution_data = json!({
            "i_list": i_list,
            "m_list": m_list,
        });
        let sol_data_str =
            serde_json::to_string(&solution_data).expect("Fail to convert json to string");
        write("./tmp/init_sol.json", sol_data_str).expect("Unable to write file");
    }

    // Call python script to run ILP
    let order_var_int = matches.is_present("order_var_int");
    let class_constraint = matches.is_present("class_constraint");
    let no_order = matches.is_present("no_order");
    let mut arg_vec = vec!["extractor/extract.py"];
    if order_var_int {
        arg_vec.push("--order_var_int");
    }
    if class_constraint {
        arg_vec.push("--eclass_constraint");
    }
    if no_order {
        arg_vec.push("--no_order");
    }
    if initialize {
        arg_vec.push("--initialize")
    }
    if let Some(time_lim) = matches.value_of("ilp_time_sec") {
        arg_vec.push("--time_lim_sec");
        arg_vec.push(time_lim);
    }
    if let Some(num_thread) = matches.value_of("ilp_num_threads") {
        arg_vec.push("--num_thread");
        arg_vec.push(num_thread);
    }
    let child = Command::new("python")
        .args(&arg_vec)
        .spawn()
        .expect("failed to execute child");
    let output = child.wait_with_output().expect("failed to get output");

    if output.status.success() {
        // Read back solved results, construct optimized graph
        let solved_str = read_to_string("./tmp/solved.json")
            .expect("Something went wrong reading the solved file");
        let solved_data: SolvedResults =
            serde_json::from_str(&solved_str).expect("JSON was not well-formatted");

        let mut node_picked: HashMap<Id, Mdl> = HashMap::new();
        for (i, x_i) in solved_data.solved_x.iter().enumerate() {
            if *x_i == 1 {
                let eclass_id = m_id_map[g_i[i]];
                if node_picked.contains_key(&eclass_id) {
                    println!("Duplicate node in eclass");
                    println!("{}", node_picked.get(&eclass_id).unwrap().display_op());
                    println!("{}", i_to_nodes[i].display_op());
                    continue;
                }
                //assert!(!node_picked.contains_key(&eclass_id));
                node_picked.insert(eclass_id, i_to_nodes[i].clone());
            }
        }

        let mut expr = RecExpr::default();
        let mut added_memo: HashMap<Id, Id> = Default::default();
        let _ = construct_best_rec(&node_picked, root, &mut added_memo, egraph, &mut expr);
        (expr, solved_data.time)
    } else {
        panic!("Python script failed");
    }
}

/// This function gets the following stats:
///     Total number of enodes
///     Total number of eclasses
///     Average number of enodes per class
///     Total number of edges (children relationships)
///     Total number of equivalent programs represented (power of 2)
fn get_stats(egraph: &EGraph<Mdl, TensorAnalysis>) -> (usize, usize, f32, usize, f32) {
    let num_enodes = egraph.total_size();
    let num_classes = egraph.number_of_classes();
    let avg_nodes_per_class = num_enodes as f32 / (num_classes as f32);
    let num_edges = egraph
        .classes()
        .fold(0, |acc, c| c.iter().fold(0, |sum, n| n.len() + sum) + acc);
    let num_programs = egraph
        .classes()
        .fold(0.0, |acc, c| acc + (c.len() as f32).log2());
    (
        num_enodes,
        num_classes,
        avg_nodes_per_class,
        num_edges,
        num_programs,
    )
}

fn get_full_graph_runtime(runner: &Runner<Mdl, TensorAnalysis, ()>, process: bool) -> f32 {
    let mut g = runner.egraph.analysis.graph.borrow_mut();
    unsafe {
        // This is calling TASO's preprocess_weights function before evaluating full graph
        // run time. It removes op that has only weights as its inputs. Since TASO only cares
        // about inference time, such ops can be pre-computed
        if process {
            let processed_g = g.preprocess_weights();
            // (*processed_g).export_to_file_raw(CString::new("/usr/tensat/optimized.onnx").unwrap().into_raw());
            (*processed_g).run()
        } else {
            //(*g).export_to_file_raw(CString::new("/usr/tensat/orig.onnx").unwrap().into_raw());
            (*g).run()
        }
    }
}

fn save_model(runner: &Runner<Mdl, TensorAnalysis, ()>, file_name: &str) {
    let mut g = runner.egraph.analysis.graph.borrow_mut();
    unsafe {
        (*g).export_to_file_raw(CString::new(file_name).unwrap().into_raw());
    }
}

/// Same as `save_model`, but also emits a `<file_name>.weight_names.json`
/// sidecar: guid -> sorted list of contributing weight names, for every
/// Tnsr-typed eclass in `runner`'s egraph with a non-empty `weight_names`
/// set (not just literal Weight leaves -- Enlarge/Concat-of-weights
/// eclasses are covered too). Replaces per-extraction hand-tracing: the
/// guid TASO assigns while replaying `runner`'s RecExpr and the
/// `weight_names` provenance computed for that same eclass are both read
/// off `class.data` in one pass, right after `save_model` assigns those
/// guids. Only meaningful for an *extracted* model's runner (one built via
/// `with_expr` from a RecExpr drawn from a provenance-seeded saturation
/// egraph, e.g. via --weight_names_json) -- on an unseeded egraph every
/// weight_names set is either empty or a synthetic "w_N" name.
fn save_model_with_provenance(runner: &Runner<Mdl, TensorAnalysis, ()>, file_name: &str) {
    save_model(runner, file_name);
    let mut entries: HashMap<String, Vec<String>> = HashMap::new();
    for class in runner.egraph.classes() {
        let d = &class.data;
        if d.dtype == DataKind::Tnsr && !d.weight_names.is_empty() && !d.meta.is_null() {
            let guid = unsafe { (*d.meta).op.guid };
            let mut names: Vec<String> = d.weight_names.iter().cloned().collect();
            names.sort();
            entries.insert(guid.to_string(), names);
        }
    }
    let sidecar_path = format!("{}.weight_names.json", file_name);
    let json = serde_json::to_string_pretty(&entries).expect("failed to serialize weight_names sidecar");
    write(&sidecar_path, json).expect("failed to write weight_names sidecar");
}

/// --mode parse_check: the authoritative oracle for whether a rule is in a form
/// current tensat accepts. Reads a rules file (one `lhs=>rhs` per line) and reports,
/// per line, whether BOTH sides parse as `Pattern<Mdl>` (the exact parser
/// `rules_from_str` uses). Used to settle each op's egg arity/child-order when
/// extending pb2egg, and as the core assertion of the parse-validity regression test.
/// Prints "OK <rule>" / "FAIL <rule>" per line and a final summary; exit code is
/// nonzero iff any line failed.
fn parse_check(matches: clap::ArgMatches) {
    let file = matches.value_of("rules").expect("Pls supply a rules file.");
    let text = read_to_string(file).expect("reading rules file");
    let mut ok = 0usize;
    let mut fail = 0usize;
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        let mut it = l.splitn(2, "=>");
        let lhs = it.next().unwrap().trim();
        let rhs = it.next();
        let lhs_ok = lhs.parse::<Pattern<Mdl>>().is_ok();
        let rhs_ok = rhs
            .map(|r| r.trim().parse::<Pattern<Mdl>>().is_ok())
            .unwrap_or(false);
        if lhs_ok && rhs_ok {
            ok += 1;
            println!("OK   {}", l);
        } else {
            fail += 1;
            println!("FAIL {} (lhs_ok={} rhs_ok={})", l, lhs_ok, rhs_ok);
        }
    }
    println!("parse_check: {} OK, {} FAIL", ok, fail);
    if fail > 0 {
        std::process::exit(1);
    }
}

fn prove_taso_rules(matches: clap::ArgMatches) {
    env_logger::init();

    let file = matches
        .value_of("rules")
        .expect("Pls supply taso rules file.");
    let taso_rules = read_to_string(file).expect("Something went wrong reading the file");

    println!("Parsing rules...");
    let initial = parse_rules(&taso_rules);
    println!("Parsed rules!");

    let mut to_prove = initial.clone();
    while !to_prove.is_empty() {
        let n_before = to_prove.len();
        to_prove = verify(&to_prove);
        let n_proved = n_before - to_prove.len();
        println!("Proved {} on this trip", n_proved);
        if n_proved == 0 {
            println!("\nCouldn't prove {} rule(s)", to_prove.len());
            for pair in &to_prove {
                let i = initial.iter().position(|p| p == pair).unwrap();
                println!("  {}: {} => {}", i, pair.0, pair.1);
            }
            break;
        }
    }
}

/// Ground one side of a rule (an egg PatternAst) into `expr`: each pattern Var
/// becomes a fresh Input leaf of shape [d,d] (shared across sides via `varmap`),
/// each ENode is copied with remapped children. Returns None (=> "keep this rule,
/// don't try to prune it") if the side uses any non-elementwise op, since those
/// need shape assignment we don't attempt here (elementwise ops always typecheck
/// under a uniform square shape). The groundable set is exactly the PWL/AC family.
fn ground_side(
    ast: &egg::RecExpr<egg::ENodeOrVar<Mdl>>,
    expr: &mut egg::RecExpr<Mdl>,
    varmap: &mut HashMap<egg::Var, Id>,
    d: i32,
) -> Option<Id> {
    let nodes = ast.as_ref();
    let mut ids: Vec<Id> = vec![Id::from(0usize); nodes.len()];
    for i in 0..nodes.len() {
        ids[i] = match &nodes[i] {
            egg::ENodeOrVar::Var(sym) => {
                if let Some(id) = varmap.get(sym) {
                    *id
                } else {
                    let nm = format!("rv{}@{}_{}", varmap.len(), d, d);
                    let name_id = expr.add(Mdl::Var(egg::Symbol::from(nm)));
                    let inp = expr.add(Mdl::Input([name_id]));
                    varmap.insert(*sym, inp);
                    inp
                }
            }
            egg::ENodeOrVar::ENode(m) => {
                let groundable = matches!(
                    m,
                    Mdl::Ewadd(_)
                        | Mdl::Ewsub(_)
                        | Mdl::Ewmax(_)
                        | Mdl::Ewmin(_)
                        | Mdl::Ewmul(_)
                        | Mdl::Relu(_)
                );
                if !groundable {
                    return None;
                }
                let mut node = m.clone();
                node.update_children(|c| ids[usize::from(c)]);
                expr.add(node)
            }
        };
    }
    Some(ids[nodes.len() - 1])
}

/// --mode redundancy: greedily prune rules whose LHS=RHS equality is re-derivable
/// from the OTHER (kept) rules within `--redundancy_iters` e-graph iterations, in
/// the SAME sound engine that will use them. Preserves the equational closure;
/// only ever removes. Elementwise/PWL rules are checked; any rule with a
/// non-elementwise op is conservatively kept. Writes kept rules to --out_file.
fn prune_redundant(matches: clap::ArgMatches) {
    let file = matches.value_of("rules").expect("Pls supply a rules file.");
    let text = read_to_string(file).expect("reading rules file");
    let budget: usize = matches
        .value_of("redundancy_iters")
        .unwrap()
        .parse()
        .expect("--redundancy_iters must be an integer");
    let d: i32 = 4; // uniform square shape for grounded vars (elementwise-safe)
    // Per-check saturation caps (memory/time guards). Tunable via --n_nodes/--n_sec so a
    // large rule set (which OOMs at a high node cap) can be pruned with a lower cap.
    let check_nodes: usize = matches.value_of("n_nodes").unwrap().parse().unwrap_or(20000);
    let check_secs: u64 = matches.value_of("n_sec").unwrap().parse().unwrap_or(5);

    let rule_strs: Vec<String> = text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect();
    let n = rule_strs.len();

    // Parse each "lhs=>rhs" into (lhs_ast, rhs_ast); None if it doesn't parse.
    let parsed: Vec<Option<(egg::RecExpr<egg::ENodeOrVar<Mdl>>, egg::RecExpr<egg::ENodeOrVar<Mdl>>)>> =
        rule_strs
            .iter()
            .map(|s| {
                let mut it = s.splitn(2, "=>");
                let l = it.next().unwrap().trim();
                let r = it.next().map(|x| x.trim());
                match (l.parse::<Pattern<Mdl>>(), r.map(|x| x.parse::<Pattern<Mdl>>())) {
                    (Ok(lp), Some(Ok(rp))) => Some((lp.ast, rp.ast)),
                    _ => None,
                }
            })
            .collect();

    // Order candidates largest-LHS-first: prefer removing complex rules, keeping
    // simple generators (assoc/comm survive -- none derives another).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| {
        std::cmp::Reverse(parsed[i].as_ref().map(|(l, _)| l.as_ref().len()).unwrap_or(0))
    });

    let mut kept = vec![true; n];
    let mut n_groundable = 0usize;
    let mut pruned = 0usize;

    for &i in &order {
        if !kept[i] {
            continue;
        }
        let (la, ra) = match &parsed[i] {
            Some(p) => p,
            None => continue, // unparseable -> keep
        };
        // Ground both sides into one RecExpr, joined by a Noop so both are reachable.
        let mut expr = egg::RecExpr::default();
        let mut varmap: HashMap<egg::Var, Id> = HashMap::new();
        let id_l = match ground_side(la, &mut expr, &mut varmap, d) {
            Some(x) => x,
            None => continue, // non-elementwise -> keep
        };
        let id_r = match ground_side(ra, &mut expr, &mut varmap, d) {
            Some(x) => x,
            None => continue,
        };
        let _root = expr.add(Mdl::Noop([id_l, id_r]));
        n_groundable += 1;

        // The other kept rules (no blacklist filtering: we want maximal
        // derivability). Saturate the grounded terms under the iteration budget.
        // (Recompiling the subset per check is cheap vs. the saturation itself.)
        let subset_strs: Vec<&str> = (0..n)
            .filter(|&j| kept[j] && j != i)
            .map(|j| rule_strs[j].as_str())
            .collect();
        let subset = rules_from_str(subset_strs, /*filter_after=*/ false);
        let runner = Runner::<Mdl, TensorAnalysis, ()>::default()
            .with_iter_limit(budget)
            .with_node_limit(check_nodes)
            .with_time_limit(std::time::Duration::from_secs(check_secs))
            .with_expr(&expr)
            .run(&subset[..]);
        let rt = runner.roots[0];
        let (cl, cr) = runner.egraph[rt]
            .nodes
            .iter()
            .find_map(|nd| {
                if let Mdl::Noop([a, b]) = nd {
                    Some((*a, *b))
                } else {
                    None
                }
            })
            .expect("Noop root");
        if runner.egraph.find(cl) == runner.egraph.find(cr) {
            kept[i] = false;
            pruned += 1;
        }
    }

    println!(
        "redundancy: {} rules, {} groundable (elementwise/PWL), budget {} iters",
        n, n_groundable, budget
    );
    println!(
        "redundancy: pruned {} redundant, kept {} ({} non-groundable kept as-is)",
        pruned,
        n - pruned,
        n - n_groundable
    );
    if let Some(outf) = matches.value_of("out_file") {
        let kept_rules: Vec<&str> = (0..n).filter(|&j| kept[j]).map(|j| rule_strs[j].as_str()).collect();
        std::fs::write(outf, kept_rules.join("\n")).expect("writing out_file");
        println!("redundancy: wrote {} kept rules to {}", kept_rules.len(), outf);
    }
}
