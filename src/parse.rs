use crate::input::*;
use crate::model::*;
use egg::*;
use pest::{iterators::Pair, Parser};
use root::taso::*;
use std::collections::HashMap;

#[derive(pest_derive::Parser)]
#[grammar = "equation.pest"]
pub struct EqParser;

pub fn parse_exp(e: Pair<Rule>) -> String {
    match e.as_rule() {
        Rule::name => e.as_str().to_owned(),
        Rule::expr => parse_exp(e.into_inner().next().unwrap()),
        Rule::apply => {
            let mut inner_rules = e.into_inner();
            let op = parse_exp(inner_rules.next().unwrap());
            let args = parse_exp(inner_rules.next().unwrap());
            format!("({} {})", op, args)
        }
        Rule::args => {
            let arg_ss: Vec<_> = e.into_inner().map(parse_exp).collect();
            arg_ss.join(" ")
        }
        _ => unreachable!(),
    }
}

pub fn parse_eq(e: Pair<Rule>) -> (RecExpr<Mdl>, RecExpr<Mdl>) {
    match e.as_rule() {
        Rule::eq => {
            let mut inner_rules = e.into_inner();
            let lhs = parse_exp(inner_rules.next().unwrap());
            let rhs = parse_exp(inner_rules.next().unwrap());
            (lhs.parse().unwrap(), rhs.parse().unwrap())
        }
        _ => unreachable!(),
    }
}

pub fn parse_rules(rs_s: &str) -> Vec<(RecExpr<Mdl>, RecExpr<Mdl>)> {
    let rs = EqParser::parse(Rule::prog, rs_s)
        .expect("parse error")
        .next()
        .unwrap();
    match rs.as_rule() {
        Rule::prog => rs.into_inner().map(parse_eq).collect(),
        _ => unreachable!(),
    }
}

pub fn convert_eq(e: Pair<Rule>) -> String {
    match e.as_rule() {
        Rule::eq => {
            let mut inner_rules = e.into_inner();
            let lhs = parse_exp(inner_rules.next().unwrap());
            let rhs = parse_exp(inner_rules.next().unwrap());
            let eq_str = format!("{}=>{}", lhs, rhs);
            str::replace(&eq_str, "input", "?input")
        }
        _ => unreachable!(),
    }
}

pub fn parse_and_convert(rs_s: &str) -> String {
    let rs = EqParser::parse(Rule::prog, rs_s)
        .expect("parse error")
        .next()
        .unwrap();
    match rs.as_rule() {
        Rule::prog => {
            let converted_rules: Vec<String> = rs.into_inner().map(convert_eq).collect();
            let joined = converted_rules.join("\n");
            joined
        }
        _ => unreachable!(),
    }
}

// parses a serialized model from taso
// see tests/parse.rs for an example
pub fn parse_model(rs_s: &str) -> GraphConverter {
    parse_model_impl(rs_s, None)
}

/// Same as `parse_model`, but seeds real weight identities (instead of the
/// synthetic "w_N" counter) via a `guid -> real_name` sidecar map, keyed by
/// each Weight op's guid as it appears in `rs_s` -- so `TensorAnalysis`'s
/// `weight_names` provenance (see model.rs) has real names to propagate
/// from this parse onward, through every later rewrite/extraction of the
/// resulting egraph. Intended for parsing a model's *baseline* (first,
/// pre-saturation) `.model` file specifically -- extractions/samples drawn
/// from the same saturation egraph never need their own sidecar, since
/// provenance is computed once per eclass and carried forward automatically.
pub fn parse_model_with_names(rs_s: &str, guid_names: &HashMap<usize, String>) -> GraphConverter {
    parse_model_impl(rs_s, Some(guid_names))
}

fn parse_model_impl(rs_s: &str, guid_names: Option<&HashMap<usize, String>>) -> GraphConverter {
    let mut ls = rs_s.lines();
    let mut g = GraphConverter::default();
    let mut nodes: HashMap<usize, Vec<TensorInfo>> = HashMap::new();
    loop {
        if let Some(l) = ls.next() {
            // node id
            let guid = l.parse::<usize>().unwrap();
            // the operator
            let op = ls.next().unwrap().parse::<u32>().unwrap();
            // children; each child has an id and an index;
            // the index is almost always 0, except when the child
            // is a split it may be 0 or 1 (indicating left or right)
            let deps: Vec<Vec<usize>> = ls
                .next()
                .unwrap()
                .split(",")
                .map(|c_s| c_s.split(":").map(|c| c.parse().unwrap()).collect())
                .collect();
            // parameters
            let params: Vec<i32> = ls
                .next()
                .unwrap()
                .split(",")
                .map(|p_s| p_s.parse().unwrap())
                .collect();
            // node is really a vec, because split may return two outputs
            let node: Vec<TensorInfo> = match op {
                OpType_OP_INPUT => vec![g.new_input(&params)],
                OpType_OP_WEIGHT => {
                    let real_name = guid_names.and_then(|m| m.get(&guid)).map(String::as_str);
                    vec![g.new_weight_named(&params, real_name)]
                },
                OpType_OP_MATMUL => vec![g.matmul(
                    nodes[&deps[0][0]][deps[0][1]],
                    nodes[&deps[1][0]][deps[1][1]],
                )],
                OpType_OP_EW_ADD => {
vec![g.add(
                    nodes[&deps[0][0]][deps[0][1]],
                    nodes[&deps[1][0]][deps[1][1]],
                )]},
                OpType_OP_EW_SUB => vec![g.sub(
                    nodes[&deps[0][0]][deps[0][1]],
                    nodes[&deps[1][0]][deps[1][1]],
                )],
                OpType_OP_EW_MAX => vec![g.max(
                    nodes[&deps[0][0]][deps[0][1]],
                    nodes[&deps[1][0]][deps[1][1]],
                )],
                OpType_OP_EW_MIN => vec![g.min(
                    nodes[&deps[0][0]][deps[0][1]],
                    nodes[&deps[1][0]][deps[1][1]],
                )],
                OpType_OP_RELU => vec![g.relu(nodes[&deps[0][0]][deps[0][1]])],
                OpType_OP_DROPOUT => vec![g.dropout(nodes[&deps[0][0]][deps[0][1]])],
                OpType_OP_RESHAPE => vec![g.reshape(nodes[&deps[0][0]][deps[0][1]], &params)],
                OpType_OP_TRANSPOSE => {
                    vec![g.transpose(nodes[&deps[0][0]][deps[0][1]], &params[1..1+(params[0] as usize)], params[params[0] as usize +1] != 0)]
                },
                OpType_OP_CONV2D => vec![g.conv2d(nodes[&deps[0][0]][deps[0][1]], nodes[&deps[1][0]][deps[1][1]], params[8], params[9], params[10], params[11])],
                OpType_OP_POOL2D_AVG => vec![g.avgpool2d(nodes[&deps[0][0]][deps[0][1]], params[5], params[6], params[7], params[8], params[9])],
                OpType_OP_POOL2D_MAX => vec![g.maxpool2d(nodes[&deps[0][0]][deps[0][1]], params[5], params[6], params[7], params[8], params[9])],
                OpType_OP_CONCAT => {
                    let inputs: Vec<TensorInfo> = deps.iter().map(|child| nodes[&child[0]][child[1]]).collect();
                    vec![g.concat_multi(params[0], &inputs)]
                },
                OpType_OP_SPLIT => todo!(), // reference 'Split' case in taso/examples/load_model.py
                OpType_OP_BATCHNORM => {
 vec![g.batchnorm(nodes[&deps[0][0]][deps[0][1]], nodes[&deps[1][0]][deps[1][1]], nodes[&deps[2][0]][deps[2][1]], nodes[&deps[3][0]][deps[3][1]], nodes[&deps[4][0]][deps[4][1]])]
                },
                o => panic!("{} not yet implemented", o),
            };
            nodes.insert(guid, node);
        } else {
            break;
        }
    }
    g
}
