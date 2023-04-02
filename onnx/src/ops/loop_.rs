use crate::model::ParseResult;
use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_hir::internal::*;
use tract_hir::ops;

pub fn loop_(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let graph_proto = node.get_attr("body")?;
    let ParseResult { model: mut body, unresolved_inputs, .. } = ctx.parse_graph(graph_proto)?;

    let mut inputs = crate::model::optional_inputs(node);
    let trip_count_input = inputs.next().unwrap();
    let condition_input = inputs.next().unwrap();
    ensure!(trip_count_input.is_some() && condition_input.is_some(), "Loop has no exit condition");

    let mut mapped_inputs = vec![];
    let mut mapped_outputs = vec![];
    let mut outer_graph_output_idx = 0;
    let mut outer_graph_input_idx = 0;

    // Iteration index as body graph input
    mapped_inputs.push(ops::scan::InputMapping::IterIndex);
    // Boolean condition as body graph input and hidden state
    let condition_initializer = match condition_input {
        Some(idx) => {
            outer_graph_input_idx += 1;
            ops::scan::StateInitializer::FromInput(idx)
        }
        None => ops::scan::StateInitializer::Value(Arc::new(tensor0(true))),
    };
    mapped_inputs.push(ops::scan::InputMapping::State { initializer: condition_initializer });

    // Condition state var
    mapped_outputs.push(ops::scan::OutputMapping { state: true, ..Default::default() });

    let exit_condition = ops::scan::ExitCondition {
        trip_count_from_input: trip_count_input,
        condition_from_state: condition_input.map(|_| 0), // always state index 0 in the state stack
    };
    if condition_input.is_some() {
        outer_graph_input_idx += 1;
    }

    // Hidden state indexes
    for ix in inputs.take_while(|e| e.is_some()).map(|e| e.unwrap()) {
        trace!("hii {ix}");
        mapped_inputs.push(ops::scan::InputMapping::State {
            initializer: ops::scan::StateInitializer::FromInput(ix),
        });
        outer_graph_input_idx += 1;
        mapped_outputs.push(ops::scan::OutputMapping {
            state: true,
            last_value_slot: Some(outer_graph_output_idx),
            ..Default::default()
        });
        outer_graph_output_idx += 1;
    }

    // Scan indexes (remeaning graph outputs)
    let graph_outputs = body.output_outlets()?.len();
    trace!("Hello");
    for _ in mapped_outputs.len()..graph_outputs {
        mapped_outputs.push(ops::scan::OutputMapping {
            scan: Some(ops::scan::ScanInfo { slot: outer_graph_output_idx, axis: 0, chunk: 1 }),
            ..Default::default()
        });
        outer_graph_output_idx += 1;
    }

    // Subgraph can access anything produced within the upper graph, transform that as additional inputs
    for _ in unresolved_inputs.iter() {
        mapped_inputs.push(ops::scan::InputMapping::Full { slot: outer_graph_input_idx });
        outer_graph_input_idx += 1;
    }

    ensure!(
        mapped_inputs.len() == body.inputs.len(),
        "Loop has unmatched number of inputs and subgraph inputs"
    );
    ensure!(
        mapped_outputs.len() == body.outputs.len(),
        "Loop has unmatched number of outputs and subgraph outputs"
    );

    // Modify the graph a bit so that it uses I64 for iteration counts instead of TDim

    let outlet = body.input_outlets()?[0];
    InferenceModelPatch::intercept(
        &body,
        body.input_outlets()?[0],
        format!("{}.input-0.cast", node.name),
        ops::cast::cast(DatumType::I64),
        body.outlet_fact(outlet)?.clone(),
    )?
    .apply(&mut body)?;

    Ok((
        Box::new(ops::scan::InferenceScan::new(
            body,
            mapped_inputs,
            mapped_outputs,
            // None,
            false,
            GenericFactoid::default(),
            exit_condition,
        )),
        unresolved_inputs,
    ))
}
