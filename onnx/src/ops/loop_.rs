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
    let ParseResult { model: body, unresolved_inputs, .. } = ctx.parse_graph(graph_proto)?;

    let mut inputs = crate::model::optional_inputs(node);
    let trip_count_input = inputs.next().unwrap();
    let condition_input = inputs.next().unwrap();
    ensure!(trip_count_input.is_some() && condition_input.is_some(), "Loop has no exit condition");

    let mut mapped_inputs = vec![];
    let mut mapped_outputs = vec![];

    // Iteration index as body graph input
    mapped_inputs.push(ops::scan::InputMapping::IterIndex);
    // Boolean condition as body graph input and hidden state
    mapped_inputs.push(ops::scan::InputMapping::State {
        initializer: condition_input
            .map(|idx| ops::scan::StateInitializer::FromInput(idx))
            .unwrap_or_else(|| ops::scan::StateInitializer::Value(Arc::new(tensor0(true)))),
    });

    let stop_condition = ops::scan::IterationStopCondition {
        trip_count_from_input: trip_count_input,
        condition_from_state: condition_input.map(|_| 0), // always state index 0 in the state stack
    };

    // Hidden state indexes
    let prev_inputs = trip_count_input.is_some() as usize + condition_input.is_some() as usize;
    for ix in inputs.take_while(|e| e.is_some()).map(Option::unwrap) {
        mapped_inputs.push(ops::scan::InputMapping::State {
            initializer: ops::scan::StateInitializer::FromInput(ix),
        });
        mapped_outputs.push(ops::scan::OutputMapping {
            state: true,
            last_value_slot: Some(ix - prev_inputs),
            scan: None,
            full_dim_hint: None,
        });
    }

    // Scan indexes

    // Subgraph can access anything produced within the upper graph, transform that as additional inputs
    for (ix, _input) in unresolved_inputs.iter().enumerate() {
        mapped_inputs.push(ops::scan::InputMapping::Full {
            slot: model.input_outlets()?.len() - closure_inputs + ix,
        });
    }

    // Ok((Box::new(InferenceLoop { body, body_mapping }), unresolved_inputs))
    Ok((
        Box::new(ops::scan::InferenceScan::new(
            body,
            mapped_inputs,
            mapped_outputs,
            None,
            true,
            GenericFactoid::default(),
        )),
        unresolved_inputs,
    ))
}

#[derive(Clone, new, Debug, Hash)]
struct Loop {}

impl Expansion for Loop {
    fn name(&self) -> Cow<str> {
        "Loop".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        todo!()
    }

    fn wire(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        todo!()
    }
}
