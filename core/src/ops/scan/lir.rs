use crate::ops::OpStateFreeze;

use super::*;
use tract_data::internal::*;

#[derive(Debug, Clone, new)]
pub struct LirScanOpParams {
    /// Skip the first n iterations
    pub skip: usize,
    pub plan: Arc<TypedSimplePlan<TypedModel>>,
    pub input_mapping: Vec<InputMapping>,
    pub output_mapping: Vec<OutputMapping<TDim>>,
    pub exit_condition: ExitCondition,
}

#[derive(Debug, Clone, new)]
pub struct LirScan(Arc<LirScanOpParams>);

impl std::ops::Deref for LirScan {
    type Target = LirScanOpParams;
    fn deref(&self) -> &LirScanOpParams {
        &self.0
    }
}

impl LirScan {
    pub fn iteration_count(&self, inputs: &[&TypedFact]) -> TractResult<Option<TDim>> {
        // From trip count
        if let Some(idx) = self.exit_condition.trip_count_from_input.and_then(|e| inputs[e].konst) {
            return Ok(Some(
                idx.to_scalar::<TDim>().context("Trip count is not a scalar")?.clone(),
            ));
        }

        // From scan input
        Ok(self
            .input_mapping
            .iter()
            .find_map(|it| match it {
                InputMapping::Scan(info) => Some(info),
                _ => None,
            })
            .map(|info| {
                let outside_dim = inputs[info.slot].shape[info.axis].clone();
                outside_dim / info.chunk
            }))
    }
}

impl Op for LirScan {
    fn name(&self) -> Cow<str> {
        "Scan".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut lines = vec![];
        for (ix, im) in self.input_mapping.iter().enumerate() {
            lines.push(format!("Model input  #{ix}: {im:?}"));
        }
        for (ix, om) in self.output_mapping.iter().enumerate() {
            lines.push(format!("Model output #{ix}: {om:?}"));
        }
        Ok(lines)
    }

    op_as_typed_op!();
}

impl EvalOp for LirScan {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(State {
            position: 0,
            hidden_state: tvec!(),
            model_state: TypedSimpleState::new(Arc::clone(&self.plan))?,
            op: Arc::clone(&self.0),
        })))
    }
}

#[derive(Clone, Debug)]
struct State {
    op: Arc<LirScanOpParams>,
    position: usize,
    hidden_state: TVec<TValue>,
    model_state: TypedSimpleState<TypedModel, Arc<TypedSimplePlan<TypedModel>>>,
}

#[derive(Debug, Clone)]
struct FrozenState {
    op: Arc<LirScanOpParams>,
    position: usize,
    hidden_state: TVec<Tensor>,
    model_state: TypedFrozenSimpleState<TypedModel, Arc<TypedSimplePlan<TypedModel>>>,
}

impl OpStateFreeze for State {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenState {
            op: self.op.clone(),
            position: self.position,
            hidden_state: self.hidden_state.iter().map(|t| t.clone().into_tensor()).collect(),
            model_state: self.model_state.freeze(),
        })
    }
}

impl FrozenOpState for FrozenState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(State {
            op: self.op.clone(),
            position: self.position,
            hidden_state: self.hidden_state.iter().map(|t| t.clone().into_tvalue()).collect(),
            model_state: self.model_state.unfreeze(),
        })
    }
}

impl State {
    pub(super) fn slice_input(
        input: &Tensor,
        axis: usize,
        chunk_ix: usize,
        chunk_dim: isize,
    ) -> TractResult<Tensor> {
        unsafe {
            let full_len = input.shape()[axis];
            let mut shape: TVec<usize> = input.shape().into();
            shape[axis] = chunk_dim.unsigned_abs();
            let mut t = Tensor::uninitialized_dt(input.datum_type(), &shape)?;
            if chunk_dim < 0 {
                let chunk_dim = (-chunk_dim) as usize;
                for i in 0..chunk_dim {
                    if chunk_dim * chunk_ix + i < full_len {
                        let dst_ix = chunk_dim - i - 1;
                        let src_ix = full_len - 1 - (chunk_ix * chunk_dim + i);
                        t.assign_slice_unchecked(dst_ix..=dst_ix, input, src_ix..=src_ix, axis);
                    }
                }
            } else if (chunk_ix + 1) * chunk_dim as usize > full_len {
                let chunk_dim = chunk_dim as usize;
                let remain = full_len - chunk_ix * chunk_dim;
                let mut shape: TVec<usize> = input.shape().into();
                shape[axis] = chunk_dim;
                t.assign_slice_unchecked(..remain, input, chunk_ix * chunk_dim.., axis);
            } else {
                let start = chunk_dim as usize * chunk_ix;
                let end = start + chunk_dim as usize;
                t.assign_slice_unchecked(.., input, start..end, axis);
            }
            Ok(t)
        }
    }

    pub(super) fn assign_output(
        output: &mut Tensor,
        axis: usize,
        element_value: &Tensor,
        i: usize,
        backward: bool,
    ) {
        let full_len = output.shape()[axis];
        let offset = if backward {
            full_len - 1 - i * element_value.shape()[axis]
        } else {
            i * element_value.shape()[axis]
        };
        let count = element_value.shape()[axis].min(output.shape()[axis] - offset);
        unsafe {
            output.assign_slice_unchecked(offset..offset + count, element_value, ..count, axis)
        };
    }
}

impl OpState for State {
    fn eval(
        &mut self,
        session: &mut SessionState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let State { op, ref mut hidden_state, ref mut position, ref mut model_state } = self;
        // initialize state at first pass
        if hidden_state.len() == 0 {
            for input in &op.input_mapping {
                if let InputMapping::State { initializer } = input {
                    hidden_state.push(match initializer {
                        StateInitializer::FromInput(slot) => inputs[*slot].clone(),
                        StateInitializer::Value(v) => (**v).to_owned().into_tvalue(),
                    });
                }
            }
        }

        let iters = {
            op.input_mapping
                .iter()
                .find_map(|it| match it {
                    InputMapping::Scan(info) => Some(info),
                    _ => None,
                })
                .map(|info| inputs[info.slot].shape()[info.axis].divceil(info.chunk.unsigned_abs()))
        };

        if let Some(idx) = self.op.exit_condition.trip_count_from_input.map(|e| inputs[e]) {
            let trip_count = idx
                .to_scalar::<TDim>()
                .context("Trip count is not a scalar TDim")?
                .clone()
                .as_i64()
                .context("Trip count is not a scalar TDim")?;
            iters = Some(
                trip_count
                    .try_into()
                    .with_context(|| format!("{} is not a valid trip count", trip_count))?,
            );
        }

        // When we also have a condition, the final trip count is not really known before execution.
        let trip_count_known_before =
            self.op.exit_condition.condition_from_state.is_none() && iters.is_some();

        let mut outputs = tvec!();
        for (ix, output) in op.output_mapping.iter().enumerate() {
            if let Some(info) = output.scan {
                let fact = op.plan.model().output_fact(ix)?;
                let mut shape: TVec<usize> =
                    fact.shape.eval_to_usize(&session.resolved_symbols)?.into_owned();
                let scanning_dim = output
                    .full_dim_hint
                    .as_ref()
                    .and_then(|d| d.to_usize().ok())
                    .unwrap_or(shape[info.axis] * iters);
                shape[info.axis] = scanning_dim;
                let t = unsafe { Tensor::uninitialized_dt(fact.datum_type, &shape)? };
                outputs.push((info.slot, t));
            }
            if let Some(slot) = output.last_value_slot {
                outputs.push((slot, Tensor::default()));
            }
        }
        outputs.sort_by_key(|a| a.0);
        let mut outputs: TVec<Tensor> = outputs.into_iter().map(|(_slot, v)| v).collect();

        for i in 0.. {
            *position += 1;
            if *position <= op.skip {
                continue;
            }

            // exit conditions

            // iter condition
            if iters.map(|e| e == i) == Some(true) {
                trace!("break out of loop for reaching iteration count");
                break;
            }
            // boolean hidden state condition
            if let Some(idx) = self.op.exit_condition.condition_from_state {
                let scalar: &bool = hidden_state[idx]
                    .to_scalar()
                    .context("Loop exit condition is not a scalar boolean")?;
                if *scalar {
                    trace!("break out of loop from boolean condition");
                    break;
                }
            }

            hidden_state.reverse();

            // prepare inputs

            let iter_inputs: TVec<TValue> = op
                .input_mapping
                .iter()
                .map(|m| {
                    Ok(match m {
                        InputMapping::State { .. } => Some(hidden_state.pop().unwrap()),
                        InputMapping::Scan(info) => Some(
                            Self::slice_input(&inputs[info.slot], info.axis, i, info.chunk)?
                                .into_tvalue(),
                        ),
                        InputMapping::Full { slot } => Some(inputs[*slot].clone()),
                        InputMapping::IterIndex => Some(tensor0(i.to_dim()).into()),
                    })
                })
                .collect::<TractResult<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            // run the loop body

            trace!("iter_inputs #{}: {:?}", i, iter_inputs);
            let iter_outputs =
                model_state.run(iter_inputs).with_context(|| "Evaluating inner body")?;
            trace!("iter_outputs #{}: {:?}", i, iter_outputs);

            // handle the outputs

            for (v, mapping) in iter_outputs.into_iter().zip(&op.output_mapping) {
                if let Some(info) = mapping.scan {
                    Self::assign_output(&mut outputs[info.slot], info.axis, &v, i, info.chunk < 0);
                }

                if i == iters - 1 {
                    if let Some(slot) = mapping.last_value_slot {
                        outputs[slot] = v.clone().into_tensor();
                    }
                }
                if mapping.state {
                    hidden_state.push(v);
                }
            }
        }

        Ok(outputs.into_iter().map(|t| t.into_tvalue()).collect())
    }
}

impl TypedOp for LirScan {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut outputs = tvec!();
        let iters = {
            let info = self.input_mapping.iter().find_map(|it| it.as_scan()).unwrap();
            inputs[info.slot].shape[info.axis].clone().div_ceil(info.chunk.unsigned_abs() as _)
        };
        for (ix, output) in self.output_mapping.iter().enumerate() {
            let fact = self.plan.model().output_fact(ix)?;
            if let Some(slot) = output.last_value_slot {
                outputs.push((slot, fact.datum_type.fact(fact.shape.clone())));
            }
            if let Some(info) = output.scan {
                let mut shape = fact.shape.clone();
                let scanning_dim =
                    output.full_dim_hint.clone().unwrap_or(shape[info.axis].clone() * &iters);
                shape.set(info.axis, scanning_dim);
                outputs.push((info.slot, fact.datum_type.fact(shape)));
            }
        }
        outputs.sort_by_key(|a| a.0);
        let outputs: TVec<_> = outputs.into_iter().map(|(_slot, v)| v).collect();
        Ok(outputs)
    }
}
