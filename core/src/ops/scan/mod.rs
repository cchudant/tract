use crate::internal::*;
use std::fmt;

mod lir;
mod mir;

pub use lir::LirScan;
pub use mir::Scan;

/// When multiple conditions are provided, it acts as an AND of them.
/// If no condition is provided, it will default to using the scan inputs as trip count.
#[derive(Debug, Clone, new, Default)]
pub struct ExitCondition {
    /// Use a boolean state as a continue/break condition.
    /// Index of the state from the state stack.
    pub condition_from_state: Option<usize>,
    /// Stop the loop at a number of iteration.
    /// The number is taken from an input.
    pub trip_count_from_input: Option<usize>,
}

#[derive(Clone, new, Hash, Eq, PartialEq, Copy)]
pub struct ScanInfo {
    /// Input/output slot index
    pub slot: usize,
    /// Axis on which to scan
    pub axis: usize,
    /// A single scan iteration should has `chunk` elements.
    pub chunk: isize,
}

#[derive(Clone, new, Hash)]
pub enum InputMapping {
    /// Input from outside the body graph, in full.
    Full { slot: usize },
    /// Hidden state carried and modified between iterations.
    State { initializer: StateInitializer },
    /// A Scan input.
    Scan(ScanInfo),
    /// Iteration index
    IterIndex,
}

impl InputMapping {
    pub fn as_state(&self) -> Option<&StateInitializer> {
        match self {
            InputMapping::State { initializer } => Some(initializer),
            _ => None,
        }
    }

    pub fn as_scan(&self) -> Option<&ScanInfo> {
        match self {
            InputMapping::Scan(s) => Some(s),
            _ => None,
        }
    }

    pub fn invisible(&self) -> bool {
        matches!(self, InputMapping::State { initializer: StateInitializer::Value(_) })
    }

    pub fn slot(&self) -> Option<usize> {
        match self {
            InputMapping::Full { slot } => Some(*slot),
            InputMapping::Scan(info) => Some(info.slot),
            InputMapping::State { initializer } => match initializer {
                StateInitializer::FromInput(slot) => Some(*slot),
                _ => None,
            },
        }
    }
}

impl fmt::Debug for InputMapping {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InputMapping::Full { slot } => write!(fmt, "Full, inlet {slot}"),
            InputMapping::State { initializer } => {
                write!(fmt, "State initialized by {initializer:?}")
            }
            InputMapping::Scan(info) => {
                write!(
                    fmt,
                    "Scan inlet {}, axis: {}, chunk: {:?}.",
                    info.slot, info.axis, info.chunk
                )
            }
        }
    }
}

/// This contains the info of what to do with a body graph output after
/// each iteration.
#[derive(Clone, new, Hash, Default)]
pub struct OutputMapping<F: Clone> {
    /// `Some` if this output is a scan output.
    pub scan: Option<ScanInfo>,

    /// `Some` if this output should be mapped to a final output
    /// of the Scan instruction.
    /// The value represents the index of the input to fill.
    pub last_value_slot: Option<usize>,

    /// `true` if this outputs a hidden state variable.
    pub state: bool,

    /// Optional dimension of this output.
    pub full_dim_hint: Option<F>,
}

impl<F: Clone> OutputMapping<F> {
    pub fn invisible(&self) -> bool {
        self.scan.is_none() && self.last_value_slot.is_none()
    }
}

impl<F: Clone + DimLike> OutputMapping<F> {
    pub fn concretize_dims(&self, values: &SymbolValues) -> TractResult<OutputMapping<F>> {
        Ok(Self {
            full_dim_hint: self.full_dim_hint.as_ref().map(|h| h.eval(values)),
            ..self.clone()
        })
    }
}

impl<F: Clone + fmt::Display> fmt::Debug for OutputMapping<F> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        if self.state {
            write!(fmt, "State. ")?;
        }
        if let Some(last_value_slot) = self.last_value_slot {
            write!(fmt, "Last value to outlet {last_value_slot}. ")?;
        }
        if let Some(info) = self.scan {
            write!(fmt, "Full value to outlet {} (axis: {}). ", info.slot, info.axis)?;
        }
        if let Some(full_dim_hint) = &self.full_dim_hint {
            write!(fmt, "Full len {full_dim_hint}. ")?;
        }
        Ok(())
    }
}

#[derive(Clone, new, Hash)]
pub enum StateInitializer {
    FromInput(usize),
    Value(Arc<Tensor>),
}

impl fmt::Debug for StateInitializer {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use StateInitializer::*;
        match self {
            FromInput(i) => write!(fmt, "inlet {i}"),
            Value(t) => write!(fmt, "tensor {t:?}"),
        }
    }
}
