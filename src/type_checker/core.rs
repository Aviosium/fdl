use std::collections::HashMap;
use std::error;
use std::fmt;
use std::ops::Range;

use super::reachability;
use crate::parser::spans::{Span, SpannedError as TypeError};

type ID = usize;

#[derive(Copy, Clone, Debug)]
pub struct Value(ID);
#[derive(Copy, Clone, Debug)]
pub struct Use(ID);

pub type LazyFlow = (Value, Use);

#[derive(Clone, Debug)]
pub struct Bound {
    bounds: Vec<(i128, i128)>,
}

impl Default for Bound {
    fn default() -> Self {
        Bound { bounds: Vec::new() }
    }
}

impl Bound {
    pub fn from_number(number: i128) -> Bound {
        Bound {
            bounds: vec![(number, number)]
        }
    }

    pub fn from_range(range: Range<i128>) -> Bound {
        Bound {
            bounds: vec![(*range.start(), *range.end())]
        }
    }

    pub fn merge(mut self) -> Bound {
        let mut ranges = self.bounds;
        ranges.sort_by_key(|(_, i)| i);
        let mut result = Vec::new();
        if let Some((mut prev_start, mut prev_end)) = ranges.pop() {
            while let Some((start, end)) = ranges.pop() {
                if end >= prev_start {
                    prev_start = prev_start.min(start)
                } else {
                    result.push((prev_start, prev_end));
                    prev_start = start;
                    prev_end = end;
                }
            }
            result.push((prev_start, prev_end));
        }
        self.bounds = result;
        self
    }

    pub fn add(&self, other: &Bound) -> Bound {
        let mut new = self.clone();
        new.bounds = new
            .bounds
            .into_iter()
            .flat_map(|(start, end)| {
                other
                    .bounds
                    .iter()
                    .map(|&(other_start, other_end)| (start + other_start, end + other_end))
                    .collect()
            })
            .collect();
        new.merge()
    }

    pub fn sub(&self, other: &Bound) -> Bound {
        let mut new_other = other.clone();
        new_other.bounds = new_other
            .bounds
            .into_iter()
            .map(|(start, end)| (-end, -start))
            .collect();
        self.add(&new_other)
    }

    pub fn contains(&self, other: &Bound) -> bool {
        let mut own_ranges = self.bounds.clone();
        let mut other_ranges = other.bounds.clone();
        own_ranges.sort_unstable_by_key(|(i, _)| i);
        other_ranges.sort_unstable_by_key(|(i, _)| i);

        let mut i = 0;
        let mut j = 0;

        while i < other_ranges.len() && j < own_ranges.len() {
            let (other_start, other_end) = other_ranges[i];
            let (own_start, own_end) = own_ranges[j];

            if other_start >= own_start && other_end <= other_end {
                i += 1;
            } else {
                j += 1;
            }
        }

        i == other_ranges.len()
    }
}

#[derive(Debug, Clone)]
enum VTypeHead {
    VBool,
    VFloat,
    VUnboundedInt,
    VBoundedInt(Bound),
    VNull,
    VStr,
    VIntOrFloat,
    VFunc {
        arg: Use,
        ret: Value,
    },
    VObj {
        fields: HashMap<String, Value>,
        proto: Option<Value>,
    },
    VCase {
        case: (String, Value),
    },
    VRef {
        write: Option<Use>,
        read: Option<Value>,
    },
}
#[derive(Debug, Clone)]
enum UTypeHead {
    UBool,
    UFloat,
    UUnboundedInt,
    UBoundedInt(Bound),
    UNull,
    UStr,
    UIntOrFloat,
    UFunc {
        arg: Value,
        ret: Use,
    },
    UObj {
        field: (String, Use),
    },
    UCase {
        cases: HashMap<String, (Use, LazyFlow)>,
        wildcard: Option<(Use, LazyFlow)>,
    },
    URef {
        write: Option<Value>,
        read: Option<Use>,
    },
    UNullCase {
        nonnull: Use,
    },
}

fn check_heads(
    lhs_ind: ID,
    lhs: &(VTypeHead, Span),
    rhs_ind: ID,
    rhs: &(UTypeHead, Span),
    out: &mut Vec<(Value, Use)>,
) -> Result<(), TypeError> {
    use UTypeHead::*;
    use VTypeHead::*;

    match (&lhs.0, &rhs.0) {
        (&VBool, &UBool) => Ok(()),
        (&VFloat, &UFloat) => Ok(()),
        (&VUnboundedInt, &UUnboundedInt) => Ok(()),
        (&VNull, &UNull) => Ok(()),
        (&VStr, &UStr) => Ok(()),
        (&VUnboundedInt, &UIntOrFloat) => Ok(()),
        (&VFloat, &UIntOrFloat) => Ok(()),
        (&VIntOrFloat, &UIntOrFloat) => Ok(()),

        (
            &VFunc {
                arg: arg1,
                ret: ret1,
            },
            &UFunc {
                arg: arg2,
                ret: ret2,
            },
        ) => {
            out.push((ret1, ret2));
            // flip the order since arguments are contravariant
            out.push((arg2, arg1));
            Ok(())
        }
        (
            &VObj {
                fields: ref fields1,
                proto,
            },
            &UObj {
                field: (ref name, rhs2),
            },
        ) => {
            // Check if the accessed field is defined
            if let Some(&lhs2) = fields1.get(name) {
                out.push((lhs2, rhs2));
                Ok(())
            } else if let Some(lhs2) = proto {
                out.push((lhs2, Use(rhs_ind)));
                Ok(())
            } else {
                Err(TypeError::new2(
                    format!(
                        "TypeError: Missing field {}\nNote: Field is accessed here",
                        name
                    ),
                    rhs.1,
                    "But the record is defined without that field here.",
                    lhs.1,
                ))
            }
        }
        (
            &VCase {
                case: (ref name, lhs2),
            },
            &UCase {
                cases: ref cases2,
                wildcard,
            },
        ) => {
            // Check if the right case is handled
            if let Some((rhs2, lazy_flow)) = cases2.get(name).copied() {
                out.push((lhs2, rhs2));
                out.push(lazy_flow);
                Ok(())
            } else if let Some((rhs2, lazy_flow)) = wildcard {
                out.push((Value(lhs_ind), rhs2));
                out.push(lazy_flow);
                Ok(())
            } else {
                Err(TypeError::new2(
                    format!(
                        "TypeError: Unhandled case {}\nNote: Case originates here",
                        name
                    ),
                    lhs.1,
                    "But it is not handled here.",
                    rhs.1,
                ))
            }
        }

        (
            &VRef {
                read: r1,
                write: w1,
            },
            &URef {
                read: r2,
                write: w2,
            },
        ) => {
            if let Some(r2) = r2 {
                if let Some(r1) = r1 {
                    out.push((r1, r2));
                } else {
                    return Err(TypeError::new2(
                        "TypeError: Reference is not readable.\nNote: Ref is made write-only here",
                        lhs.1,
                        "But is read here.",
                        rhs.1,
                    ));
                }
            }
            if let Some(w2) = w2 {
                if let Some(w1) = w1 {
                    // flip the order since writes are contravariant
                    out.push((w2, w1));
                } else {
                    return Err(TypeError::new2(
                        "TypeError: Reference is not writable.\nNote: Ref is made read-only here",
                        lhs.1,
                        "But is written here.",
                        rhs.1,
                    ));
                }
            }
            Ok(())
        }

        (&VNull, &UNullCase { .. }) => Ok(()),
        (_, &UNullCase { nonnull }) => {
            out.push((Value(lhs_ind), nonnull));
            Ok(())
        }

        _ => {
            let found = match lhs.0 {
                VBool => "boolean",
                VFloat => "float",
                VUnboundedInt => "integer",
                VNull => "null",
                VStr => "string",
                VIntOrFloat => "float or integer",
                VFunc { .. } => "function",
                VObj { .. } => "record",
                VCase { .. } => "case",
                VRef { .. } => "ref",
            };
            let expected = match rhs.0 {
                UBool => "boolean",
                UFloat => "float",
                UUnboundedInt => "integer",
                UNull => "null",
                UStr => "string",
                UIntOrFloat => "float or integer",
                UFunc { .. } => "function",
                UObj { .. } => "record",
                UCase { .. } => "case",
                URef { .. } => "ref",
                UNullCase { .. } => unreachable!(),
            };

            Err(TypeError::new2(
                format!("TypeError: Value is required to be a {} here,", expected),
                rhs.1,
                format!("But that value may be a {} originating here.", found),
                lhs.1,
            ))
        }
    }
}

#[derive(Debug, Clone)]
enum TypeNode {
    Var,
    Value((VTypeHead, Span)),
    Use((UTypeHead, Span)),
}
#[derive(Debug, Clone)]
pub struct TypeCheckerCore {
    r: reachability::Reachability,
    types: Vec<TypeNode>,
}
impl TypeCheckerCore {
    pub fn new() -> Self {
        Self {
            r: Default::default(),
            types: Vec::new(),
        }
    }

    pub fn flow(&mut self, lhs: Value, rhs: Use) -> Result<(), TypeError> {
        let mut pending_edges = vec![(lhs, rhs)];
        let mut type_pairs_to_check = Vec::new();
        while let Some((lhs, rhs)) = pending_edges.pop() {
            self.r.add_edge(lhs.0, rhs.0, &mut type_pairs_to_check);

            // Check if adding that edge resulted in any new type pairs needing to be checked
            while let Some((lhs, rhs)) = type_pairs_to_check.pop() {
                if let TypeNode::Value(lhs_head) = &self.types[lhs] {
                    if let TypeNode::Use(rhs_head) = &self.types[rhs] {
                        check_heads(lhs, lhs_head, rhs, rhs_head, &mut pending_edges)?;
                    }
                }
            }
        }
        assert!(pending_edges.is_empty() && type_pairs_to_check.is_empty());
        Ok(())
    }

    fn new_val(&mut self, val_type: VTypeHead, span: Span) -> Value {
        let i = self.r.add_node();
        assert_eq!(i, self.types.len());
        self.types.push(TypeNode::Value((val_type, span)));
        Value(i)
    }

    fn new_use(&mut self, constraint: UTypeHead, span: Span) -> Use {
        let i = self.r.add_node();
        assert_eq!(i, self.types.len());
        self.types.push(TypeNode::Use((constraint, span)));
        Use(i)
    }

    pub fn var(&mut self) -> (Value, Use) {
        let i = self.r.add_node();
        assert_eq!(i, self.types.len());
        self.types.push(TypeNode::Var);
        (Value(i), Use(i))
    }

    pub fn bool(&mut self, span: Span) -> Value {
        self.new_val(VTypeHead::VBool, span)
    }
    pub fn float(&mut self, span: Span) -> Value {
        self.new_val(VTypeHead::VFloat, span)
    }
    pub fn int(&mut self, span: Span) -> Value {
        self.new_val(VTypeHead::VUnboundedInt, span)
    }
    pub fn null(&mut self, span: Span) -> Value {
        self.new_val(VTypeHead::VNull, span)
    }
    pub fn str(&mut self, span: Span) -> Value {
        self.new_val(VTypeHead::VStr, span)
    }
    pub fn int_or_float(&mut self, span: Span) -> Value {
        self.new_val(VTypeHead::VIntOrFloat, span)
    }

    pub fn bool_use(&mut self, span: Span) -> Use {
        self.new_use(UTypeHead::UBool, span)
    }
    pub fn float_use(&mut self, span: Span) -> Use {
        self.new_use(UTypeHead::UFloat, span)
    }
    pub fn int_use(&mut self, span: Span) -> Use {
        self.new_use(UTypeHead::UUnboundedInt, span)
    }
    pub fn null_use(&mut self, span: Span) -> Use {
        self.new_use(UTypeHead::UNull, span)
    }
    pub fn str_use(&mut self, span: Span) -> Use {
        self.new_use(UTypeHead::UStr, span)
    }
    pub fn int_or_float_use(&mut self, span: Span) -> Use {
        self.new_use(UTypeHead::UIntOrFloat, span)
    }

    pub fn func(&mut self, arg: Use, ret: Value, span: Span) -> Value {
        self.new_val(VTypeHead::VFunc { arg, ret }, span)
    }
    pub fn func_use(&mut self, arg: Value, ret: Use, span: Span) -> Use {
        self.new_use(UTypeHead::UFunc { arg, ret }, span)
    }

    pub fn obj(&mut self, fields: Vec<(String, Value)>, proto: Option<Value>, span: Span) -> Value {
        let fields = fields.into_iter().collect();
        self.new_val(VTypeHead::VObj { fields, proto }, span)
    }
    pub fn obj_use(&mut self, field: (String, Use), span: Span) -> Use {
        self.new_use(UTypeHead::UObj { field }, span)
    }

    pub fn case(&mut self, case: (String, Value), span: Span) -> Value {
        self.new_val(VTypeHead::VCase { case }, span)
    }
    pub fn case_use(
        &mut self,
        cases: Vec<(String, (Use, LazyFlow))>,
        wildcard: Option<(Use, LazyFlow)>,
        span: Span,
    ) -> Use {
        let cases = cases.into_iter().collect();
        self.new_use(UTypeHead::UCase { cases, wildcard }, span)
    }

    pub fn reference(&mut self, write: Option<Use>, read: Option<Value>, span: Span) -> Value {
        self.new_val(VTypeHead::VRef { write, read }, span)
    }
    pub fn reference_use(&mut self, write: Option<Value>, read: Option<Use>, span: Span) -> Use {
        self.new_use(UTypeHead::URef { write, read }, span)
    }

    pub fn null_check_use(&mut self, nonnull: Use, span: Span) -> Use {
        self.new_use(UTypeHead::UNullCase { nonnull }, span)
    }

    pub fn save(&self) -> SavePoint {
        (self.types.len(), self.r.clone())
    }
    pub fn restore(&mut self, mut save: SavePoint) {
        self.types.truncate(save.0);
        std::mem::swap(&mut self.r, &mut save.1);
    }
}

type SavePoint = (usize, reachability::Reachability);
