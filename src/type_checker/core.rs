use std::collections::{HashMap, HashSet};

use super::reachability;
use super::set::Set;
use crate::parser::spans::{Span, SpannedError as TypeError};

type ID = usize;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Value(ID);
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Use(ID);

pub type LazyFlow = (Value, Use);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VTypeHead {
    VBool,
    VFloat,
    VNull,
    VStr,
    VIntOrFloat,
    VFunc {
        arg: Vec<Use>,
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
    VArray {
        value: Value,
        index: Use,
        len: Value,
    },
    VBoundedInt {
        set: Set,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UTypeHead {
    UBool,
    UFloat,
    UNull,
    UStr,
    UIntOrFloat,
    UFunc {
        arg: Vec<Value>,
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
    UArray {
        value: Use,
        len: Use,
    },
    UArrayAccess {
        value: Use,
        index: Value,
    },
    UArrayLen {
        length: Use,
    },
    UBoundedInt {
        set: Set,
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
        (&VNull, &UNull) => Ok(()),
        (&VStr, &UStr) => Ok(()),
        (&VFloat, &UIntOrFloat) => Ok(()),
        (&VIntOrFloat, &UIntOrFloat) => Ok(()),
        (&VBoundedInt { .. }, &UIntOrFloat) => Ok(()),

        (
            VFunc {
                arg: arg1,
                ret: ret1,
            },
            UFunc {
                arg: arg2,
                ret: ret2,
            },
        ) => {
            out.push((*ret1, *ret2));
            // flip the order since arguments are contravariant
            if arg1.len() != arg2.len() {
                return Err(TypeError::new2("TypeError: Function called with invalid argument number\nNote: Arguments defined here", lhs.1, "And used here", rhs.1));
            }
            for (arg2, arg1) in arg2.iter().zip(arg1) {
                out.push((*arg2, *arg1));
            }
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
                    format!("TypeError: Missing field {name}\nNote: Field is accessed here"),
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
                    format!("TypeError: Unhandled case {name}\nNote: Case originates here"),
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

        (
            &VArray {
                value: a_value,
                len: a_len,
                ..
            },
            &UArray {
                value: b_value,
                len: b_len,
            },
        ) => {
            out.push((a_value, b_value));
            out.push((a_len, b_len));
            Ok(())
        }

        (
            &VArray {
                value,
                index: a_index,
                ..
            },
            &UArrayAccess { value: used, index },
        ) => {
            out.push((index, a_index));
            out.push((value, used));
            Ok(())
        }
        (&VArray { len, .. }, &UArrayLen { length }) => {
            out.push((len, length));
            Ok(())
        }

        (&VNull, &UNullCase { .. }) => Ok(()),
        (_, &UNullCase { nonnull }) => {
            out.push((Value(lhs_ind), nonnull));
            Ok(())
        }
        (VBoundedInt { set: v_set }, UBoundedInt { set: u_set }) => {
            if v_set.is_subset(u_set) {
                Ok(())
            } else {
                Err(TypeError::new2(
                    format!(
                        "TypeError: integer bounds not satisfied.\nExpected integer {}",
                        u_set.format()
                    ),
                    rhs.1,
                    format!("But found integer {} here.", v_set.format()),
                    lhs.1,
                ))
            }
        }

        _ => {
            let found = match &lhs.0 {
                VBool => "boolean".to_string(),
                VFloat => "float".to_string(),
                VNull => "nil".to_string(),
                VStr => "string".to_string(),
                VIntOrFloat => "float or integer".to_string(),
                VFunc { .. } => "function".to_string(),
                VObj { .. } => "record".to_string(),
                VCase { .. } => "case".to_string(),
                VRef { .. } => "ref".to_string(),
                VArray { .. } => "array".to_string(),
                VBoundedInt { set } => format!("integer {}", set.format()),
            };
            let expected = match &rhs.0 {
                UBool => "boolean".to_string(),
                UFloat => "float".to_string(),
                UNull => "nil".to_string(),
                UStr => "string".to_string(),
                UIntOrFloat => "float or integer".to_string(),
                UFunc { .. } => "function".to_string(),
                UObj { .. } => "record".to_string(),
                UCase { .. } => "case".to_string(),
                URef { .. } => "ref".to_string(),
                UNullCase { .. } => unreachable!(),
                UArray { .. } => "array".to_string(),
                UArrayAccess { .. } => "array access".to_string(),
                UArrayLen { .. } => "array len".to_string(),
                UBoundedInt { set } => format!("integer {}", set.format()),
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

    pub fn debug_value(&self, value: &Value) {
        println!("Type of {value:?}: {:?}", self.types[value.0])
    }

    pub fn debug_use(&self, use_: &Use) {
        println!("Type of {use_:?}: {:?}", self.types[use_.0])
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

    fn resolve_var(&self, id: ID) -> Vec<TypeNode> {
        let mut visited = HashSet::new();
        visited.insert(id);
        let mut to_search = vec![id];
        let mut results = Vec::new();
        while let Some(id) = to_search.pop() {
            for edge in self.r.get_edge(id) {
                visited.insert(*edge);
                match &self.types[*edge] {
                    TypeNode::Var => to_search.push(*edge),
                    val => results.push(val.clone()),
                }
            }
        }
        results
    }

    pub fn get_value(&self, value: Value) -> Vec<VTypeHead> {
        match &self.types[value.0] {
            TypeNode::Value((val, _)) => vec![val.clone()],
            TypeNode::Var => self
                .resolve_var(value.0)
                .into_iter()
                .map(|val| {
                    if let TypeNode::Value((val, _)) = val {
                        val
                    } else {
                        panic!("inconsistent types, got {val:?}")
                    }
                })
                .collect(),
            TypeNode::Use(_) => {
                panic!("Inconsistent types")
            }
        }
    }

    pub fn get_use(&self, use_: Use) -> UTypeHead {
        match &self.types[use_.0] {
            TypeNode::Use((use_, _)) => use_.clone(),
            TypeNode::Var => UTypeHead::UNull,
            TypeNode::Value(_) => {
                panic!("Inconsistent types")
            }
        }
    }

    pub fn bool(&mut self, span: Span) -> Value {
        self.new_val(VTypeHead::VBool, span)
    }
    pub fn float(&mut self, span: Span) -> Value {
        self.new_val(VTypeHead::VFloat, span)
    }
    pub fn int(&mut self, span: Span) -> Value {
        self.bounded_int(Set::new(i64::MIN..=i64::MAX), span)
    }
    pub fn bounded_int(&mut self, set: Set, span: Span) -> Value {
        self.new_val(VTypeHead::VBoundedInt { set }, span)
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
        self.bounded_int_use(Set::new(i64::MIN..=i64::MAX), span)
    }
    pub fn bounded_int_use(&mut self, set: Set, span: Span) -> Use {
        self.new_use(UTypeHead::UBoundedInt { set }, span)
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

    pub fn func(&mut self, arg: Vec<Use>, ret: Value, span: Span) -> Value {
        self.new_val(VTypeHead::VFunc { arg, ret }, span)
    }
    pub fn func_use(&mut self, arg: Vec<Value>, ret: Use, span: Span) -> Use {
        self.new_use(UTypeHead::UFunc { arg, ret }, span)
    }

    pub fn obj(&mut self, fields: Vec<(String, Value)>, proto: Option<Value>, span: Span) -> Value {
        let fields = fields.into_iter().collect();
        self.new_val(VTypeHead::VObj { fields, proto }, span)
    }
    pub fn obj_use(&mut self, field: (String, Use), span: Span) -> Use {
        self.new_use(UTypeHead::UObj { field }, span)
    }

    pub fn array(&mut self, value: Value, len: i64, span: Span) -> Value {
        let index = self.bounded_int_use(Set::new(0..=len - 1), span);
        let len = self.bounded_int(Set::new(len..=len), span);
        self.new_val(VTypeHead::VArray { value, index, len }, span)
    }
    pub fn array_use(&mut self, value: Use, length: i64, span: Span) -> Use {
        let len = self.bounded_int_use(Set::new(length..=length), span);
        self.new_use(UTypeHead::UArray { value, len }, span)
    }
    pub fn array_access(&mut self, value: Use, index: Value, span: Span) -> Use {
        self.new_use(UTypeHead::UArrayAccess { value, index }, span)
    }
    pub fn array_length(&mut self, length: Use, span: Span) -> Use {
        self.new_use(UTypeHead::UArrayLen { length }, span)
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
