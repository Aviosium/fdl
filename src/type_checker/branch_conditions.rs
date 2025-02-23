use std::collections::HashMap;

use crate::parser::{
    SpannedError,
    ast::{Expr, ExprType, Op},
    spans::Span,
};

use super::{
    core::{TypeCheckerCore, VTypeHead, Value},
    set::Set,
    typecheck::Bindings,
};

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum NotNull {
    Undecided,
    True(Value),
    False,
    Conflicting,
}

#[derive(Debug, Clone)]
pub struct Condition {
    pub not_null: NotNull,
    pub range: Option<Set>,
    pub last_span: Span,
}

impl Condition {
    pub fn not_null(value: Value, last_span: Span) -> Self {
        Self {
            not_null: NotNull::True(value),
            range: None,
            last_span,
        }
    }

    pub fn null(last_span: Span) -> Self {
        Self {
            not_null: NotNull::False,
            range: None,
            last_span,
        }
    }

    pub fn ranged(set: Set, last_span: Span) -> Self {
        Self {
            not_null: NotNull::Undecided,
            range: Some(set),
            last_span,
        }
    }

    pub fn and(&mut self, other: Condition, span: Span) {
        match (self.not_null, other.not_null) {
            (NotNull::Undecided, _) => {
                self.not_null = other.not_null;
                self.last_span = other.last_span;
            }
            (_, NotNull::Undecided) => {}
            (NotNull::True(_), NotNull::True(_)) => {
                todo!("merge values")
            }
            (NotNull::False, NotNull::False) => {}
            (NotNull::True(_), NotNull::False) | (NotNull::False, NotNull::True(_)) => {
                self.not_null = NotNull::Conflicting;
                self.last_span = span;
            }
            (NotNull::Conflicting, _) | (_, NotNull::Conflicting) => {
                self.not_null = NotNull::Conflicting
            }
        }
        if let Some(range) = other.range {
            if let Some(existing) = &mut self.range {
                existing.and(&range);
                self.last_span = span;
            } else {
                self.range = Some(range);
                self.last_span = other.last_span;
            }
        }
    }

    pub fn or(&mut self, other: Condition, span: Span) {
        match (self.not_null, other.not_null) {
            (NotNull::Undecided, _) => {}
            (_, NotNull::Undecided) => self.not_null = NotNull::Undecided,
            (NotNull::True(_), NotNull::True(_)) => {
                todo!("merge values")
            }
            (NotNull::False, NotNull::False) => {}
            (NotNull::True(_), NotNull::False) | (NotNull::False, NotNull::True(_)) => {
                self.not_null = NotNull::Undecided
            }
            (NotNull::Conflicting, _) | (_, NotNull::Conflicting) => {
                self.not_null = NotNull::Conflicting
            }
        }
        if let Some(range) = other.range {
            if let Some(existing) = &mut self.range {
                existing.merge(&range);
                self.last_span = span;
            }
        } else if self.range.take().is_some() {
            self.last_span = span;
        }
    }
}

fn combine_and(
    target: &mut HashMap<String, Condition>,
    source: HashMap<String, Condition>,
    span: Span,
) {
    for (key, value) in source {
        if let Some(existing) = target.get_mut(&key) {
            existing.and(value, span);
        } else {
            target.insert(key, value);
        }
    }
}

fn combine_or(
    target: &mut HashMap<String, Condition>,
    mut source: HashMap<String, Condition>,
    span: Span,
) {
    for (key, value) in target.iter_mut() {
        if let Some(other) = source.remove(key) {
            value.or(other, span);
        } else {
            value.range = None;
            value.not_null = NotNull::Undecided;
        }
    }
}

#[derive(Debug, Clone)]
pub struct Conditions {
    pub enforced: HashMap<String, Condition>,
    pub avoided: HashMap<String, Condition>,
}

impl Conditions {
    pub fn new() -> Self {
        Self {
            enforced: HashMap::new(),
            avoided: HashMap::new(),
        }
    }

    pub fn not(&mut self) {
        std::mem::swap(&mut self.enforced, &mut self.avoided);
    }

    pub fn and(&mut self, other: Conditions, span: Span) {
        combine_and(&mut self.enforced, other.enforced, span);
        combine_or(&mut self.avoided, other.avoided, span);
    }

    pub fn or(&mut self, other: Conditions, span: Span) {
        combine_or(&mut self.enforced, other.enforced, span);
        combine_and(&mut self.avoided, other.avoided, span);
    }
}

enum ConditionVariant {
    Null(Value),
    Set(Set),
}

fn get_condition_variant(type_head: Option<VTypeHead>, value: Value) -> Option<ConditionVariant> {
    match type_head {
        Some(VTypeHead::VBoundedInt { set }) => Some(ConditionVariant::Set(set)),
        Some(VTypeHead::VNull { .. }) => Some(ConditionVariant::Null(value)),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn get_binop_conditions(
    op: &Op,
    full_span: Span,
    left_cond: Option<Conditions>,
    right_cond: Option<Conditions>,
    left: &Expr,
    right: &Expr,
    left_type_head: Option<VTypeHead>,
    right_type_head: Option<VTypeHead>,
    left_value: Value,
    right_value: Value,
) -> Option<Conditions> {
    let cond = match (&left.inner, &right.inner) {
        (ExprType::Variable(var), _) => {
            let name = var.to_string();
            let cond = get_condition_variant(right_type_head, left_value);
            cond.map(|cond| (name, cond, false))
        }
        (_, ExprType::Variable(var)) => {
            let name = var.to_string();
            let cond = get_condition_variant(left_type_head, right_value);
            cond.map(|cond| (name, cond, true))
        }
        _ => None,
    };
    match op {
        Op::And => match (left_cond, right_cond) {
            (Some(mut left), Some(right)) => {
                left.and(right, full_span);
                Some(left)
            }
            (Some(left), None) | (None, Some(left)) => Some(left),
            (None, None) => None,
        },
        Op::Or => match (left_cond, right_cond) {
            (Some(mut left), Some(right)) => {
                left.or(right, full_span);
                Some(left)
            }
            (Some(mut left), None) | (None, Some(mut left)) => {
                left.not();
                Some(left)
            }
            (None, None) => None,
        },
        Op::Lt | Op::Gt | Op::Gte | Op::Lte => match cond {
            Some((name, ConditionVariant::Set(set), is_left)) => {
                let mut new = Conditions::new();
                // This works, but could definitely be more beautiful
                let low_inclusive = i64::MIN..=set.max();
                let low_exclusive = i64::MIN..=set.max().saturating_sub(1);
                let high_inclusive = set.min()..=i64::MAX;
                let high_exclusive = set.min().saturating_add(1)..=i64::MAX;
                let (enforced, avoided) = match (op, is_left) {
                    (Op::Lt, false) => (low_exclusive, high_inclusive),
                    (Op::Lt, true) => (high_exclusive, low_inclusive),
                    (Op::Lte, false) => (low_inclusive, high_exclusive),
                    (Op::Lte, true) => (high_inclusive, low_exclusive),
                    (Op::Gt, false) => (high_exclusive, low_inclusive),
                    (Op::Gt, true) => (low_exclusive, high_inclusive),
                    (Op::Gte, false) => (high_inclusive, low_exclusive),
                    (Op::Gte, true) => (low_inclusive, high_exclusive),
                    _ => unreachable!(),
                };
                let enforced = Condition::ranged(Set::new(enforced), full_span);
                let avoided = Condition::ranged(Set::new(avoided), full_span);
                new.enforced.insert(name.clone(), enforced);
                new.avoided.insert(name, avoided);
                Some(new)
            }
            _ => None,
        },
        Op::Eq | Op::Neq => match cond {
            Some((name, ConditionVariant::Set(set), _is_left)) => {
                let mut new = Conditions::new();
                let mut not_set = Set::new(i64::MIN..=set.max().saturating_sub(1));
                not_set.merge(&Set::new(set.min().saturating_add(1)..=i64::MAX));
                let (enforced, avoided) = match op {
                    Op::Eq => (set, not_set),
                    Op::Neq => (not_set, set),
                    _ => unreachable!(),
                };
                let enforced = Condition::ranged(enforced, full_span);
                let avoided = Condition::ranged(avoided, full_span);
                new.enforced.insert(name.clone(), enforced);
                new.avoided.insert(name, avoided);
                Some(new)
            }
            Some((name, ConditionVariant::Null(value), _is_left)) => match op {
                Op::Eq => {
                    let mut new = Conditions::new();
                    let enforced = Condition::null(full_span);
                    let avoided = Condition::not_null(value, full_span);
                    new.enforced.insert(name.clone(), enforced);
                    new.avoided.insert(name, avoided);
                    Some(new)
                }
                Op::Neq => {
                    let mut new = Conditions::new();
                    let enforced = Condition::not_null(value, full_span);
                    let avoided = Condition::null(full_span);
                    new.enforced.insert(name.clone(), enforced);
                    new.avoided.insert(name, avoided);
                    Some(new)
                }
                _ => unreachable!(),
            },
            _ => None,
        },
        _ => None,
    }
}

pub fn apply_conditions(
    engine: &mut TypeCheckerCore,
    bindings: &mut Bindings,
    conditions: &HashMap<String, Condition>,
) -> Result<(), SpannedError> {
    for (name, condition) in conditions {
        let other_val = if let Some(set) = &condition.range {
            let val = engine.bounded_int(set.clone(), condition.last_span);
            bindings.insert(name.clone(), val);
            Some(val)
        } else {
            None
        };
        match condition.not_null {
            NotNull::True(_type) => {
                let (nnvar_type, nnvar_bound) = engine.var();
                let bound = engine.null_check_use(nnvar_bound, condition.last_span);
                if let Some(other_val) = other_val {
                    engine.flow(other_val, bound)?;
                } else {
                    engine.flow(_type, bound)?;
                }

                bindings.insert(name.clone(), nnvar_type);
            }
            NotNull::False => {
                let val = engine.null(condition.last_span);
                bindings.insert(name.clone(), val);
            }
            _ => {}
        }
    }
    Ok(())
}
