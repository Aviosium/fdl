use std::collections::HashMap;

use super::core::*;
use super::set::Set;
use crate::parser::ast;
use crate::parser::spans::{Span, SpannedError as SyntaxError};

pub type Result<T> = std::result::Result<T, SyntaxError>;

pub struct Bindings {
    m: HashMap<String, Value>,
    changes: Vec<(String, Option<Value>)>,
    functions: HashMap<String, Value>,
}

impl Bindings {
    pub fn new() -> Self {
        Self {
            m: HashMap::new(),
            changes: Vec::new(),
            functions: HashMap::new(),
        }
    }

    fn get(&self, k: &str) -> Option<&Value> {
        self.m.get(k)
    }

    fn insert(&mut self, k: String, v: Value) {
        let old = self.m.insert(k.clone(), v);
        self.changes.push((k, old));
    }

    fn unwind(&mut self, n: usize) {
        while self.changes.len() > n {
            let (k, old) = self.changes.pop().unwrap();
            match old {
                Some(v) => self.m.insert(k, v),
                None => self.m.remove(&k),
            };
        }
    }

    fn in_child_scope<T>(&mut self, cb: impl FnOnce(&mut Self) -> T) -> T {
        let n = self.changes.len();
        let res = cb(self);
        self.unwind(n);
        res
    }
}

fn parse_type(
    engine: &mut TypeCheckerCore,
    bindings: &mut HashMap<String, ((Value, Use), Span)>,
    tyexpr: &ast::TypeExpr,
) -> Result<(Value, Use)> {
    use ast::TypeExpr::*;
    match tyexpr {
        Alias(lhs, (name, span)) => {
            let (utype_value, utype) = engine.var();
            let (vtype, vtype_bound) = engine.var();

            let old = bindings.insert(name.to_string(), ((utype_value, vtype_bound), *span));
            if let Some((_, old_span)) = old {
                return Err(SyntaxError::new2(
                    format!("SyntaxError: Redefinition of type variable '{}", name),
                    *span,
                    "Note: Type variable was already defined here",
                    old_span,
                ));
            }

            let lhs_type = parse_type(engine, bindings, lhs)?;
            engine.flow(lhs_type.0, vtype_bound)?;
            engine.flow(utype_value, lhs_type.1)?;
            Ok((vtype, utype))
        }
        Case(ext, cases, span) => {
            // Create a dummy variable to use as the lazy flow values
            let dummy = engine.var();
            let (vtype, vtype_bound) = engine.var();

            let utype_wildcard = if let Some(ext) = ext {
                let ext_type = parse_type(engine, bindings, ext)?;
                engine.flow(ext_type.0, vtype_bound)?;
                Some((ext_type.1, dummy))
            } else {
                None
            };

            let mut utype_case_arms = Vec::new();
            for ((tag, tag_span), wrapped_expr) in cases {
                let wrapped_type = parse_type(engine, bindings, wrapped_expr)?;

                let case_value = engine.case((tag.clone(), wrapped_type.0), *tag_span);
                engine.flow(case_value, vtype_bound)?;
                utype_case_arms.push((tag.clone(), (wrapped_type.1, dummy)));
            }

            let utype = engine.case_use(utype_case_arms, utype_wildcard, *span);
            Ok((vtype, utype))
        }
        Ident((s, span)) => match s.as_str() {
            "bool" => Ok((engine.bool(*span), engine.bool_use(*span))),
            "float" => Ok((engine.float(*span), engine.float_use(*span))),
            "int" => Ok((engine.int(*span), engine.int_use(*span))),
            "nil" => Ok((engine.null(*span), engine.null_use(*span))),
            "str" => Ok((engine.str(*span), engine.str_use(*span))),
            "number" => {
                let (vtype, vtype_bound) = engine.var();
                let float_lit = engine.float(*span);
                let int_lit = engine.int(*span);
                engine.flow(float_lit, vtype_bound)?;
                engine.flow(int_lit, vtype_bound)?;
                Ok((vtype, engine.int_or_float_use(*span)))
            }
            "top" => {
                let (_, utype) = engine.var();
                let (vtype, vtype_bound) = engine.var();
                let float_lit = engine.float(*span);
                let bool_lit = engine.bool(*span);
                engine.flow(float_lit, vtype_bound)?;
                engine.flow(bool_lit, vtype_bound)?;
                Ok((vtype, utype))
            }
            "bot" => {
                let (vtype, _) = engine.var();
                let (utype_value, utype) = engine.var();
                let float_lit = engine.float_use(*span);
                let bool_lit = engine.bool_use(*span);
                engine.flow(utype_value, float_lit)?;
                engine.flow(utype_value, bool_lit)?;
                Ok((vtype, utype))
            }
            "_" => Ok(engine.var()),
            _ => Err(SyntaxError::new1(
                "SyntaxError: Unrecognized simple type (choices are bool, float, int, str, number, nil, top, bot, or _)",
                *span,
            )),
        },
        BoundedInt(ranges,  span) => {
            let mut set = Set::empty();
            if ranges.is_empty() {
                set.insert_and_merge(i64::MIN..=i64::MAX)
            }
            for (min, max) in ranges {
                let min = min.as_ref().map(|(v, span)| {
                    v.parse().map_err(|_| SyntaxError::new1("ValueError: integer out of bounds", *span))
                }).transpose()?.unwrap_or(i64::MIN);
                let max = max.as_ref().map(|(v, span)| {
                    v.parse().map_err(|_| SyntaxError::new1("ValueError: integer out of bounds", *span))
                }).transpose()?.unwrap_or(i64::MAX);
                set.insert_and_merge(min..=max)
            }
            Ok((engine.bounded_int(set.clone(), *span), engine.bounded_int_use(set, *span)))
        }
        NonZero((v, span)) => {
            let (_inner_value, inner_use) = parse_type(engine, bindings, v)?;
            let inner_use = engine.get_use(inner_use);
            let UTypeHead::UBoundedInt { mut set } = inner_use else { unreachable!("Unreachable via grammar")};
            set.remove_int(0);
            Ok((engine.bounded_int(set.clone(), *span), engine.bounded_int_use(set, *span)))
        }
        Nullable(lhs, span) => {
            let lhs_type = parse_type(engine, bindings, lhs)?;
            let utype = engine.null_check_use(lhs_type.1, *span);

            let (vtype, vtype_bound) = engine.var();
            let null_lit = engine.null(*span);
            engine.flow(lhs_type.0, vtype_bound)?;
            engine.flow(null_lit, vtype_bound)?;
            Ok((vtype, utype))
        }
        Record(ext, fields, span) => {
            let (utype_value, utype) = engine.var();

            let vtype_wildcard = if let Some(ext) = ext {
                let ext_type = parse_type(engine, bindings, ext)?;
                engine.flow(utype_value, ext_type.1)?;
                Some(ext_type.0)
            } else {
                None
            };

            let mut vtype_fields = Vec::new();

            for ((name, name_span), wrapped_expr) in fields {
                let wrapped_type = parse_type(engine, bindings, wrapped_expr)?;

                let obj_use = engine.obj_use((name.clone(), wrapped_type.1), *name_span);
                engine.flow(utype_value, obj_use)?;
                vtype_fields.push((name.clone(), wrapped_type.0));
            }

            let vtype = engine.obj(vtype_fields, vtype_wildcard, *span);
            Ok((vtype, utype))
        }
        Ref(lhs, (rw, span)) => {
            use ast::Readability::*;
            let lhs_type = parse_type(engine, bindings, lhs)?;

            let write = if *rw == ReadOnly {
                (None, None)
            } else {
                (Some(lhs_type.1), Some(lhs_type.0))
            };
            let read = if *rw == WriteOnly {
                (None, None)
            } else {
                (Some(lhs_type.0), Some(lhs_type.1))
            };

            let vtype = engine.reference(write.0, read.0, *span);
            let utype = engine.reference_use(write.1, read.1, *span);
            Ok((vtype, utype))
        }
        TypeVar((name, span)) => {
            if let Some((res, _)) = bindings.get(name.as_str()) {
                Ok(*res)
            } else {
                Err(SyntaxError::new1(
                    format!("ValueError: Undefined type variable {}", name),
                    *span,
                ))
            }
        }
        Array(inner, count, span) => {
            let count: i64 = count.parse().map_err(|_| SyntaxError::new1("ValueError: integer out of bounds", *span))?;
            let (inner_v, inner_u) = parse_type(engine, bindings, inner)?;
            let value = engine.array(inner_v, count, *span);
            let use_ = engine.array_use(inner_u, count, *span);
            Ok((value, use_))
        }
    }
}

fn parse_type_signature(
    engine: &mut TypeCheckerCore,
    tyexpr: &ast::TypeExpr,
) -> Result<(Value, Use)> {
    let mut bindings = HashMap::new();
    parse_type(engine, &mut bindings, tyexpr)
}

fn process_let_pattern(
    engine: &mut TypeCheckerCore,
    bindings: &mut Bindings,
    pat: &ast::LetPattern,
) -> Result<Use> {
    use ast::LetPattern::*;

    let (mut arg_type, arg_bound) = engine.var();
    match pat {
        Var((name, _readability, type_)) => {
            if let Some((type_, _span)) = type_ {
                let (type_val, type_use) = parse_type_signature(engine, type_)?;
                engine.flow(arg_type, type_use)?;
                arg_type = type_val;
            }
            bindings.insert(name.clone(), arg_type);
            // todo!("readability, type");
        }
        Record(pairs) => {
            let mut field_names = HashMap::with_capacity(pairs.len());

            for ((name, name_span), type_, sub_pattern) in pairs {
                if let Some(old_span) = field_names.insert(name, *name_span) {
                    return Err(SyntaxError::new2(
                        "SyntaxError: Repeated field pattern name",
                        *name_span,
                        "Note: Field was already bound here",
                        old_span,
                    ));
                }

                let field_bound = process_let_pattern(engine, bindings, sub_pattern)?;
                if let Some((type_, _type_span)) = type_ {
                    let (type_val, _type_use) = parse_type_signature(engine, type_)?;
                    engine.flow(type_val, field_bound)?;
                }
                let bound = engine.obj_use((name.clone(), field_bound), *name_span);
                engine.flow(arg_type, bound)?;
            }
        }
    };
    Ok(arg_bound)
}

fn merge_values(mut values: Vec<VTypeHead>) -> Option<VTypeHead> {
    let mut result = values.pop()?;
    for value in values {
        if value == result {
            continue;
        } else if let (
            VTypeHead::VBoundedInt { set: r_set },
            VTypeHead::VBoundedInt { set: l_set },
        ) = (&result, &value)
        {
            let mut set = r_set.clone();
            set.merge(l_set);
            result = VTypeHead::VBoundedInt { set }
        } else {
            return None;
        }
    }
    Some(result)
}

fn new_bounds(
    engine: &mut TypeCheckerCore,
    full_span: &Span,
    l_set: Set,
    r_set: Set,
    function: impl Fn(i64, i64) -> i64,
) -> Result<Value> {
    let min = function(l_set.min(), r_set.min());
    let max = function(l_set.max(), r_set.max());
    Ok(engine.bounded_int(Set::new(min..=max), *full_span))
}
fn check_expr(
    engine: &mut TypeCheckerCore,
    bindings: &mut Bindings,
    expr: &ast::Expr,
) -> Result<Value> {
    use ast::Expr::*;

    match expr {
        BinOp((lhs_expr, lhs_span), (rhs_expr, rhs_span), op, full_span) => {
            use ast::Op::*;
            let lhs_type = check_expr(engine, bindings, lhs_expr)?;
            let rhs_type = check_expr(engine, bindings, rhs_expr)?;
            let lhs_type_head = merge_values(engine.get_value(lhs_type));
            let rhs_type_head = merge_values(engine.get_value(rhs_type));
            let bounded = if let (
                Some(VTypeHead::VBoundedInt { set: l_set }),
                Some(VTypeHead::VBoundedInt { set: r_set }),
            ) = (lhs_type_head, rhs_type_head)
            {
                Some((l_set, r_set))
            } else {
                None
            };

            match op {
                Div => {
                    let lhs_bound = engine.int_or_float_use(*lhs_span);
                    let rhs_bound = engine.int_or_float_use(*rhs_span);
                    engine.flow(lhs_type, lhs_bound)?;
                    engine.flow(rhs_type, rhs_bound)?;
                    Ok(engine.float(*full_span))
                }
                IntDiv => {
                    let lhs_bound = engine.int_use(*lhs_span);
                    let mut rhs_bound = Set::new(i64::MIN..=-1);
                    rhs_bound.insert_and_merge(1..=i64::MAX);
                    let rhs_bound = engine.bounded_int_use(rhs_bound, *rhs_span);
                    engine.flow(lhs_type, lhs_bound)?;
                    engine.flow(rhs_type, rhs_bound)?;
                    if let Some((l_set, r_set)) = bounded {
                        let mut values = vec![
                            l_set.min() / r_set.min(),
                            l_set.min() / r_set.max(),
                            l_set.max() / r_set.min(),
                            l_set.max() / r_set.max(),
                        ];
                        if r_set.min() <= 1 && r_set.max() >= 1 {
                            values.push(l_set.min());
                            values.push(l_set.max());
                        }
                        if r_set.min() <= -1 && r_set.max() >= -1 {
                            values.push(l_set.min() / -1);
                            values.push(l_set.max() / -1);
                        }
                        Ok(engine.bounded_int(
                            Set::new(*values.iter().min().unwrap()..=*values.iter().max().unwrap()),
                            *full_span,
                        ))
                    } else {
                        Ok(engine.int(*full_span))
                    }
                }
                BitAnd | BitOr | BitLsh | BitRsh | BitXor => {
                    let lhs_bound = engine.int_use(*lhs_span);
                    let rhs_bound = engine.int_use(*rhs_span);
                    engine.flow(lhs_type, lhs_bound)?;
                    engine.flow(rhs_type, rhs_bound)?;
                    Ok(engine.int(*full_span))
                }
                Add | Sub | Mult | Pot => {
                    let lhs_bound = engine.int_or_float_use(*lhs_span);
                    let rhs_bound = engine.int_or_float_use(*rhs_span);
                    engine.flow(lhs_type, lhs_bound)?;
                    engine.flow(rhs_type, rhs_bound)?;
                    if let Some((l_set, r_set)) = bounded {
                        let op = match op {
                            Add => |a, b| a + b,
                            Sub => |a, b| a - b,
                            Mult => |a, b| a * b,
                            Pot => |a: i64, b| a.pow(b as u32),
                            _ => unreachable!(),
                        };
                        new_bounds(engine, full_span, l_set, r_set, op)
                    } else {
                        Ok(engine.int_or_float(*full_span))
                    }
                }
                Rem => {
                    let lhs_bound = engine.int_or_float_use(*lhs_span);
                    let rhs_bound = engine.int_or_float_use(*rhs_span);
                    engine.flow(lhs_type, lhs_bound)?;
                    engine.flow(rhs_type, rhs_bound)?;
                    if let Some((_l_set, r_set)) = bounded {
                        Ok(engine.bounded_int(Set::new(0..=r_set.max() - 1), *full_span))
                    } else {
                        Ok(engine.int_or_float(*full_span))
                    }
                }
                Lt | Lte | Gt | Gte => {
                    let lhs_bound = engine.int_or_float_use(*lhs_span);
                    let rhs_bound = engine.int_or_float_use(*rhs_span);
                    engine.flow(lhs_type, lhs_bound)?;
                    engine.flow(rhs_type, rhs_bound)?;
                    Ok(engine.bool(*full_span))
                }
                And | Or => {
                    let lhs_bound = engine.bool_use(*lhs_span);
                    let rhs_bound = engine.bool_use(*rhs_span);
                    engine.flow(lhs_type, lhs_bound)?;
                    engine.flow(rhs_type, rhs_bound)?;
                    Ok(engine.bool(*full_span))
                }
                Eq | Neq => Ok(engine.bool(*full_span)),
            }
        }
        UnOp((expr, expr_span), op, unop_span) => {
            use ast::UnOp::*;
            let expr_type = check_expr(engine, bindings, expr)?;

            match op {
                Minus => {
                    let expr_bound = engine.int_or_float_use(*expr_span);
                    engine.flow(expr_type, expr_bound)?;
                    Ok(engine.int_or_float(*unop_span))
                }
                Len => {
                    let (index_type, index_bound) = engine.var();
                    let expr_bound = engine.array_length(index_bound, *expr_span);
                    engine.flow(expr_type, expr_bound)?;
                    Ok(index_type)
                }
                Not => {
                    let expr_bound = engine.bool_use(*expr_span);
                    engine.flow(expr_type, expr_bound)?;
                    Ok(engine.bool(*unop_span))
                }
                BitNot => {
                    let expr_bound = engine.int_use(*expr_span);
                    engine.flow(expr_type, expr_bound)?;
                    Ok(engine.int(*unop_span))
                }
            }
        }
        Call(func_expr, arg_expr, span) => {
            let mut arguments = Vec::new();
            for arg in arg_expr.clone() {
                arguments.push(check_expr(engine, bindings, &arg)?);
            }

            let (ret_type, ret_bound) = engine.var();
            let bound = engine.func_use(arguments, ret_bound, *span);
            let Some(func_type) = bindings.functions.get(func_expr) else {
                return Err(SyntaxError::new1(
                    format!("NameError: Undefined function {}", func_expr),
                    *span,
                ));
            };
            engine.flow(*func_type, bound)?;
            Ok(ret_type)
        }
        Case((tag, span), val_expr) => {
            let val_type = check_expr(engine, bindings, val_expr)?;
            Ok(engine.case((tag.clone(), val_type), *span))
        }
        FieldAccess(lhs_expr, name, span) => {
            let lhs_type = check_expr(engine, bindings, lhs_expr)?;

            let (field_type, field_bound) = engine.var();
            let bound = engine.obj_use((name.clone(), field_bound), *span);
            engine.flow(lhs_type, bound)?;
            Ok(field_type)
        }
        If((cond_expr, span), then_expr, else_expr) => {
            // Handle conditions of the form foo == null and foo != null specially
            if let BinOp((lhs, _), (rhs, _), op, ..) = &**cond_expr {
                if let Variable((name, _)) = &**lhs {
                    if let Literal(ast::Literal::Nil, ..) = **rhs {
                        if let Some(lhs_type) = bindings.get(name.as_str()) {
                            // Flip order of branches if they wrote if foo == null instead of !=
                            let (ok_expr, else_expr) = match op {
                                ast::Op::Neq => {
                                    (then_expr.clone(), else_expr.clone().unwrap_or_default())
                                }
                                ast::Op::Eq => {
                                    (else_expr.clone().unwrap_or_default(), then_expr.clone())
                                }
                                _ => unreachable!(),
                            };

                            let (nnvar_type, nnvar_bound) = engine.var();
                            let bound = engine.null_check_use(nnvar_bound, *span);
                            engine.flow(*lhs_type, bound)?;

                            let ok_type = bindings.in_child_scope(|bindings| {
                                bindings.insert(name.clone(), nnvar_type);
                                check_toplevel(engine, bindings, &ok_expr)
                            })?;
                            let else_type = check_toplevel(engine, bindings, &else_expr)?;

                            let (merged, merged_bound) = engine.var();
                            engine.flow(ok_type, merged_bound)?;
                            engine.flow(else_type, merged_bound)?;
                            return Ok(merged);
                        }
                    }
                }
            }

            let cond_type = check_expr(engine, bindings, cond_expr)?;
            let bound = engine.bool_use(*span);
            engine.flow(cond_type, bound)?;

            let then_type = check_toplevel(engine, bindings, then_expr)?;
            let else_type = match else_expr {
                Some(else_expr) => check_toplevel(engine, bindings, else_expr)?,
                None => engine.null(*span),
            };

            let (merged, merged_bound) = engine.var();
            engine.flow(then_type, merged_bound)?;
            engine.flow(else_type, merged_bound)?;
            Ok(merged)
        }
        Literal(type_, (code, span)) => {
            use ast::Literal::*;
            let span = *span;
            Ok(match type_ {
                Bool => engine.bool(span),
                Float => engine.float(span),
                Int => {
                    let val = code.parse().map_err(|_| {
                        SyntaxError::new1("ValueError: integer out of bounds", span)
                    })?;
                    engine.bounded_int(Set::new(val..=val), span)
                }
                Nil => engine.null(span),
            })
        }
        Match(match_expr, cases, span) => {
            let match_type = check_expr(engine, bindings, match_expr)?;
            let (result_type, result_bound) = engine.var();

            // Result types from the match arms
            let mut case_type_pairs = Vec::with_capacity(cases.len());
            let mut wildcard_type = None;

            // Pattern reachability checking
            let mut case_names = HashMap::with_capacity(cases.len());
            let mut wildcard = None;

            for ((pattern, pattern_span), rhs_expr) in cases {
                if let Some(old_span) = wildcard {
                    return Err(SyntaxError::new2(
                        "SyntaxError: Unreachable match pattern",
                        *pattern_span,
                        "Note: Unreachable due to previous wildcard pattern here",
                        old_span,
                    ));
                }

                use ast::MatchPattern::*;
                match pattern {
                    Case(tag, name) => {
                        if let Some(old_span) = case_names.insert(tag, *pattern_span) {
                            return Err(SyntaxError::new2(
                                "SyntaxError: Unreachable match pattern",
                                *pattern_span,
                                "Note: Unreachable due to previous case pattern here",
                                old_span,
                            ));
                        }

                        let (wrapped_type, wrapped_bound) = engine.var();
                        let rhs_type = bindings.in_child_scope(|bindings| {
                            bindings.insert(name.clone(), wrapped_type);
                            check_expr(engine, bindings, rhs_expr)
                        })?;

                        case_type_pairs
                            .push((tag.clone(), (wrapped_bound, (rhs_type, result_bound))));
                    }
                    Wildcard(name) => {
                        wildcard = Some(*pattern_span);

                        let (wrapped_type, wrapped_bound) = engine.var();
                        let rhs_type = bindings.in_child_scope(|bindings| {
                            bindings.insert(name.clone(), wrapped_type);
                            check_expr(engine, bindings, rhs_expr)
                        })?;

                        wildcard_type = Some((wrapped_bound, (rhs_type, result_bound)));
                    }
                }
            }

            let bound = engine.case_use(case_type_pairs, wildcard_type, *span);
            engine.flow(match_type, bound)?;

            Ok(result_type)
        }
        Record(fields, span) => {
            let mut field_names = HashMap::with_capacity(fields.len());
            let mut field_type_pairs = Vec::with_capacity(fields.len());
            for ((name, name_span), expr) in fields {
                if let Some(old_span) = field_names.insert(name, *name_span) {
                    return Err(SyntaxError::new2(
                        "SyntaxError: Repeated field name",
                        *name_span,
                        "Note: Field was already defined here",
                        old_span,
                    ));
                }

                let t = check_expr(engine, bindings, expr)?;
                field_type_pairs.push((name.clone(), t));
            }
            Ok(engine.obj(field_type_pairs, None, *span))
        }
        Variable((name, span)) => {
            if let Some(v) = bindings.get(name.as_str()) {
                Ok(*v)
            } else {
                Err(SyntaxError::new1(
                    format!("NameError: Undefined variable {}", name),
                    *span,
                ))
            }
        }
        ArrayAccess(array, idx, span) => {
            let array = check_expr(engine, bindings, array)?;
            let idx = check_expr(engine, bindings, idx)?;
            let (field_type, field_bound) = engine.var();
            let bound = engine.array_access(field_bound, idx, *span);
            engine.flow(array, bound)?;
            Ok(field_type)
        }
        LiteralArray((entries, span)) => {
            let (field_type, field_bound) = engine.var();
            let len = entries.len() as i64;
            for entry in entries {
                let expr = check_expr(engine, bindings, entry)?;
                engine.flow(expr, field_bound)?;
            }
            let array = engine.array(field_type, len, *span);
            Ok(array)
        }
        RepeatedArray(((value, rep), span)) => {
            let (field_type, field_bound) = engine.var();
            let expr = check_expr(engine, bindings, value)?;
            engine.flow(expr, field_bound)?;
            let rep = rep
                .parse()
                .map_err(|_| SyntaxError::new1("ValueError: length out of bounds", *span))?;
            if rep <= 0 {
                return Err(SyntaxError::new1(
                    "ValueError: length must be greater than 0",
                    *span,
                ));
            }
            let array = engine.array(field_type, rep, *span);
            Ok(array)
        }
    }
}

fn check_let_def(
    engine: &mut TypeCheckerCore,
    bindings: &mut Bindings,
    lhs: &ast::LetPattern,
    expr: &ast::Expr,
) -> Result<()> {
    let var_type = check_expr(engine, bindings, expr)?;
    let bound = process_let_pattern(engine, bindings, lhs)?;
    engine.flow(var_type, bound)?;
    Ok(())
}
pub fn check_toplevel(
    engine: &mut TypeCheckerCore,
    bindings: &mut Bindings,
    def: &ast::TopLevel,
) -> Result<Value> {
    use ast::TopLevel::*;
    match def {
        Expr(expr) => Ok(check_expr(engine, bindings, expr)?),
        LetDef(pattern, var_expr) => {
            check_let_def(engine, bindings, pattern, var_expr)?;
            Ok(engine.null(Span(0)))
        }
        FuncDef(((name, arg_pattern, retype, body), span)) => {
            let (uses, body_type) = bindings.in_child_scope(|bindings| {
                let mut argument_uses = Vec::new();
                for ((name, type_), _span) in arg_pattern {
                    let (type_v, type_u) = parse_type_signature(engine, type_)?;
                    bindings.insert(name.clone(), type_v);
                    argument_uses.push(type_u);
                }
                let body_type = check_toplevel(engine, bindings, body)?;
                Ok((argument_uses, body_type))
            })?;
            let (retype_value, retype_use) = parse_type_signature(engine, &retype.0)?;
            bindings
                .functions
                .insert(name.clone(), engine.func(uses, retype_value, *span));
            engine.flow(body_type, retype_use)?;
            Ok(engine.null(Span(0)))
        }
        Block(block, ret_type) => {
            for elem in block {
                check_toplevel(engine, bindings, elem)?;
            }
            if let Some(ret_type) = ret_type {
                check_expr(engine, bindings, &ret_type.0)
            } else {
                Ok(engine.null(Span(0)))
            }
        }
        Assign((name, name_span), (value, _value_span)) => {
            let Some(scheme) = bindings.get(name) else {
                return Err(SyntaxError::new1(
                    format!("NameError: Undefined variable {}", name),
                    *name_span,
                ));
            };
            let (val, use_) = engine.var();
            engine.flow(*scheme, use_)?;
            let value = check_expr(engine, bindings, value)?;
            engine.flow(value, use_)?;
            bindings.insert(name.clone(), val);
            Ok(engine.null(Span(0)))
        }
    }
}

pub struct TypeckState {
    core: TypeCheckerCore,
    bindings: Bindings,
}
impl TypeckState {
    pub fn new() -> Self {
        Self {
            core: TypeCheckerCore::new(),
            bindings: Bindings::new(),
        }
    }

    pub fn check_script(&mut self, parsed: &[ast::TopLevel]) -> Result<()> {
        // Create temporary copy of the entire type state so we can roll
        // back all the changes if the script contains an error.
        let temp = self.core.save();

        for item in parsed {
            if let Err(e) = check_toplevel(&mut self.core, &mut self.bindings, item) {
                // Roll back changes to the type state and bindings
                self.core.restore(temp);
                self.bindings.unwind(0);
                return Err(e);
            }
        }

        // Now that script type-checked successfully, make the global definitions permanent
        // by removing them from the changes rollback list
        self.bindings.changes.clear();
        Ok(())
    }
}

impl Default for TypeckState {
    fn default() -> Self {
        Self::new()
    }
}
