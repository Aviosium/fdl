use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::error;
use std::fmt;
use std::rc::Rc;

use super::core::*;
use crate::parser::ast;
use crate::parser::spans::{Span, SpannedError as SyntaxError};

type Result<T> = std::result::Result<T, SyntaxError>;

#[derive(Clone)]
enum Scheme {
    Mono(Value),
    Poly(Rc<dyn Fn(&mut TypeCheckerCore) -> Result<Value>>),
}

struct Bindings {
    m: HashMap<String, Scheme>,
    changes: Vec<(String, Option<Scheme>)>,
}
impl Bindings {
    fn new() -> Self {
        Self {
            m: HashMap::new(),
            changes: Vec::new(),
        }
    }

    fn get(&self, k: &str) -> Option<&Scheme> {
        self.m.get(k)
    }

    fn insert_scheme(&mut self, k: String, v: Scheme) {
        let old = self.m.insert(k.clone(), v);
        self.changes.push((k, old));
    }

    fn insert(&mut self, k: String, v: Value) {
        self.insert_scheme(k, Scheme::Mono(v))
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
        Func(((lhs, rhs), span)) => {
            let lhs_type = parse_type(engine, bindings, lhs)?;
            let rhs_type = parse_type(engine, bindings, rhs)?;

            let utype = engine.func_use(lhs_type.0, rhs_type.1, *span);
            let vtype = engine.func(lhs_type.1, rhs_type.0, *span);
            Ok((vtype, utype))
        }
        Ident((s, span)) => match s.as_str() {
            "bool" => Ok((engine.bool(*span), engine.bool_use(*span))),
            "float" => Ok((engine.float(*span), engine.float_use(*span))),
            "int" => Ok((engine.int(*span), engine.int_use(*span))),
            "null" => Ok((engine.null(*span), engine.null_use(*span))),
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
                "SyntaxError: Unrecognized simple type (choices are bool, float, int, str, number, null, top, bot, or _)",
                *span,
            )),
        },
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
                    format!("SyntaxError: Undefined type variable {}", name),
                    *span,
                ))
            }
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

    let (arg_type, arg_bound) = engine.var();
    match pat {
        Var(name) => {
            bindings.insert(name.clone(), arg_type);
        }
        Record(pairs) => {
            let mut field_names = HashMap::with_capacity(pairs.len());

            for ((name, name_span), sub_pattern) in pairs {
                if let Some(old_span) = field_names.insert(&*name, *name_span) {
                    return Err(SyntaxError::new2(
                        "SyntaxError: Repeated field pattern name",
                        *name_span,
                        "Note: Field was already bound here",
                        old_span,
                    ));
                }

                let field_bound = process_let_pattern(engine, bindings, &*sub_pattern)?;
                let bound = engine.obj_use((name.clone(), field_bound), *name_span);
                engine.flow(arg_type, bound)?;
            }
        }
    };
    Ok(arg_bound)
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

            Ok(match op {
                Div => {
                    let lhs_bound = engine.int_or_float_use(*lhs_span);
                    let rhs_bound = engine.int_or_float_use(*rhs_span);
                    engine.flow(lhs_type, lhs_bound)?;
                    engine.flow(rhs_type, rhs_bound)?;
                    engine.float(*full_span)
                }
                IntDiv => {
                    let lhs_bound = engine.int_or_float_use(*lhs_span);
                    let rhs_bound = engine.int_or_float_use(*rhs_span);
                    engine.flow(lhs_type, lhs_bound)?;
                    engine.flow(rhs_type, rhs_bound)?;
                    engine.int(*full_span)
                }
                BitAnd | BitOr | BitLsh | BitRsh | BitXor => {
                    let lhs_bound = engine.int_use(*lhs_span);
                    let rhs_bound = engine.int_use(*rhs_span);
                    engine.flow(lhs_type, lhs_bound)?;
                    engine.flow(rhs_type, rhs_bound)?;
                    engine.int(*full_span)
                }
                Add | Sub | Mult | Pot | Rem => {
                    let lhs_bound = engine.int_or_float_use(*lhs_span);
                    let rhs_bound = engine.int_or_float_use(*rhs_span);
                    engine.flow(lhs_type, lhs_bound)?;
                    engine.flow(rhs_type, rhs_bound)?;
                    engine.int_or_float(*full_span)
                }
                Lt | Lte | Gt | Gte => {
                    let lhs_bound = engine.int_or_float_use(*lhs_span);
                    let rhs_bound = engine.int_or_float_use(*rhs_span);
                    engine.flow(lhs_type, lhs_bound)?;
                    engine.flow(rhs_type, rhs_bound)?;
                    engine.bool(*full_span)
                }
                And | Or => {
                    let lhs_bound = engine.bool_use(*lhs_span);
                    let rhs_bound = engine.bool_use(*rhs_span);
                    engine.flow(lhs_type, lhs_bound)?;
                    engine.flow(rhs_type, rhs_bound)?;
                    engine.bool(*full_span)
                }
                Eq | Neq => engine.bool(*full_span),
            })
        }
        Call(func_expr, arg_expr, span) => {
            let func_type = check_expr(engine, bindings, func_expr)?;
            let arg_type = check_expr(engine, bindings, arg_expr)?;

            let (ret_type, ret_bound) = engine.var();
            let bound = engine.func_use(arg_type, ret_bound, *span);
            engine.flow(func_type, bound)?;
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
            /*if let BinOp((lhs, _), (rhs, _), ast::OpType::AnyCmp, op, ..) = &**cond_expr {
                if let Variable((name, _)) = &**lhs {
                    if let Literal(ast::Literal::Null, ..) = **rhs {
                        if let Some(scheme) = bindings.get(name.as_str()) {
                            if let Scheme::Mono(lhs_type) = scheme {
                                // Flip order of branches if they wrote if foo == null instead of !=
                                let (ok_expr, else_expr) = match op {
                                    ast::Op::Neq => (then_expr, else_expr),
                                    ast::Op::Eq => (else_expr, then_expr),
                                    _ => unreachable!(),
                                };

                                let (nnvar_type, nnvar_bound) = engine.var();
                                let bound = engine.null_check_use(nnvar_bound, *span);
                                engine.flow(*lhs_type, bound)?;

                                let ok_type = bindings.in_child_scope(|bindings| {
                                    bindings.insert(name.clone(), nnvar_type);
                                    check_expr(engine, bindings, ok_expr)
                                })?;
                                let else_type = check_expr(engine, bindings, else_expr)?;

                                let (merged, merged_bound) = engine.var();
                                engine.flow(ok_type, merged_bound)?;
                                engine.flow(else_type, merged_bound)?;
                                return Ok(merged);
                            }
                        }
                    }
                }
            }*/

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
                Int => engine.int(span),
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
                        if let Some(old_span) = case_names.insert(&*tag, *pattern_span) {
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
                if let Some(old_span) = field_names.insert(&*name, *name_span) {
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
            if let Some(scheme) = bindings.get(name.as_str()) {
                match scheme {
                    Scheme::Mono(v) => Ok(*v),
                    Scheme::Poly(cb) => cb(engine),
                }
            } else {
                Err(SyntaxError::new1(
                    format!("SyntaxError: Undefined variable {}", name),
                    *span,
                ))
            }
        }
    }
}

fn check_let(
    engine: &mut TypeCheckerCore,
    bindings: &mut Bindings,
    expr: &ast::Expr,
) -> Result<Scheme> {
    if let ast::Expr::FuncDef(..) = expr {
        let saved_bindings = RefCell::new(Bindings {
            m: bindings.m.clone(),
            changes: Vec::new(),
        });
        let saved_expr = expr.clone();

        let f: Rc<dyn Fn(&mut TypeCheckerCore) -> Result<Value>> = Rc::new(move |engine| {
            check_expr(engine, &mut saved_bindings.borrow_mut(), &saved_expr)
        });

        f(engine)?;
        Ok(Scheme::Poly(f))
    } else {
        let var_type = check_expr(engine, bindings, expr)?;
        Ok(Scheme::Mono(var_type))
    }
}

fn check_toplevel(
    engine: &mut TypeCheckerCore,
    bindings: &mut Bindings,
    def: &ast::TopLevel,
) -> Result<Value> {
    use ast::TopLevel::*;
    match def {
        Expr(expr) => Ok(check_expr(engine, bindings, expr)?),
        LetDef(name, readability, type_, var_expr) => {
            let var_scheme = check_let(engine, bindings, var_expr)?;
            bindings.insert_scheme(name.clone(), var_scheme);
            Ok(engine.null(Span(0))) // Span should be unreachable due to parser structure
        }
        FuncDef(((name, arg_pattern, body), span)) => {
            let (arg_bound, body_type) = bindings.in_child_scope(|bindings| {
                let arg_bound = process_let_pattern(engine, bindings, arg_pattern)?;
                let body_type = check_toplevel(engine, bindings, body)?;
                Ok((arg_bound, body_type))
            })?;
            Ok(engine.func(arg_bound, body_type, *span))
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
