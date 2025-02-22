use super::spans::{Span, Spanned};
pub use crate::type_checker::core::Value;

#[derive(Debug, Clone, Copy)]
pub enum Literal {
    Bool,
    Float,
    Int,
    Nil,
}

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Add,
    Sub,
    Mult,
    Div,
    IntDiv,
    Rem,
    BitOr,
    BitAnd,
    BitXor,
    BitLsh,
    BitRsh,
    Pot,

    Lt,
    Lte,
    Gt,
    Gte,
    Or,
    And,
    Eq,
    Neq,
}

#[derive(Debug, Clone, Copy)]
pub enum UnOp {
    Minus,
    Len,
    Not,
    BitNot,
}

type SpannedType = Spanned<Box<TypeExpr>>;

#[derive(Debug, Clone)]
pub enum LetPattern {
    Var((String, Option<Spanned<Readability>>, Option<SpannedType>)),
    Record(Vec<(Spanned<String>, Option<SpannedType>, Box<LetPattern>)>),
}

#[derive(Debug, Clone)]
pub enum MatchPattern {
    Case(String, String), // TODO
    Wildcard(String),     // TODO
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub inner: ExprType,
    pub span: Span,
    pub type_: Value,
}

#[derive(Debug, Clone)]
#[allow(clippy::vec_box)]
pub enum ExprType {
    BinOp {
        lhs: Spanned<Box<Expr>>,
        rhs: Spanned<Box<Expr>>,
        op: Op,
    },
    UnOp {
        expr: Spanned<Box<Expr>>,
        op: UnOp,
    },
    Call {
        name: String,
        args: Vec<Box<Expr>>,
    },
    // Case(Spanned<String>, Box<Expr>), // TODO
    FieldAccess {
        obj: Box<Expr>,
        field: String,
    },
    ArrayAccess {
        array: Box<Expr>,
        idx: Box<Expr>,
    },
    If {
        cond: Box<Expr>,
        block: TopLevel,
        else_: Option<TopLevel>,
    },
    Literal {
        type_: Literal,
        value: String,
    },
    // Match(Box<Expr>, Vec<(Spanned<MatchPattern>, Box<Expr>)>, Span), // TODO
    Record(Vec<(Spanned<String>, Box<Expr>)>),
    LiteralArray(Vec<Box<Expr>>),
    RepeatedArray(Box<Expr>, String),
    Variable(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readability {
    ReadWrite,
    ReadOnly,
    WriteOnly,
}

#[derive(Debug, Clone)]
pub enum TypeExpr {
    Case(
        Option<Box<TypeExpr>>,
        Vec<(Spanned<String>, Box<TypeExpr>)>,
        Span,
    ), // TODO
    Ident(Spanned<String>),        // TODO
    Nullable(Box<TypeExpr>, Span), // TODO
    Record(
        Option<Box<TypeExpr>>,
        Vec<(Spanned<String>, Box<TypeExpr>)>,
        Span,
    ), // TODO
    Ref(Box<TypeExpr>, Spanned<Readability>), // TODO
    TypeVar(Spanned<String>),      // TODO
    Alias(Box<TypeExpr>, Spanned<String>),
    BoundedInt(
        Vec<(Option<Spanned<String>>, Option<Spanned<String>>)>,
        Span,
    ),
    NonZero(SpannedType),
    Array(Box<TypeExpr>, String, Span),
}

#[derive(Debug, Clone)]
pub enum TopLevel {
    Expr(Box<Expr>),
    Block(Vec<TopLevel>, Option<Spanned<Box<Expr>>>),
    LetDef(LetPattern, Box<Expr>),
    Assign(Spanned<String>, Spanned<Box<Expr>>),
    FuncDef(
        Spanned<(
            String,
            Vec<Spanned<(String, Box<TypeExpr>)>>,
            SpannedType,
            Box<TopLevel>,
        )>,
    ),
}

impl Default for TopLevel {
    fn default() -> Self {
        TopLevel::Expr(Box::new(Expr {
            inner: ExprType::Literal {
                type_: Literal::Nil,
                value: "nil".to_string(),
            },
            span: Span(0),
            type_: Value(0),
        }))
    }
}

pub(crate) fn create_expr(inner: ExprType, span: Span) -> Box<Expr> {
    Box::new(Expr {
        inner,
        span,
        type_: Value(0),
    })
}
