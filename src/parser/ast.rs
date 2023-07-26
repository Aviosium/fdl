use super::spans::{Span, Spanned};

#[derive(Debug, Clone)]
pub enum Literal {
    Bool,
    Float,
    Int,
    Nil,
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub enum UnOp {
    Minus,
    Len,
    Not,
    BitNot,
}

#[derive(Debug, Clone)]
pub enum LetPattern {
    Var(String),                                     // TODO
    Record(Vec<(Spanned<String>, Box<LetPattern>)>), // TODO
}

#[derive(Debug, Clone)]
pub enum MatchPattern {
    Case(String, String), // TODO
    Wildcard(String),     // TODO
}

#[derive(Debug, Clone)]
pub enum Expr {
    BinOp(Spanned<Box<Expr>>, Spanned<Box<Expr>>, Op, Span),
    UnOp(Spanned<Box<Expr>>, UnOp, Span),
    Call(String, Vec<Box<Expr>>, Span),
    Case(Spanned<String>, Box<Expr>), // TODO
    FieldAccess(Box<Expr>, String, Span),
    ArrayAccess(Box<Expr>, Box<Expr>, Span),
    If(Spanned<Box<Expr>>, TopLevel, Option<TopLevel>),
    Literal(Literal, Spanned<String>),
    Match(Box<Expr>, Vec<(Spanned<MatchPattern>, Box<Expr>)>, Span), // TODO
    Record(Vec<(Spanned<String>, Box<Expr>)>, Span),
    LiteralArray(Spanned<Vec<Box<Expr>>>),
    RepeatedArray(Spanned<(Box<Expr>, String)>),
    Variable(Spanned<String>),
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
    Func(Spanned<(Box<TypeExpr>, Box<TypeExpr>)>), // TODO
    Ident(Spanned<String>),                        // TODO
    Nullable(Box<TypeExpr>, Span),                 // TODO
    Record(
        Option<Box<TypeExpr>>,
        Vec<(Spanned<String>, Box<TypeExpr>)>,
        Span,
    ), // TODO
    Ref(Box<TypeExpr>, Spanned<Readability>),      // TODO
    TypeVar(Spanned<String>),                      // TODO
    Alias(Box<TypeExpr>, Spanned<String>),
}

#[derive(Debug, Clone)]
pub enum TopLevel {
    Expr(Box<Expr>),
    Block(Vec<TopLevel>, Option<Spanned<Box<Expr>>>),
    LetDef(
        String,
        Option<Spanned<Readability>>,
        Option<Spanned<Box<TypeExpr>>>,
        Box<Expr>,
    ),
    Assign(Spanned<String>, Spanned<Box<Expr>>),
    FuncDef(
        Spanned<(
            String,
            Vec<Spanned<(String, Option<Box<TypeExpr>>)>>,
            Box<TopLevel>,
        )>,
    ),
}
