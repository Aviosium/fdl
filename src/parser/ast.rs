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
#[allow(clippy::vec_box)]
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
        TopLevel::Expr(Box::new(Expr::Literal(
            Literal::Nil,
            ("nil".to_string(), Span(0)),
        )))
    }
}
