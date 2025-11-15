// Abstract Syntax Tree definitions for Lumen

use std::fmt;

/// Source location for error reporting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Default for Span {
    fn default() -> Self {
        Self { start: 0, end: 0 }
    }
}

/// Top-level program
#[derive(Debug, Clone)]
pub struct Program {
    pub declarations: Vec<Declaration>,
}

#[derive(Debug, Clone)]
pub enum Declaration {
    Dim(DimDecl),
    Function(FunctionDecl),
    Model(ModelDecl),
}

/// Dimension declaration: dim batch, dim features = 784
#[derive(Debug, Clone)]
pub struct DimDecl {
    pub name: String,
    pub value: Option<i64>,  // None = symbolic, Some(n) = concrete
    pub span: Span,
}

/// Identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

impl Ident {
    pub fn new(name: String, span: Span) -> Self {
        Self { name, span }
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Type system
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Primitive(PrimitiveType),
    Tensor {
        dtype: Box<Type>,
        dims: Vec<DimExpr>,
    },
    Named(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveType {
    I32,
    I64,
    F32,
    F64,
    Bool,
}

impl fmt::Display for PrimitiveType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PrimitiveType::I32 => write!(f, "i32"),
            PrimitiveType::I64 => write!(f, "i64"),
            PrimitiveType::F32 => write!(f, "f32"),
            PrimitiveType::F64 => write!(f, "f64"),
            PrimitiveType::Bool => write!(f, "bool"),
        }
    }
}

/// Dimension expression
#[derive(Debug, Clone, PartialEq)]
pub enum DimExpr {
    Const(i64),
    Var(String),
    Binary {
        op: BinaryOp,
        left: Box<DimExpr>,
        right: Box<DimExpr>,
    },
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    MatMul,  // @
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Mod => write!(f, "%"),
            BinaryOp::Pow => write!(f, "**"),
            BinaryOp::Eq => write!(f, "=="),
            BinaryOp::Ne => write!(f, "!="),
            BinaryOp::Lt => write!(f, "<"),
            BinaryOp::Le => write!(f, "<="),
            BinaryOp::Gt => write!(f, ">"),
            BinaryOp::Ge => write!(f, ">="),
            BinaryOp::And => write!(f, "and"),
            BinaryOp::Or => write!(f, "or"),
            BinaryOp::MatMul => write!(f, "@"),
        }
    }
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

/// Model declaration
#[derive(Debug, Clone)]
pub struct ModelDecl {
    pub name: String,
    pub members: Vec<ModelMember>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ModelMember {
    Param(ParamDecl),
    Function(FunctionDecl),
}

/// Parameter declaration in model
#[derive(Debug, Clone)]
pub struct ParamDecl {
    pub name: String,
    pub ty: Type,
    pub initializer: Initializer,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Initializer {
    Zeros,
    Ones,
    Normal { mean: f64, std: f64 },
    Uniform { low: f64, high: f64 },
    Xavier,
    Kaiming,
}

/// Function declaration
#[derive(Debug, Clone)]
pub struct FunctionDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<Type>,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

/// Block of statements
#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

/// Statements
#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        name: String,
        ty: Option<Type>,
        value: Expr,
        span: Span,
    },
    Return {
        value: Option<Expr>,
        span: Span,
    },
    Expr(Expr),
}

/// Expressions
#[derive(Debug, Clone)]
pub enum Expr {
    Literal(Literal, Span),
    Ident(String, Span),
    FieldAccess {
        object: Box<Expr>,
        field: String,
        span: Span,
    },
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
        span: Span,
    },
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
        span: Span,
    },
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Literal(_, s) => *s,
            Expr::Ident(_, s) => *s,
            Expr::FieldAccess { span, .. } => *span,
            Expr::Call { span, .. } => *span,
            Expr::Binary { span, .. } => *span,
            Expr::Unary { span, .. } => *span,
        }
    }
}

/// Literals
#[derive(Debug, Clone)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
}
