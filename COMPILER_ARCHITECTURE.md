# Lumen Compiler Architecture

## 1. Overview

The Lumen compiler is designed as a multi-pass, ahead-of-time (AOT) compiler that transforms Lumen source code into optimized executable code. The architecture prioritizes:

1. **Modularity**: Clear separation between compilation stages
2. **Static Analysis**: Maximum compile-time verification
3. **Optimization**: Multiple IR-level transformation passes
4. **Extensibility**: Easy addition of new backends and optimizations

### Implementation Language

**Primary: Rust**

Rationale:
- Strong type system aids compiler correctness
- Memory safety without GC overhead
- Excellent ecosystem for parsers (pest, lalrpop, nom)
- LLVM bindings available (inkwell)
- Performance suitable for compiler workloads

---

## 2. Compilation Pipeline

```
┌──────────────────┐
│  Lumen Source    │
│    (.lumen)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Lexer/Parser   │  ◄── pest grammar
│                  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   AST (typed)    │  ◄── syntax tree with source spans
│                  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Name Resolution │  ◄── symbol table, scope analysis
│                  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Type & Shape    │  ◄── bidirectional type checking
│  Inference       │      constraint solving
│                  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  HIR (High-level │  ◄── desugared, type-annotated IR
│  Intermediate    │
│  Representation) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Autograd        │  ◄── reverse-mode AD graph construction
│  Transform       │      gradient computation
│                  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  MIR (Mid-level  │  ◄── computation graph with gradients
│  IR)             │      control flow lowering
│                  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Optimization    │  ◄── fusion, CSE, DCE, constant folding
│  Passes          │      memory planning
│                  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LIR (Low-level  │  ◄── scheduled operations
│  IR)             │      memory layout decisions
│                  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Code Generation │  ◄── LLVM IR / C++ / CUDA
│                  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Executable /    │
│  Library         │
└──────────────────┘
```

---

## 3. Component Design

### 3.1 Lexer & Parser

#### 3.1.1 Technology Choice

**pest** (PEG parser generator) or **lalrpop** (LR parser generator)

- **pest**: Easier to write, better error messages, PEG suitable for DSL
- **lalrpop**: More powerful, LR(1) parsing, better performance

**Recommendation**: Start with **pest** for rapid prototyping

#### 3.1.2 Parser Output: AST

```rust
// src/ast.rs

use std::fmt;

/// Source location for error reporting
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub column: usize,
}

/// Top-level AST node
#[derive(Debug, Clone)]
pub struct Program {
    pub declarations: Vec<Declaration>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Declaration {
    Dim(DimDecl),
    Const(ConstDecl),
    Function(FunctionDecl),
    Model(ModelDecl),
    Train(TrainDecl),
}

// Dimension declaration
#[derive(Debug, Clone)]
pub struct DimDecl {
    pub name: Ident,
    pub value: Option<DimExpr>,  // None = symbolic, Some = concrete
    pub constraint: Option<Constraint>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum DimExpr {
    Const(i64),
    Var(Ident),
    Binary {
        op: BinaryOp,
        left: Box<DimExpr>,
        right: Box<DimExpr>,
    },
}

#[derive(Debug, Clone)]
pub enum Constraint {
    Binary {
        op: ComparisonOp,
        left: DimExpr,
        right: DimExpr,
    },
    And(Box<Constraint>, Box<Constraint>),
    Or(Box<Constraint>, Box<Constraint>),
}

#[derive(Debug, Clone, Copy)]
pub enum ComparisonOp {
    Eq, Ne, Lt, Le, Gt, Ge, Mod,
}

// Type AST
#[derive(Debug, Clone)]
pub enum Type {
    Primitive(PrimitiveType),
    Tensor {
        dtype: Box<Type>,
        dims: Vec<DimExpr>,
    },
    Function {
        params: Vec<Type>,
        ret: Option<Box<Type>>,
    },
    Named(Ident),
}

#[derive(Debug, Clone, Copy)]
pub enum PrimitiveType {
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    F16, BF16, F32, F64,
    Bool,
}

// Model declaration
#[derive(Debug, Clone)]
pub struct ModelDecl {
    pub name: Ident,
    pub members: Vec<ModelMember>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ModelMember {
    Param(ParamDecl),
    Buffer(BufferDecl),
    Function(FunctionDecl),
}

#[derive(Debug, Clone)]
pub struct ParamDecl {
    pub name: Ident,
    pub ty: Type,
    pub initializer: Option<Initializer>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Initializer {
    Zeros,
    Ones,
    Normal { mean: Expr, std: Expr },
    Uniform { low: Expr, high: Expr },
    Xavier,
    Kaiming,
    Constant(Expr),
}

// Function declaration
#[derive(Debug, Clone)]
pub struct FunctionDecl {
    pub name: Ident,
    pub params: Vec<Param>,
    pub return_type: Option<Type>,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: Ident,
    pub ty: Type,
    pub span: Span,
}

// Statements
#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        pattern: Pattern,
        ty: Option<Type>,
        value: Expr,
        span: Span,
    },
    Assign {
        target: Expr,
        value: Expr,
        span: Span,
    },
    If {
        condition: Expr,
        then_block: Block,
        else_block: Option<Box<Stmt>>,
        span: Span,
    },
    For {
        var: Ident,
        range: Expr,
        body: Block,
        span: Span,
    },
    While {
        condition: Expr,
        body: Block,
        span: Span,
    },
    Minimize {
        loss: Expr,
        span: Span,
    },
    Return {
        value: Option<Expr>,
        span: Span,
    },
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub enum Pattern {
    Ident(Ident),
    Tuple(Vec<Ident>),
}

// Expressions
#[derive(Debug, Clone)]
pub enum Expr {
    Literal(Literal, Span),
    Ident(Ident),
    FieldAccess {
        object: Box<Expr>,
        field: Ident,
        span: Span,
    },
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },
    Index {
        object: Box<Expr>,
        indices: Vec<Expr>,
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
    If {
        condition: Box<Expr>,
        then_block: Block,
        else_block: Block,
        span: Span,
    },
    Block(Block),
}

#[derive(Debug, Clone)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Tensor(Vec<Expr>),
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add, Sub, Mul, Div, Mod, Pow,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or,
    MatMul,  // @
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Neg, Not,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

// Training block
#[derive(Debug, Clone)]
pub struct TrainDecl {
    pub stmts: Vec<TrainStmt>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum TrainStmt {
    UseModel {
        model_type: Type,
        name: Ident,
        span: Span,
    },
    Dataset {
        name: Ident,
        args: Vec<(Ident, Expr)>,
        pattern: Pattern,
        span: Span,
    },
    Optimizer {
        name: Ident,
        args: Vec<(Ident, Expr)>,
        span: Span,
    },
    Stmt(Stmt),
}
```

---

### 3.2 Symbol Table & Name Resolution

```rust
// src/resolve.rs

use std::collections::HashMap;
use crate::ast::*;

pub type SymbolId = usize;

#[derive(Debug, Clone)]
pub struct SymbolTable {
    scopes: Vec<Scope>,
    current_scope: usize,
    symbols: Vec<Symbol>,
}

#[derive(Debug, Clone)]
pub struct Scope {
    parent: Option<usize>,
    bindings: HashMap<String, SymbolId>,
}

#[derive(Debug, Clone)]
pub enum Symbol {
    Dim {
        name: String,
        value: Option<DimExpr>,
        constraint: Option<Constraint>,
    },
    Const {
        name: String,
        ty: Type,
    },
    Function {
        name: String,
        params: Vec<Type>,
        return_type: Option<Type>,
    },
    Model {
        name: String,
        members: HashMap<String, SymbolId>,
    },
    Param {
        name: String,
        ty: Type,
    },
    Variable {
        name: String,
        ty: Option<Type>,  // inferred during type checking
        mutable: bool,
    },
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            scopes: vec![Scope { parent: None, bindings: HashMap::new() }],
            current_scope: 0,
            symbols: Vec::new(),
        }
    }

    pub fn enter_scope(&mut self) {
        let parent = self.current_scope;
        self.scopes.push(Scope {
            parent: Some(parent),
            bindings: HashMap::new(),
        });
        self.current_scope = self.scopes.len() - 1;
    }

    pub fn exit_scope(&mut self) {
        if let Some(parent) = self.scopes[self.current_scope].parent {
            self.current_scope = parent;
        }
    }

    pub fn insert(&mut self, name: String, symbol: Symbol) -> SymbolId {
        let id = self.symbols.len();
        self.symbols.push(symbol);
        self.scopes[self.current_scope].bindings.insert(name, id);
        id
    }

    pub fn lookup(&self, name: &str) -> Option<SymbolId> {
        let mut scope_id = self.current_scope;
        loop {
            if let Some(&id) = self.scopes[scope_id].bindings.get(name) {
                return Some(id);
            }
            match self.scopes[scope_id].parent {
                Some(parent) => scope_id = parent,
                None => return None,
            }
        }
    }

    pub fn get(&self, id: SymbolId) -> &Symbol {
        &self.symbols[id]
    }
}

pub struct NameResolver {
    symbol_table: SymbolTable,
    errors: Vec<ResolutionError>,
}

#[derive(Debug, Clone)]
pub struct ResolutionError {
    pub message: String,
    pub span: Span,
}

impl NameResolver {
    pub fn resolve(&mut self, program: &Program) -> Result<SymbolTable, Vec<ResolutionError>> {
        // Two-pass:
        // 1. Collect all top-level declarations
        // 2. Resolve references
        for decl in &program.declarations {
            self.collect_declaration(decl);
        }

        for decl in &program.declarations {
            self.resolve_declaration(decl);
        }

        if self.errors.is_empty() {
            Ok(self.symbol_table.clone())
        } else {
            Err(self.errors.clone())
        }
    }

    fn collect_declaration(&mut self, decl: &Declaration) {
        // Insert names into symbol table
        // ...
    }

    fn resolve_declaration(&mut self, decl: &Declaration) {
        // Resolve all references within declarations
        // ...
    }
}
```

---

### 3.3 Type & Shape Checker

```rust
// src/typeck.rs

use crate::ast::*;
use crate::resolve::SymbolTable;
use std::collections::HashMap;

pub type TypeId = usize;

#[derive(Debug, Clone)]
pub struct TypeContext {
    types: Vec<TensorType>,
    dim_constraints: ConstraintSet,
}

#[derive(Debug, Clone)]
pub struct TensorType {
    pub dtype: PrimitiveType,
    pub dims: Vec<DimValue>,
}

#[derive(Debug, Clone)]
pub enum DimValue {
    Const(i64),
    Symbolic(String),  // dimension variable
    Expr {
        op: BinaryOp,
        left: Box<DimValue>,
        right: Box<DimValue>,
    },
}

#[derive(Debug, Clone)]
pub struct ConstraintSet {
    constraints: Vec<DimConstraint>,
}

#[derive(Debug, Clone)]
pub enum DimConstraint {
    Equal(DimValue, DimValue),
    LessThan(DimValue, DimValue),
    Divisible(DimValue, i64),
    // ...
}

impl ConstraintSet {
    pub fn add(&mut self, constraint: DimConstraint) {
        self.constraints.push(constraint);
    }

    pub fn solve(&self) -> Result<Substitution, ConstraintError> {
        // Constraint solving algorithm
        // - Unification for equality constraints
        // - Arithmetic simplification
        // - SMT solver for complex constraints (optional)
        todo!()
    }
}

pub type Substitution = HashMap<String, DimValue>;

#[derive(Debug, Clone)]
pub struct ConstraintError {
    pub message: String,
    pub conflicting: Vec<DimConstraint>,
}

pub struct TypeChecker {
    symbol_table: SymbolTable,
    type_context: TypeContext,
    errors: Vec<TypeError>,
}

#[derive(Debug, Clone)]
pub struct TypeError {
    pub kind: TypeErrorKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum TypeErrorKind {
    Mismatch {
        expected: Type,
        found: Type,
    },
    ShapeMismatch {
        expected: Vec<DimValue>,
        found: Vec<DimValue>,
    },
    UndefinedVariable(String),
    InvalidOperation {
        op: String,
        operand_types: Vec<Type>,
    },
    ConstraintViolation(String),
}

impl TypeChecker {
    pub fn check(&mut self, program: &Program) -> Result<TypeContext, Vec<TypeError>> {
        for decl in &program.declarations {
            self.check_declaration(decl)?;
        }

        // Solve dimension constraints
        self.type_context.dim_constraints.solve()
            .map_err(|e| vec![TypeError {
                kind: TypeErrorKind::ConstraintViolation(e.message),
                span: Span { start: 0, end: 0, line: 0, column: 0 },
            }])?;

        if self.errors.is_empty() {
            Ok(self.type_context.clone())
        } else {
            Err(self.errors.clone())
        }
    }

    fn check_declaration(&mut self, decl: &Declaration) -> Result<(), TypeError> {
        match decl {
            Declaration::Function(f) => self.check_function(f),
            Declaration::Model(m) => self.check_model(m),
            // ...
            _ => Ok(()),
        }
    }

    fn check_function(&mut self, func: &FunctionDecl) -> Result<(), TypeError> {
        // Enter new scope
        // Check function body
        // Verify return type matches
        todo!()
    }

    fn infer_expr(&mut self, expr: &Expr) -> Result<Type, TypeError> {
        match expr {
            Expr::Literal(lit, span) => self.infer_literal(lit, *span),
            Expr::Ident(id) => self.infer_ident(id),
            Expr::Binary { op, left, right, span } => {
                self.infer_binary(*op, left, right, *span)
            }
            Expr::Call { func, args, span } => {
                self.infer_call(func, args, *span)
            }
            // ...
            _ => todo!(),
        }
    }

    fn infer_binary(&mut self, op: BinaryOp, left: &Expr, right: &Expr, span: Span)
        -> Result<Type, TypeError> {
        let left_ty = self.infer_expr(left)?;
        let right_ty = self.infer_expr(right)?;

        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                self.check_elementwise_compatible(&left_ty, &right_ty, span)
            }
            BinaryOp::MatMul => {
                self.check_matmul_compatible(&left_ty, &right_ty, span)
            }
            // ...
            _ => todo!(),
        }
    }

    fn check_elementwise_compatible(&mut self, left: &Type, right: &Type, span: Span)
        -> Result<Type, TypeError> {
        // Implement broadcasting rules
        // Generate shape constraints
        todo!()
    }

    fn check_matmul_compatible(&mut self, left: &Type, right: &Type, span: Span)
        -> Result<Type, TypeError> {
        // Extract tensor shapes
        // Check [..., M, K] @ [..., K, N] -> [..., M, N]
        // Add constraint: left.dims[-1] == right.dims[-2]
        todo!()
    }
}
```

---

### 3.4 High-Level IR (HIR)

```rust
// src/hir.rs

use crate::ast::{BinaryOp, UnaryOp, PrimitiveType};
use std::collections::HashMap;

pub type NodeId = usize;
pub type SymbolId = usize;

/// High-level IR: type-checked, desugared AST
#[derive(Debug, Clone)]
pub struct HirProgram {
    pub dims: HashMap<String, DimDef>,
    pub models: HashMap<String, ModelDef>,
    pub functions: HashMap<String, FunctionDef>,
    pub train_blocks: Vec<TrainBlock>,
}

#[derive(Debug, Clone)]
pub struct DimDef {
    pub name: String,
    pub value: Option<i64>,  // None = symbolic
}

#[derive(Debug, Clone)]
pub struct ModelDef {
    pub name: String,
    pub params: Vec<ParamDef>,
    pub buffers: Vec<BufferDef>,
    pub methods: Vec<FunctionDef>,
}

#[derive(Debug, Clone)]
pub struct ParamDef {
    pub name: String,
    pub ty: TensorType,
    pub init: InitOp,
}

#[derive(Debug, Clone)]
pub struct BufferDef {
    pub name: String,
    pub ty: TensorType,
}

#[derive(Debug, Clone)]
pub enum InitOp {
    Zeros,
    Ones,
    Normal { mean: f64, std: f64 },
    Uniform { low: f64, high: f64 },
    Xavier,
    Kaiming,
}

#[derive(Debug, Clone)]
pub struct TensorType {
    pub dtype: PrimitiveType,
    pub shape: Vec<DimExpr>,
}

#[derive(Debug, Clone)]
pub enum DimExpr {
    Const(i64),
    Var(String),
    Binary {
        op: BinaryOp,
        left: Box<DimExpr>,
        right: Box<DimExpr>,
    },
}

#[derive(Debug, Clone)]
pub struct FunctionDef {
    pub name: String,
    pub params: Vec<(String, TensorType)>,
    pub return_type: Option<TensorType>,
    pub body: Vec<HirStmt>,
}

#[derive(Debug, Clone)]
pub enum HirStmt {
    Let {
        var: String,
        ty: TensorType,
        value: HirExpr,
    },
    Assign {
        var: String,
        value: HirExpr,
    },
    Return(Option<HirExpr>),
    If {
        condition: HirExpr,
        then_block: Vec<HirStmt>,
        else_block: Option<Vec<HirStmt>>,
    },
    For {
        var: String,
        start: i64,
        end: i64,
        body: Vec<HirStmt>,
    },
}

#[derive(Debug, Clone)]
pub enum HirExpr {
    Var(String, TensorType),
    Const(ConstValue, TensorType),
    Binary {
        op: BinaryOp,
        left: Box<HirExpr>,
        right: Box<HirExpr>,
        ty: TensorType,
    },
    Unary {
        op: UnaryOp,
        operand: Box<HirExpr>,
        ty: TensorType,
    },
    Call {
        func: String,
        args: Vec<HirExpr>,
        ty: TensorType,
    },
    FieldAccess {
        object: Box<HirExpr>,
        field: String,
        ty: TensorType,
    },
}

#[derive(Debug, Clone)]
pub enum ConstValue {
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Debug, Clone)]
pub struct TrainBlock {
    pub model_instance: String,
    pub model_type: String,
    pub dataset: DatasetConfig,
    pub optimizer: OptimizerConfig,
    pub loop_body: Vec<HirStmt>,
}

#[derive(Debug, Clone)]
pub struct DatasetConfig {
    pub name: String,
    pub batch_size: i64,
    pub shuffle: bool,
}

#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub kind: OptimizerKind,
    pub lr: f64,
}

#[derive(Debug, Clone)]
pub enum OptimizerKind {
    SGD { momentum: f64, weight_decay: f64 },
    Adam { betas: (f64, f64), eps: f64, weight_decay: f64 },
}
```

---

### 3.5 Autograd Transform

```rust
// src/autograd.rs

use crate::hir::*;
use std::collections::HashMap;

pub struct AutogradEngine {
    forward_graph: ComputeGraph,
    backward_graph: ComputeGraph,
}

#[derive(Debug, Clone)]
pub struct ComputeGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<(NodeId, NodeId)>,
}

#[derive(Debug, Clone)]
pub enum GraphNode {
    Input { name: String, ty: TensorType },
    Param { name: String, ty: TensorType },
    Op { op: TensorOp, inputs: Vec<NodeId>, output_ty: TensorType },
    Gradient { of: NodeId, ty: TensorType },
}

#[derive(Debug, Clone)]
pub enum TensorOp {
    Add, Sub, Mul, Div,
    MatMul,
    Relu, Sigmoid, Tanh,
    Conv2d { stride: (i64, i64), padding: (i64, i64) },
    MaxPool2d { kernel: (i64, i64), stride: (i64, i64) },
    Reshape { shape: Vec<DimExpr> },
    Transpose { dim0: usize, dim1: usize },
    Sum { axis: Option<usize>, keepdim: bool },
    // ...
}

impl AutogradEngine {
    pub fn differentiate(&mut self, loss_node: NodeId) -> ComputeGraph {
        // Reverse-mode automatic differentiation
        // 1. Topological sort of forward graph
        // 2. For each node in reverse order, compute gradient
        // 3. Apply chain rule

        let topo_order = self.topological_sort(loss_node);
        let mut gradients: HashMap<NodeId, NodeId> = HashMap::new();

        // Seed gradient: d(loss)/d(loss) = 1
        let loss_grad = self.backward_graph.add_node(GraphNode::Op {
            op: TensorOp::Constant(1.0),
            inputs: vec![],
            output_ty: self.forward_graph.nodes[loss_node].output_type(),
        });
        gradients.insert(loss_node, loss_grad);

        for &node_id in topo_order.iter().rev() {
            let node = &self.forward_graph.nodes[node_id];
            if let GraphNode::Op { op, inputs, .. } = node {
                let grad_output = gradients[&node_id];

                // Compute gradients for each input
                for (i, &input_id) in inputs.iter().enumerate() {
                    let grad_input = self.vjp(op, inputs, i, grad_output);

                    // Accumulate gradient
                    *gradients.entry(input_id).or_insert(grad_input) =
                        self.backward_graph.add_node(GraphNode::Op {
                            op: TensorOp::Add,
                            inputs: vec![gradients[&input_id], grad_input],
                            output_ty: self.forward_graph.nodes[input_id].output_type(),
                        });
                }
            }
        }

        self.backward_graph.clone()
    }

    fn vjp(&mut self, op: &TensorOp, inputs: &[NodeId], input_idx: usize, grad_output: NodeId)
        -> NodeId {
        // Vector-Jacobian Product for each operation
        match op {
            TensorOp::Add => grad_output,  // d(x + y)/dx = 1
            TensorOp::Mul => {
                // d(x * y)/dx = y
                let other_idx = 1 - input_idx;
                self.backward_graph.add_node(GraphNode::Op {
                    op: TensorOp::Mul,
                    inputs: vec![grad_output, inputs[other_idx]],
                    output_ty: self.forward_graph.nodes[inputs[input_idx]].output_type(),
                })
            }
            TensorOp::MatMul => {
                // d(A @ B)/dA = grad_out @ B^T
                // d(A @ B)/dB = A^T @ grad_out
                if input_idx == 0 {
                    let b_t = self.backward_graph.add_node(GraphNode::Op {
                        op: TensorOp::Transpose { dim0: 0, dim1: 1 },
                        inputs: vec![inputs[1]],
                        output_ty: todo!(),
                    });
                    self.backward_graph.add_node(GraphNode::Op {
                        op: TensorOp::MatMul,
                        inputs: vec![grad_output, b_t],
                        output_ty: todo!(),
                    })
                } else {
                    let a_t = self.backward_graph.add_node(GraphNode::Op {
                        op: TensorOp::Transpose { dim0: 0, dim1: 1 },
                        inputs: vec![inputs[0]],
                        output_ty: todo!(),
                    });
                    self.backward_graph.add_node(GraphNode::Op {
                        op: TensorOp::MatMul,
                        inputs: vec![a_t, grad_output],
                        output_ty: todo!(),
                    })
                }
            }
            TensorOp::Relu => {
                // d(relu(x))/dx = (x > 0) * grad_out
                let mask = self.backward_graph.add_node(GraphNode::Op {
                    op: TensorOp::Greater,
                    inputs: vec![inputs[0], /* zero */],
                    output_ty: todo!(),
                });
                self.backward_graph.add_node(GraphNode::Op {
                    op: TensorOp::Mul,
                    inputs: vec![mask, grad_output],
                    output_ty: todo!(),
                })
            }
            // ... other operations
            _ => todo!(),
        }
    }

    fn topological_sort(&self, start: NodeId) -> Vec<NodeId> {
        // Standard DFS-based topological sort
        todo!()
    }
}
```

---

### 3.6 Mid-Level IR (MIR)

```rust
// src/mir.rs

/// MIR: Flattened computation graph with control flow
#[derive(Debug, Clone)]
pub struct MirProgram {
    pub functions: Vec<MirFunction>,
}

#[derive(Debug, Clone)]
pub struct MirFunction {
    pub name: String,
    pub params: Vec<(String, TensorType)>,
    pub return_type: Option<TensorType>,
    pub basic_blocks: Vec<BasicBlock>,
}

pub type BlockId = usize;

#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: BlockId,
    pub instructions: Vec<Instruction>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Alloc {
        dest: VarId,
        ty: TensorType,
    },
    Load {
        dest: VarId,
        src: VarId,
    },
    Store {
        dest: VarId,
        src: VarId,
    },
    TensorOp {
        dest: VarId,
        op: TensorOp,
        args: Vec<VarId>,
    },
}

pub type VarId = usize;

#[derive(Debug, Clone)]
pub enum Terminator {
    Return(Option<VarId>),
    Jump(BlockId),
    Branch {
        condition: VarId,
        true_block: BlockId,
        false_block: BlockId,
    },
}
```

---

### 3.7 Optimization Passes

```rust
// src/optimize.rs

pub trait OptimizationPass {
    fn run(&self, mir: &mut MirProgram) -> bool;  // returns true if changed
}

/// Dead Code Elimination
pub struct DeadCodeElimination;

impl OptimizationPass for DeadCodeElimination {
    fn run(&self, mir: &mut MirProgram) -> bool {
        // Remove unused variables and instructions
        todo!()
    }
}

/// Common Subexpression Elimination
pub struct CommonSubexpressionElimination;

impl OptimizationPass for CommonSubexpressionElimination {
    fn run(&self, mir: &mut MirProgram) -> bool {
        // Detect and reuse common computations
        todo!()
    }
}

/// Operator Fusion
pub struct OperatorFusion;

impl OptimizationPass for OperatorFusion {
    fn run(&self, mir: &mut MirProgram) -> bool {
        // Fuse elementwise operations, matmul+bias+activation, etc.
        // Examples:
        // - (A + B) + C => A + B + C (single kernel)
        // - matmul(x, w) + b => fused_linear(x, w, b)
        // - relu(matmul(x, w) + b) => fused_linear_relu(x, w, b)
        todo!()
    }
}

/// Constant Folding
pub struct ConstantFolding;

impl OptimizationPass for ConstantFolding {
    fn run(&self, mir: &mut MirProgram) -> bool {
        // Evaluate constant expressions at compile time
        todo!()
    }
}

pub struct OptimizationPipeline {
    passes: Vec<Box<dyn OptimizationPass>>,
}

impl OptimizationPipeline {
    pub fn default() -> Self {
        Self {
            passes: vec![
                Box::new(ConstantFolding),
                Box::new(CommonSubexpressionElimination),
                Box::new(DeadCodeElimination),
                Box::new(OperatorFusion),
            ],
        }
    }

    pub fn run(&self, mir: &mut MirProgram) {
        let mut changed = true;
        while changed {
            changed = false;
            for pass in &self.passes {
                changed |= pass.run(mir);
            }
        }
    }
}
```

---

### 3.8 Code Generation Backend

#### 3.8.1 Backend Choices

**Option 1: LLVM IR Backend** (Recommended for production)
- Pros: Excellent optimization, mature toolchain, cross-platform
- Cons: Complex, requires external tensor library

**Option 2: C++ Backend with Eigen/BLAS**
- Pros: Simple, portable, good CPU performance
- Cons: Manual optimization needed

**Option 3: CUDA Backend**
- Pros: Direct GPU control, maximum performance
- Cons: NVIDIA-only, complex memory management

**Option 4: PyTorch/JAX Interop Backend** (Recommended for prototyping)
- Pros: Fastest to implement, leverages existing optimizations
- Cons: Not truly standalone, requires Python runtime

**Recommendation for Phase 1**: **PyTorch Interop Backend** for rapid validation, then migrate to **LLVM + Custom Runtime**

#### 3.8.2 PyTorch Interop Backend

```rust
// src/codegen/pytorch.rs

pub struct PyTorchCodegen {
    output: String,
}

impl PyTorchCodegen {
    pub fn generate(&mut self, mir: &MirProgram) -> String {
        self.output.push_str("import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\n");

        for func in &mir.functions {
            self.generate_function(func);
        }

        self.output.clone()
    }

    fn generate_function(&mut self, func: &MirFunction) {
        self.output.push_str(&format!("def {}(", func.name));

        for (i, (name, ty)) in func.params.iter().enumerate() {
            if i > 0 { self.output.push_str(", "); }
            self.output.push_str(name);
        }
        self.output.push_str("):\n");

        for bb in &func.basic_blocks {
            for instr in &bb.instructions {
                self.generate_instruction(instr);
            }
            self.generate_terminator(&bb.terminator);
        }

        self.output.push_str("\n");
    }

    fn generate_instruction(&mut self, instr: &Instruction) {
        match instr {
            Instruction::TensorOp { dest, op, args } => {
                self.output.push_str(&format!("    v{} = ", dest));
                match op {
                    TensorOp::Add => {
                        self.output.push_str(&format!("v{} + v{}", args[0], args[1]));
                    }
                    TensorOp::MatMul => {
                        self.output.push_str(&format!("torch.matmul(v{}, v{})", args[0], args[1]));
                    }
                    TensorOp::Relu => {
                        self.output.push_str(&format!("torch.relu(v{})", args[0]));
                    }
                    // ... other ops
                    _ => todo!(),
                }
                self.output.push_str("\n");
            }
            _ => todo!(),
        }
    }

    fn generate_terminator(&mut self, term: &Terminator) {
        match term {
            Terminator::Return(Some(var)) => {
                self.output.push_str(&format!("    return v{}\n", var));
            }
            Terminator::Return(None) => {
                self.output.push_str("    return\n");
            }
            _ => todo!(),
        }
    }
}
```

---

## 4. Project Structure

```
lumen/
├── Cargo.toml
├── README.md
├── docs/
│   ├── LANGUAGE_SPEC.md
│   └── COMPILER_ARCHITECTURE.md
├── src/
│   ├── main.rs              # CLI entry point
│   ├── lib.rs               # Library root
│   ├── ast.rs               # AST definitions
│   ├── parser/
│   │   ├── mod.rs
│   │   ├── lexer.rs         # Tokenization
│   │   └── grammar.pest     # pest grammar
│   ├── resolve.rs           # Name resolution & symbol table
│   ├── typeck.rs            # Type & shape checking
│   ├── hir.rs               # High-level IR
│   ├── mir.rs               # Mid-level IR
│   ├── autograd.rs          # Automatic differentiation
│   ├── optimize/
│   │   ├── mod.rs
│   │   ├── fusion.rs
│   │   ├── cse.rs
│   │   └── dce.rs
│   ├── codegen/
│   │   ├── mod.rs
│   │   ├── pytorch.rs       # PyTorch backend
│   │   └── llvm.rs          # LLVM backend (future)
│   └── runtime/
│       ├── mod.rs
│       └── tensor.rs        # Runtime tensor ops (future)
├── tests/
│   ├── parser_tests.rs
│   ├── typeck_tests.rs
│   └── integration_tests.rs
└── examples/
    ├── mlp.lumen
    ├── cnn.lumen
    └── transformer.lumen
```

---

## 5. Build System

```toml
# Cargo.toml

[package]
name = "lumen"
version = "0.1.0"
edition = "2021"

[dependencies]
pest = "2.7"
pest_derive = "2.7"
thiserror = "1.0"
anyhow = "1.0"
clap = { version = "4.0", features = ["derive"] }
inkwell = { version = "0.4", optional = true }  # LLVM bindings

[dev-dependencies]
criterion = "0.5"

[[bin]]
name = "lumenc"
path = "src/main.rs"

[lib]
name = "lumen"
path = "src/lib.rs"

[features]
default = []
llvm-backend = ["inkwell"]
```

---

## 6. Testing Strategy

1. **Unit Tests**: Each module (parser, typeck, etc.) has dedicated tests
2. **Integration Tests**: End-to-end compilation of example programs
3. **Snapshot Tests**: Compare generated code against expected output
4. **Performance Benchmarks**: Compare against PyTorch on standard models

---

**Next Steps**: Proceed to Phase 3 - Minimal Working Prototype
