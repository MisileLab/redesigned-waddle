// High-Level Intermediate Representation

use crate::ast::*;
use crate::typeck::TypeContext;
use crate::error::{LumenError, Result};
use std::collections::HashMap;

pub type VarId = usize;

/// High-level IR: typed, desugared AST
#[derive(Debug, Clone)]
pub struct HirProgram {
    pub dims: Vec<HirDim>,
    pub models: Vec<HirModel>,
    pub functions: Vec<HirFunction>,
}

#[derive(Debug, Clone)]
pub struct HirDim {
    pub name: String,
    pub value: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct HirModel {
    pub name: String,
    pub params: Vec<HirParam>,
    pub methods: Vec<HirFunction>,
}

#[derive(Debug, Clone)]
pub struct HirParam {
    pub name: String,
    pub ty: HirType,
    pub init: HirInit,
}

#[derive(Debug, Clone)]
pub enum HirInit {
    Zeros,
    Ones,
    Normal { mean: f64, std: f64 },
    Uniform { low: f64, high: f64 },
    Xavier,
    Kaiming,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HirType {
    I32,
    I64,
    F32,
    F64,
    Bool,
    Tensor {
        dtype: Box<HirType>,
        shape: Vec<HirDimExpr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum HirDimExpr {
    Const(i64),
    Symbolic(String),
}

#[derive(Debug, Clone)]
pub struct HirFunction {
    pub name: String,
    pub params: Vec<HirFunctionParam>,
    pub return_type: Option<HirType>,
    pub body: Vec<HirStmt>,
    pub next_var_id: VarId,
}

#[derive(Debug, Clone)]
pub struct HirFunctionParam {
    pub var_id: VarId,
    pub name: String,
    pub ty: HirType,
}

#[derive(Debug, Clone)]
pub enum HirStmt {
    Assign {
        var_id: VarId,
        name: String,
        ty: HirType,
        value: HirExpr,
    },
    Return {
        value: Option<HirExpr>,
    },
}

#[derive(Debug, Clone)]
pub enum HirExpr {
    Var {
        var_id: VarId,
        name: String,
        ty: HirType,
    },
    Literal {
        value: HirLiteral,
        ty: HirType,
    },
    Call {
        func: String,
        args: Vec<HirExpr>,
        ty: HirType,
    },
    BinOp {
        op: HirBinOp,
        left: Box<HirExpr>,
        right: Box<HirExpr>,
        ty: HirType,
    },
    UnaryOp {
        op: HirUnaryOp,
        operand: Box<HirExpr>,
        ty: HirType,
    },
    FieldAccess {
        object: Box<HirExpr>,
        field: String,
        ty: HirType,
    },
}

impl HirExpr {
    pub fn ty(&self) -> &HirType {
        match self {
            HirExpr::Var { ty, .. } => ty,
            HirExpr::Literal { ty, .. } => ty,
            HirExpr::Call { ty, .. } => ty,
            HirExpr::BinOp { ty, .. } => ty,
            HirExpr::UnaryOp { ty, .. } => ty,
            HirExpr::FieldAccess { ty, .. } => ty,
        }
    }
}

#[derive(Debug, Clone)]
pub enum HirLiteral {
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Debug, Clone, Copy)]
pub enum HirBinOp {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
}

#[derive(Debug, Clone, Copy)]
pub enum HirUnaryOp {
    Neg,
    Not,
}

pub struct HirLowering {
    type_context: TypeContext,
    var_counter: VarId,
    var_map: HashMap<String, VarId>,
}

impl HirLowering {
    pub fn new(type_context: TypeContext) -> Self {
        Self {
            type_context,
            var_counter: 0,
            var_map: HashMap::new(),
        }
    }

    pub fn lower(&mut self, program: &Program) -> Result<HirProgram> {
        let mut dims = Vec::new();
        let mut models = Vec::new();
        let mut functions = Vec::new();

        for decl in &program.declarations {
            match decl {
                Declaration::Dim(dim) => {
                    dims.push(HirDim {
                        name: dim.name.clone(),
                        value: dim.value,
                    });
                }
                Declaration::Model(model) => {
                    models.push(self.lower_model(model)?);
                }
                Declaration::Function(func) => {
                    functions.push(self.lower_function(func)?);
                }
            }
        }

        Ok(HirProgram {
            dims,
            models,
            functions,
        })
    }

    fn lower_model(&mut self, model: &ModelDecl) -> Result<HirModel> {
        let mut params = Vec::new();
        let mut methods = Vec::new();

        for member in &model.members {
            match member {
                ModelMember::Param(param) => {
                    params.push(HirParam {
                        name: param.name.clone(),
                        ty: self.lower_type(&param.ty)?,
                        init: self.lower_init(&param.initializer),
                    });
                }
                ModelMember::Function(func) => {
                    methods.push(self.lower_function(func)?);
                }
            }
        }

        Ok(HirModel {
            name: model.name.clone(),
            params,
            methods,
        })
    }

    fn lower_function(&mut self, func: &FunctionDecl) -> Result<HirFunction> {
        self.var_counter = 0;
        self.var_map.clear();

        let mut params = Vec::new();

        for param in &func.params {
            let var_id = self.fresh_var();
            self.var_map.insert(param.name.clone(), var_id);

            params.push(HirFunctionParam {
                var_id,
                name: param.name.clone(),
                ty: self.lower_type(&param.ty)?,
            });
        }

        let body = self.lower_block(&func.body)?;

        Ok(HirFunction {
            name: func.name.clone(),
            params,
            return_type: func.return_type.as_ref().map(|t| self.lower_type(t)).transpose()?,
            body,
            next_var_id: self.var_counter,
        })
    }

    fn lower_block(&mut self, block: &Block) -> Result<Vec<HirStmt>> {
        let mut stmts = Vec::new();

        for stmt in &block.stmts {
            stmts.extend(self.lower_stmt(stmt)?);
        }

        Ok(stmts)
    }

    fn lower_stmt(&mut self, stmt: &Stmt) -> Result<Vec<HirStmt>> {
        match stmt {
            Stmt::Let { name, ty, value, .. } => {
                let var_id = self.fresh_var();
                self.var_map.insert(name.clone(), var_id);

                let value_hir = self.lower_expr(value)?;
                let stmt_ty = if let Some(ty) = ty {
                    self.lower_type(ty)?
                } else {
                    value_hir.ty().clone()
                };

                Ok(vec![HirStmt::Assign {
                    var_id,
                    name: name.clone(),
                    ty: stmt_ty,
                    value: value_hir,
                }])
            }
            Stmt::Return { value, .. } => {
                Ok(vec![HirStmt::Return {
                    value: value.as_ref().map(|e| self.lower_expr(e)).transpose()?,
                }])
            }
            Stmt::Expr(_) => Ok(vec![]), // Ignore expression statements for now
        }
    }

    fn lower_expr(&mut self, expr: &Expr) -> Result<HirExpr> {
        match expr {
            Expr::Literal(lit, _) => Ok(HirExpr::Literal {
                value: match lit {
                    Literal::Int(i) => HirLiteral::Int(*i),
                    Literal::Float(f) => HirLiteral::Float(*f),
                    Literal::Bool(b) => HirLiteral::Bool(*b),
                },
                ty: match lit {
                    Literal::Int(_) => HirType::I64,
                    Literal::Float(_) => HirType::F64,
                    Literal::Bool(_) => HirType::Bool,
                },
            }),
            Expr::Ident(name, _) => {
                let var_id = self.var_map.get(name).copied().ok_or_else(|| {
                    LumenError::CodegenError {
                        message: format!("Variable '{}' not found in HIR lowering", name),
                    }
                })?;

                let ty = self.type_context.type_map.get(name)
                    .ok_or_else(|| LumenError::CodegenError {
                        message: format!("Type for '{}' not found", name),
                    })?;

                Ok(HirExpr::Var {
                    var_id,
                    name: name.clone(),
                    ty: self.lower_type(ty)?,
                })
            }
            Expr::Call { func, args, .. } => {
                let func_name = match func.as_ref() {
                    Expr::Ident(name, _) => name.clone(),
                    Expr::FieldAccess { field, .. } => field.clone(),
                    _ => return Err(LumenError::CodegenError {
                        message: "Invalid function call".to_string(),
                    }),
                };

                let args_hir: Result<Vec<_>> = args.iter()
                    .map(|arg| self.lower_expr(arg))
                    .collect();

                // Infer result type based on function
                let result_ty = self.infer_call_result_type(&func_name, &args_hir.as_ref().unwrap())?;

                Ok(HirExpr::Call {
                    func: func_name,
                    args: args_hir?,
                    ty: result_ty,
                })
            }
            Expr::Binary { op, left, right, .. } => {
                let left_hir = self.lower_expr(left)?;
                let right_hir = self.lower_expr(right)?;

                let result_ty = self.infer_binary_result_type(*op, &left_hir, &right_hir)?;

                Ok(HirExpr::BinOp {
                    op: match op {
                        BinaryOp::Add => HirBinOp::Add,
                        BinaryOp::Sub => HirBinOp::Sub,
                        BinaryOp::Mul => HirBinOp::Mul,
                        BinaryOp::Div => HirBinOp::Div,
                        BinaryOp::MatMul => HirBinOp::MatMul,
                        _ => return Err(LumenError::CodegenError {
                            message: format!("Unsupported binary op: {:?}", op),
                        }),
                    },
                    left: Box::new(left_hir),
                    right: Box::new(right_hir),
                    ty: result_ty,
                })
            }
            Expr::Unary { op, operand, .. } => {
                let operand_hir = self.lower_expr(operand)?;
                let result_ty = operand_hir.ty().clone();

                Ok(HirExpr::UnaryOp {
                    op: match op {
                        UnaryOp::Neg => HirUnaryOp::Neg,
                        UnaryOp::Not => HirUnaryOp::Not,
                    },
                    operand: Box::new(operand_hir),
                    ty: result_ty,
                })
            }
            Expr::FieldAccess { object, field, .. } => {
                let object_hir = self.lower_expr(object)?;

                // Simplified: assume field type is known
                let field_ty = HirType::F32; // Placeholder

                Ok(HirExpr::FieldAccess {
                    object: Box::new(object_hir),
                    field: field.clone(),
                    ty: field_ty,
                })
            }
        }
    }

    fn lower_type(&self, ty: &Type) -> Result<HirType> {
        match ty {
            Type::Primitive(p) => Ok(match p {
                PrimitiveType::I32 => HirType::I32,
                PrimitiveType::I64 => HirType::I64,
                PrimitiveType::F32 => HirType::F32,
                PrimitiveType::F64 => HirType::F64,
                PrimitiveType::Bool => HirType::Bool,
            }),
            Type::Tensor { dtype, dims } => {
                let hir_dtype = Box::new(self.lower_type(dtype)?);
                let hir_shape: Result<Vec<_>> = dims.iter()
                    .map(|d| self.lower_dim_expr(d))
                    .collect();

                Ok(HirType::Tensor {
                    dtype: hir_dtype,
                    shape: hir_shape?,
                })
            }
            Type::Named(_) => {
                // Placeholder for model types
                Ok(HirType::F32)
            }
        }
    }

    fn lower_dim_expr(&self, dim: &DimExpr) -> Result<HirDimExpr> {
        match dim {
            DimExpr::Const(c) => Ok(HirDimExpr::Const(*c)),
            DimExpr::Var(v) => {
                // Check if we can resolve to constant
                if let Some(value) = self.type_context.get_dim_value(v) {
                    Ok(HirDimExpr::Const(value))
                } else {
                    Ok(HirDimExpr::Symbolic(v.clone()))
                }
            }
            DimExpr::Binary { .. } => {
                // Simplified: treat as symbolic for now
                Ok(HirDimExpr::Symbolic("expr".to_string()))
            }
        }
    }

    fn lower_init(&self, init: &Initializer) -> HirInit {
        match init {
            Initializer::Zeros => HirInit::Zeros,
            Initializer::Ones => HirInit::Ones,
            Initializer::Normal { mean, std } => HirInit::Normal { mean: *mean, std: *std },
            Initializer::Uniform { low, high } => HirInit::Uniform { low: *low, high: *high },
            Initializer::Xavier => HirInit::Xavier,
            Initializer::Kaiming => HirInit::Kaiming,
        }
    }

    fn infer_call_result_type(&self, func: &str, args: &[HirExpr]) -> Result<HirType> {
        match func {
            "matmul" => {
                if args.len() != 2 {
                    return Err(LumenError::CodegenError {
                        message: "matmul requires 2 arguments".to_string(),
                    });
                }

                if let HirType::Tensor { dtype, shape } = args[0].ty() {
                    if shape.len() >= 2 {
                        let mut result_shape = shape[..shape.len() - 1].to_vec();
                        if let HirType::Tensor { shape: shape_b, .. } = args[1].ty() {
                            if shape_b.len() >= 2 {
                                result_shape.push(shape_b[shape_b.len() - 1].clone());
                            }
                        }

                        return Ok(HirType::Tensor {
                            dtype: dtype.clone(),
                            shape: result_shape,
                        });
                    }
                }

                Err(LumenError::CodegenError {
                    message: "matmul type inference failed".to_string(),
                })
            }
            "relu" | "sigmoid" | "tanh" => {
                if args.is_empty() {
                    return Err(LumenError::CodegenError {
                        message: format!("{} requires 1 argument", func),
                    });
                }
                Ok(args[0].ty().clone())
            }
            "cross_entropy" => Ok(HirType::F32),
            _ => Err(LumenError::CodegenError {
                message: format!("Unknown function '{}'", func),
            }),
        }
    }

    fn infer_binary_result_type(&self, op: BinaryOp, left: &HirExpr, right: &HirExpr) -> Result<HirType> {
        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                Ok(left.ty().clone())
            }
            BinaryOp::MatMul => {
                self.infer_call_result_type("matmul", &[left.clone(), right.clone()])
            }
            _ => Err(LumenError::CodegenError {
                message: format!("Unsupported binary op: {:?}", op),
            }),
        }
    }

    fn fresh_var(&mut self) -> VarId {
        let id = self.var_counter;
        self.var_counter += 1;
        id
    }
}
