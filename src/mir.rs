// Mid-Level Intermediate Representation (Simplified)

use crate::hir::*;
use crate::error::Result;

/// MIR is a simplified computation graph representation
/// For now, we keep it close to HIR and focus on code generation
#[derive(Debug, Clone)]
pub struct MirProgram { models: vec![], 
    pub models: Vec<MirModel>,
    pub functions: Vec<MirFunction>,
}

#[derive(Debug, Clone)]
pub struct MirModel {
    pub name: String,
    pub params: Vec<MirParam>,
    pub methods: Vec<MirFunction>,
}

#[derive(Debug, Clone)]
pub struct MirParam {
    pub name: String,
    pub ty: HirType,
    pub init: HirInit,
}

#[derive(Debug, Clone)]
pub struct MirFunction {
    pub name: String,
    pub params: Vec<MirFunctionParam>,
    pub return_type: Option<HirType>,
    pub body: Vec<MirStmt>,
}

#[derive(Debug, Clone)]
pub struct MirFunctionParam {
    pub var_id: VarId,
    pub name: String,
    pub ty: HirType,
}

#[derive(Debug, Clone)]
pub enum MirStmt {
    Assign {
        var_id: VarId,
        name: String,
        value: MirExpr,
    },
    Return {
        value: Option<MirExpr>,
    },
}

#[derive(Debug, Clone)]
pub enum MirExpr {
    Var {
        var_id: VarId,
        name: String,
    },
    Literal(HirLiteral),
    Call {
        func: String,
        args: Vec<MirExpr>,
    },
    BinOp {
        op: HirBinOp,
        left: Box<MirExpr>,
        right: Box<MirExpr>,
    },
    UnaryOp {
        op: HirUnaryOp,
        operand: Box<MirExpr>,
    },
    FieldAccess {
        object: Box<MirExpr>,
        field: String,
    },
}

pub struct MirLowering;

impl MirLowering {
    pub fn new() -> Self {
        Self
    }

    pub fn lower(&self, hir: &HirProgram) -> Result<MirProgram> {
        let mut models = Vec::new();
        let mut functions = Vec::new();

        for model in &hir.models {
            models.push(self.lower_model(model)?);
        }

        for func in &hir.functions {
            functions.push(self.lower_function(func)?);
        }

        Ok(MirProgram { models, functions })
    }

    fn lower_model(&self, model: &HirModel) -> Result<MirModel> {
        let params = model.params.iter()
            .map(|p| MirParam {
                name: p.name.clone(),
                ty: p.ty.clone(),
                init: p.init.clone(),
            })
            .collect();

        let methods = model.methods.iter()
            .map(|m| self.lower_function(m))
            .collect::<Result<Vec<_>>>()?;

        Ok(MirModel {
            name: model.name.clone(),
            params,
            methods,
        })
    }

    fn lower_function(&self, func: &HirFunction) -> Result<MirFunction> {
        let params = func.params.iter()
            .map(|p| MirFunctionParam {
                var_id: p.var_id,
                name: p.name.clone(),
                ty: p.ty.clone(),
            })
            .collect();

        let body = func.body.iter()
            .map(|s| self.lower_stmt(s))
            .collect::<Result<Vec<_>>>()?;

        Ok(MirFunction {
            name: func.name.clone(),
            params,
            return_type: func.return_type.clone(),
            body,
        })
    }

    fn lower_stmt(&self, stmt: &HirStmt) -> Result<MirStmt> {
        match stmt {
            HirStmt::Assign { var_id, name, value, .. } => {
                Ok(MirStmt::Assign {
                    var_id: *var_id,
                    name: name.clone(),
                    value: self.lower_expr(value)?,
                })
            }
            HirStmt::Return { value } => {
                Ok(MirStmt::Return {
                    value: value.as_ref().map(|e| self.lower_expr(e)).transpose()?,
                })
            }
        }
    }

    fn lower_expr(&self, expr: &HirExpr) -> Result<MirExpr> {
        match expr {
            HirExpr::Var { var_id, name, .. } => {
                Ok(MirExpr::Var {
                    var_id: *var_id,
                    name: name.clone(),
                })
            }
            HirExpr::Literal { value, .. } => {
                Ok(MirExpr::Literal(value.clone()))
            }
            HirExpr::Call { func, args, .. } => {
                let args_mir = args.iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<Result<Vec<_>>>()?;

                Ok(MirExpr::Call {
                    func: func.clone(),
                    args: args_mir,
                })
            }
            HirExpr::BinOp { op, left, right, .. } => {
                Ok(MirExpr::BinOp {
                    op: *op,
                    left: Box::new(self.lower_expr(left)?),
                    right: Box::new(self.lower_expr(right)?),
                })
            }
            HirExpr::UnaryOp { op, operand, .. } => {
                Ok(MirExpr::UnaryOp {
                    op: *op,
                    operand: Box::new(self.lower_expr(operand)?),
                })
            }
            HirExpr::FieldAccess { object, field, .. } => {
                Ok(MirExpr::FieldAccess {
                    object: Box::new(self.lower_expr(object)?),
                    field: field.clone(),
                })
            }
        }
    }
}
