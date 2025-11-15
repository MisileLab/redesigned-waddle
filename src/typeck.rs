// Type checking and shape inference

use std::collections::HashMap;
use crate::ast::*;
use crate::resolve::{SymbolTable, Symbol};
use crate::error::{LumenError, Result};

#[derive(Debug, Clone)]
pub struct TypeContext {
    pub symbol_table: SymbolTable,
    pub type_map: HashMap<String, Type>,
    pub dim_values: HashMap<String, Option<i64>>,
}

impl TypeContext {
    pub fn new(symbol_table: SymbolTable) -> Self {
        let mut dim_values = HashMap::new();

        // Extract dimension values from symbol table
        for symbol in &symbol_table.symbols {
            if let Symbol::Dim { name, value } = symbol {
                dim_values.insert(name.clone(), *value);
            }
        }

        Self {
            symbol_table,
            type_map: HashMap::new(),
            dim_values,
        }
    }

    pub fn get_dim_value(&self, name: &str) -> Option<i64> {
        self.dim_values.get(name).and_then(|v| *v)
    }
}

pub struct TypeChecker {
    context: TypeContext,
}

impl TypeChecker {
    pub fn new(symbol_table: SymbolTable) -> Self {
        Self {
            context: TypeContext::new(symbol_table),
        }
    }

    pub fn check(&mut self, program: &Program) -> Result<TypeContext> {
        for decl in &program.declarations {
            self.check_declaration(decl)?;
        }
        Ok(self.context.clone())
    }

    fn check_declaration(&mut self, decl: &Declaration) -> Result<()> {
        match decl {
            Declaration::Dim(_) => Ok(()),
            Declaration::Model(model) => self.check_model(model),
            Declaration::Function(func) => self.check_function(func),
        }
    }

    fn check_model(&mut self, model: &ModelDecl) -> Result<()> {
        for member in &model.members {
            match member {
                ModelMember::Param(param) => {
                    self.check_type(&param.ty)?;
                }
                ModelMember::Function(func) => {
                    self.check_function(func)?;
                }
            }
        }
        Ok(())
    }

    fn check_function(&mut self, func: &FunctionDecl) -> Result<()> {
        // Check parameter types
        for param in &func.params {
            self.check_type(&param.ty)?;
            self.context.type_map.insert(param.name.clone(), param.ty.clone());
        }

        // Check return type
        if let Some(ref ret_ty) = func.return_type {
            self.check_type(ret_ty)?;
        }

        // Check body
        let body_type = self.check_block(&func.body)?;

        // Verify return type matches
        if let Some(ref expected) = func.return_type {
            if let Some(actual) = body_type {
                self.unify_types(expected, &actual)?;
            }
        }

        Ok(())
    }

    fn check_block(&mut self, block: &Block) -> Result<Option<Type>> {
        let mut last_type = None;

        for stmt in &block.stmts {
            last_type = self.check_stmt(stmt)?;
        }

        Ok(last_type)
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> Result<Option<Type>> {
        match stmt {
            Stmt::Let { name, ty, value, .. } => {
                let value_type = self.infer_expr(value)?;

                if let Some(ref expected_ty) = ty {
                    self.check_type(expected_ty)?;
                    self.unify_types(expected_ty, &value_type)?;
                }

                self.context.type_map.insert(name.clone(), value_type);
                Ok(None)
            }
            Stmt::Return { value, .. } => {
                if let Some(ref expr) = value {
                    Ok(Some(self.infer_expr(expr)?))
                } else {
                    Ok(None)
                }
            }
            Stmt::Expr(expr) => {
                self.infer_expr(expr)?;
                Ok(None)
            }
        }
    }

    fn infer_expr(&mut self, expr: &Expr) -> Result<Type> {
        match expr {
            Expr::Literal(lit, _) => Ok(self.infer_literal(lit)),
            Expr::Ident(name, span) => {
                self.context.type_map.get(name).cloned().ok_or_else(|| {
                    LumenError::TypeError {
                        message: format!("Variable '{}' not found at {:?}", name, span),
                    }
                })
            }
            Expr::FieldAccess { object, field, span } => {
                let obj_type = self.infer_expr(object)?;
                self.infer_field_access(&obj_type, field, *span)
            }
            Expr::Call { func, args, span } => {
                self.infer_call(func, args, *span)
            }
            Expr::Binary { op, left, right, span } => {
                self.infer_binary(*op, left, right, *span)
            }
            Expr::Unary { op, operand, span } => {
                self.infer_unary(*op, operand, *span)
            }
        }
    }

    fn infer_literal(&self, lit: &Literal) -> Type {
        match lit {
            Literal::Int(_) => Type::Primitive(PrimitiveType::I64),
            Literal::Float(_) => Type::Primitive(PrimitiveType::F64),
            Literal::Bool(_) => Type::Primitive(PrimitiveType::Bool),
        }
    }

    fn infer_field_access(&mut self, obj_type: &Type, field: &str, _span: Span) -> Result<Type> {
        // For model instances, look up field in model definition
        if let Type::Named(model_name) = obj_type {
            if let Some(symbol_id) = self.context.symbol_table.lookup(model_name) {
                if let Symbol::Model { methods, params, .. } = self.context.symbol_table.get(symbol_id) {
                    // Check if field is a method
                    for &method_id in methods {
                        if let Symbol::Function { name, params, return_type, .. } =
                            self.context.symbol_table.get(method_id) {
                            if name == field {
                                // Return function type (simplified)
                                return return_type.clone().ok_or_else(|| LumenError::TypeError {
                                    message: format!("Method '{}' has no return type", field),
                                });
                            }
                        }
                    }

                    // Check if field is a param
                    for &param_id in params {
                        if let Symbol::Param { name, ty, .. } = self.context.symbol_table.get(param_id) {
                            if name == field {
                                return Ok(ty.clone());
                            }
                        }
                    }
                }
            }
        }

        Err(LumenError::TypeError {
            message: format!("Field '{}' not found on type {:?}", field, obj_type),
        })
    }

    fn infer_call(&mut self, func: &Expr, args: &[Expr], span: Span) -> Result<Type> {
        // Get function name
        let func_name = match func {
            Expr::Ident(name, _) => name.clone(),
            Expr::FieldAccess { field, .. } => field.clone(),
            _ => {
                return Err(LumenError::TypeError {
                    message: format!("Invalid function call at {:?}", span),
                });
            }
        };

        // Check built-in functions
        if let Some(ret_type) = self.check_builtin_function(&func_name, args)? {
            return Ok(ret_type);
        }

        // Check user-defined functions
        if let Some(symbol_id) = self.context.symbol_table.lookup(&func_name) {
            if let Symbol::Function { params, return_type, .. } =
                self.context.symbol_table.get(symbol_id) {

                // Clone function info to avoid borrow issues
                let params_clone = params.clone();
                let return_type_clone = return_type.clone();

                // Check argument count
                if args.len() != params_clone.len() {
                    return Err(LumenError::TypeError {
                        message: format!(
                            "Function '{}' expects {} arguments, got {}",
                            func_name,
                            params_clone.len(),
                            args.len()
                        ),
                    });
                }

                // Type check arguments
                for (arg, param) in args.iter().zip(params_clone.iter()) {
                    let arg_type = self.infer_expr(arg)?;
                    self.unify_types(&param.ty, &arg_type)?;
                }

                return return_type_clone.ok_or_else(|| LumenError::TypeError {
                    message: format!("Function '{}' has no return type", func_name),
                });
            }
        }

        Err(LumenError::TypeError {
            message: format!("Unknown function '{}'", func_name),
        })
    }

    fn check_builtin_function(&mut self, name: &str, args: &[Expr]) -> Result<Option<Type>> {
        match name {
            "matmul" => {
                if args.len() != 2 {
                    return Err(LumenError::TypeError {
                        message: "matmul expects 2 arguments".to_string(),
                    });
                }

                let a_type = self.infer_expr(&args[0])?;
                let b_type = self.infer_expr(&args[1])?;

                self.check_matmul_types(&a_type, &b_type)
            }
            "relu" | "sigmoid" | "tanh" => {
                if args.len() != 1 {
                    return Err(LumenError::TypeError {
                        message: format!("{} expects 1 argument", name),
                    });
                }

                Ok(Some(self.infer_expr(&args[0])?))
            }
            "cross_entropy" => {
                if args.len() != 2 {
                    return Err(LumenError::TypeError {
                        message: "cross_entropy expects 2 arguments".to_string(),
                    });
                }

                // Return scalar f32
                Ok(Some(Type::Primitive(PrimitiveType::F32)))
            }
            _ => Ok(None),
        }
    }

    fn check_matmul_types(&self, a: &Type, b: &Type) -> Result<Option<Type>> {
        match (a, b) {
            (
                Type::Tensor { dtype: dtype_a, dims: dims_a },
                Type::Tensor { dtype: dtype_b, dims: dims_b },
            ) => {
                // Check dtypes match
                if dtype_a != dtype_b {
                    return Err(LumenError::TypeError {
                        message: format!("matmul dtype mismatch: {:?} vs {:?}", dtype_a, dtype_b),
                    });
                }

                // Check dimensions
                if dims_a.len() < 2 || dims_b.len() < 2 {
                    return Err(LumenError::TypeError {
                        message: "matmul requires at least 2D tensors".to_string(),
                    });
                }

                let k_a = &dims_a[dims_a.len() - 1];
                let k_b = &dims_b[dims_b.len() - 2];

                // Check if K dimensions match
                if !self.dims_equal(k_a, k_b) {
                    return Err(LumenError::TypeError {
                        message: format!(
                            "matmul shape mismatch: last dim of A ({:?}) != second-to-last dim of B ({:?})",
                            k_a, k_b
                        ),
                    });
                }

                // Result shape: [..., M, N]
                let mut result_dims = dims_a[..dims_a.len() - 1].to_vec();
                result_dims.push(dims_b[dims_b.len() - 1].clone());

                Ok(Some(Type::Tensor {
                    dtype: dtype_a.clone(),
                    dims: result_dims,
                }))
            }
            _ => Err(LumenError::TypeError {
                message: format!("matmul expects tensor types, got {:?} and {:?}", a, b),
            }),
        }
    }

    fn infer_binary(&mut self, op: BinaryOp, left: &Expr, right: &Expr, _span: Span) -> Result<Type> {
        let left_type = self.infer_expr(left)?;
        let right_type = self.infer_expr(right)?;

        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                self.check_arithmetic_types(&left_type, &right_type)
            }
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                self.unify_types(&left_type, &right_type)?;
                Ok(Type::Primitive(PrimitiveType::Bool))
            }
            BinaryOp::And | BinaryOp::Or => {
                self.unify_types(&Type::Primitive(PrimitiveType::Bool), &left_type)?;
                self.unify_types(&Type::Primitive(PrimitiveType::Bool), &right_type)?;
                Ok(Type::Primitive(PrimitiveType::Bool))
            }
            BinaryOp::MatMul => {
                self.check_matmul_types(&left_type, &right_type)?
                    .ok_or_else(|| LumenError::TypeError {
                        message: "matmul type inference failed".to_string(),
                    })
            }
            _ => Err(LumenError::TypeError {
                message: format!("Unsupported binary operator: {:?}", op),
            }),
        }
    }

    fn infer_unary(&mut self, op: UnaryOp, operand: &Expr, _span: Span) -> Result<Type> {
        let operand_type = self.infer_expr(operand)?;

        match op {
            UnaryOp::Neg => {
                // Negation preserves type
                Ok(operand_type)
            }
            UnaryOp::Not => {
                self.unify_types(&Type::Primitive(PrimitiveType::Bool), &operand_type)?;
                Ok(Type::Primitive(PrimitiveType::Bool))
            }
        }
    }

    fn check_arithmetic_types(&self, left: &Type, right: &Type) -> Result<Type> {
        match (left, right) {
            // Scalar + Scalar
            (Type::Primitive(a), Type::Primitive(b)) if a == b => Ok(left.clone()),

            // Tensor + Tensor (element-wise with broadcasting)
            (
                Type::Tensor { dtype: dtype_a, dims: dims_a },
                Type::Tensor { dtype: dtype_b, dims: dims_b },
            ) => {
                if dtype_a != dtype_b {
                    return Err(LumenError::TypeError {
                        message: format!("Arithmetic dtype mismatch: {:?} vs {:?}", dtype_a, dtype_b),
                    });
                }

                // Simplified broadcasting: require same rank
                if dims_a.len() != dims_b.len() {
                    return Err(LumenError::TypeError {
                        message: format!(
                            "Broadcasting not yet supported: {:?} vs {:?}",
                            dims_a, dims_b
                        ),
                    });
                }

                // Check each dimension
                for (da, db) in dims_a.iter().zip(dims_b.iter()) {
                    if !self.dims_equal(da, db) {
                        return Err(LumenError::TypeError {
                            message: format!("Shape mismatch: {:?} vs {:?}", da, db),
                        });
                    }
                }

                Ok(left.clone())
            }

            _ => Err(LumenError::TypeError {
                message: format!("Cannot perform arithmetic on {:?} and {:?}", left, right),
            }),
        }
    }

    fn check_type(&self, ty: &Type) -> Result<()> {
        match ty {
            Type::Primitive(_) => Ok(()),
            Type::Tensor { dims, .. } => {
                for dim in dims {
                    self.check_dim_expr(dim)?;
                }
                Ok(())
            }
            Type::Named(name) => {
                if self.context.symbol_table.lookup(name).is_none() {
                    return Err(LumenError::TypeError {
                        message: format!("Undefined type '{}'", name),
                    });
                }
                Ok(())
            }
        }
    }

    fn check_dim_expr(&self, dim: &DimExpr) -> Result<()> {
        match dim {
            DimExpr::Const(_) => Ok(()),
            DimExpr::Var(name) => {
                if !self.context.dim_values.contains_key(name) {
                    return Err(LumenError::TypeError {
                        message: format!("Undefined dimension '{}'", name),
                    });
                }
                Ok(())
            }
            DimExpr::Binary { left, right, .. } => {
                self.check_dim_expr(left)?;
                self.check_dim_expr(right)
            }
        }
    }

    fn unify_types(&self, expected: &Type, actual: &Type) -> Result<()> {
        match (expected, actual) {
            (Type::Primitive(a), Type::Primitive(b)) if a == b => Ok(()),
            (
                Type::Tensor { dtype: dtype_a, dims: dims_a },
                Type::Tensor { dtype: dtype_b, dims: dims_b },
            ) => {
                if dtype_a != dtype_b {
                    return Err(LumenError::TypeError {
                        message: format!("Type mismatch: expected {:?}, got {:?}", dtype_a, dtype_b),
                    });
                }

                if dims_a.len() != dims_b.len() {
                    return Err(LumenError::TypeError {
                        message: format!(
                            "Shape rank mismatch: expected {} dims, got {}",
                            dims_a.len(),
                            dims_b.len()
                        ),
                    });
                }

                for (da, db) in dims_a.iter().zip(dims_b.iter()) {
                    if !self.dims_equal(da, db) {
                        return Err(LumenError::TypeError {
                            message: format!("Shape mismatch: {:?} vs {:?}", da, db),
                        });
                    }
                }

                Ok(())
            }
            (Type::Named(a), Type::Named(b)) if a == b => Ok(()),
            _ => Err(LumenError::TypeError {
                message: format!("Type mismatch: expected {:?}, got {:?}", expected, actual),
            }),
        }
    }

    fn dims_equal(&self, a: &DimExpr, b: &DimExpr) -> bool {
        match (a, b) {
            (DimExpr::Const(ca), DimExpr::Const(cb)) => ca == cb,
            (DimExpr::Var(va), DimExpr::Var(vb)) => va == vb,
            (DimExpr::Var(v), DimExpr::Const(c)) | (DimExpr::Const(c), DimExpr::Var(v)) => {
                self.context.get_dim_value(v) == Some(*c)
            }
            _ => false, // Conservative: assume different
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolve::NameResolver;
    use crate::parser::parse;

    #[test]
    fn test_basic_type_check() {
        let source = r#"
            dim batch
            dim features = 784

            fn test(x: Tensor[f32, batch, features]) -> Tensor[f32, batch, features] {
                return x
            }
        "#;

        let program = parse(source).unwrap();
        let mut resolver = NameResolver::new();
        let symbol_table = resolver.resolve(&program).unwrap();

        let mut type_checker = TypeChecker::new(symbol_table);
        let result = type_checker.check(&program);

        assert!(result.is_ok());
    }
}
