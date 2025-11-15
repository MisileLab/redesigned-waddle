// Name resolution and symbol table

use std::collections::HashMap;
use crate::ast::*;
use crate::error::{LumenError, Result};

pub type SymbolId = usize;

#[derive(Debug, Clone)]
pub struct SymbolTable {
    scopes: Vec<Scope>,
    current_scope: usize,
    pub symbols: Vec<Symbol>,
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
        value: Option<i64>,
    },
    Model {
        name: String,
        params: Vec<SymbolId>,
        methods: Vec<SymbolId>,
    },
    Param {
        name: String,
        ty: Type,
        initializer: Initializer,
    },
    Function {
        name: String,
        params: Vec<Param>,
        return_type: Option<Type>,
    },
    Variable {
        name: String,
        ty: Option<Type>,
    },
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            scopes: vec![Scope {
                parent: None,
                bindings: HashMap::new(),
            }],
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

    pub fn insert(&mut self, name: String, symbol: Symbol) -> Result<SymbolId> {
        // Check for duplicate in current scope
        if self.scopes[self.current_scope].bindings.contains_key(&name) {
            return Err(LumenError::NameError {
                message: format!("Name '{}' is already defined in this scope", name),
            });
        }

        let id = self.symbols.len();
        self.symbols.push(symbol);
        self.scopes[self.current_scope].bindings.insert(name, id);
        Ok(id)
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
}

impl NameResolver {
    pub fn new() -> Self {
        let mut resolver = Self {
            symbol_table: SymbolTable::new(),
        };
        resolver.add_builtins();
        resolver
    }

    fn add_builtins(&mut self) {
        // Add built-in functions
        let builtins = vec![
            "matmul", "relu", "sigmoid", "tanh", "softmax",
            "cross_entropy", "mse_loss",
            "zeros", "ones", "randn", "reshape", "transpose",
        ];

        for name in builtins {
            // Add as a simple function symbol (params will be checked in type checker)
            let _ = self.symbol_table.insert(
                name.to_string(),
                Symbol::Function {
                    name: name.to_string(),
                    params: vec![],  // Builtins have flexible params
                    return_type: None,
                },
            );
        }
    }

    pub fn resolve(&mut self, program: &Program) -> Result<SymbolTable> {
        // First pass: collect all top-level declarations
        for decl in &program.declarations {
            self.collect_declaration(decl)?;
        }

        // Second pass: resolve references
        for decl in &program.declarations {
            self.resolve_declaration(decl)?;
        }

        Ok(self.symbol_table.clone())
    }

    fn collect_declaration(&mut self, decl: &Declaration) -> Result<()> {
        match decl {
            Declaration::Dim(dim) => {
                self.symbol_table.insert(
                    dim.name.clone(),
                    Symbol::Dim {
                        name: dim.name.clone(),
                        value: dim.value,
                    },
                )?;
            }
            Declaration::Model(model) => {
                let mut param_ids = Vec::new();
                let mut method_ids = Vec::new();

                // Enter model scope
                self.symbol_table.enter_scope();

                // Collect params and methods
                for member in &model.members {
                    match member {
                        ModelMember::Param(param) => {
                            let id = self.symbol_table.insert(
                                param.name.clone(),
                                Symbol::Param {
                                    name: param.name.clone(),
                                    ty: param.ty.clone(),
                                    initializer: param.initializer.clone(),
                                },
                            )?;
                            param_ids.push(id);
                        }
                        ModelMember::Function(func) => {
                            let id = self.symbol_table.insert(
                                func.name.clone(),
                                Symbol::Function {
                                    name: func.name.clone(),
                                    params: func.params.clone(),
                                    return_type: func.return_type.clone(),
                                },
                            )?;
                            method_ids.push(id);
                        }
                    }
                }

                self.symbol_table.exit_scope();

                // Insert model into parent scope
                self.symbol_table.insert(
                    model.name.clone(),
                    Symbol::Model {
                        name: model.name.clone(),
                        params: param_ids,
                        methods: method_ids,
                    },
                )?;
            }
            Declaration::Function(func) => {
                self.symbol_table.insert(
                    func.name.clone(),
                    Symbol::Function {
                        name: func.name.clone(),
                        params: func.params.clone(),
                        return_type: func.return_type.clone(),
                    },
                )?;
            }
        }
        Ok(())
    }

    fn resolve_declaration(&mut self, decl: &Declaration) -> Result<()> {
        match decl {
            Declaration::Dim(_) => Ok(()),
            Declaration::Model(model) => {
                self.symbol_table.enter_scope();

                // Re-add params and methods to scope
                for member in &model.members {
                    match member {
                        ModelMember::Param(param) => {
                            self.resolve_type(&param.ty)?;
                        }
                        ModelMember::Function(func) => {
                            self.resolve_function(func)?;
                        }
                    }
                }

                self.symbol_table.exit_scope();
                Ok(())
            }
            Declaration::Function(func) => self.resolve_function(func),
        }
    }

    fn resolve_function(&mut self, func: &FunctionDecl) -> Result<()> {
        self.symbol_table.enter_scope();

        // Add parameters to scope
        for param in &func.params {
            self.resolve_type(&param.ty)?;
            self.symbol_table.insert(
                param.name.clone(),
                Symbol::Variable {
                    name: param.name.clone(),
                    ty: Some(param.ty.clone()),
                },
            )?;
        }

        // Resolve return type
        if let Some(ref ret_ty) = func.return_type {
            self.resolve_type(ret_ty)?;
        }

        // Resolve body
        self.resolve_block(&func.body)?;

        self.symbol_table.exit_scope();
        Ok(())
    }

    fn resolve_block(&mut self, block: &Block) -> Result<()> {
        for stmt in &block.stmts {
            self.resolve_stmt(stmt)?;
        }
        Ok(())
    }

    fn resolve_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        match stmt {
            Stmt::Let { name, ty, value, .. } => {
                if let Some(ref ty) = ty {
                    self.resolve_type(ty)?;
                }
                self.resolve_expr(value)?;
                self.symbol_table.insert(
                    name.clone(),
                    Symbol::Variable {
                        name: name.clone(),
                        ty: ty.clone(),
                    },
                )?;
            }
            Stmt::Return { value, .. } => {
                if let Some(ref expr) = value {
                    self.resolve_expr(expr)?;
                }
            }
            Stmt::Expr(expr) => {
                self.resolve_expr(expr)?;
            }
        }
        Ok(())
    }

    fn resolve_expr(&mut self, expr: &Expr) -> Result<()> {
        match expr {
            Expr::Ident(name, span) => {
                if self.symbol_table.lookup(name).is_none() {
                    return Err(LumenError::NameError {
                        message: format!("Undefined variable '{}' at {:?}", name, span),
                    });
                }
            }
            Expr::FieldAccess { object, .. } => {
                self.resolve_expr(object)?;
            }
            Expr::Call { func, args, .. } => {
                self.resolve_expr(func)?;
                for arg in args {
                    self.resolve_expr(arg)?;
                }
            }
            Expr::Binary { left, right, .. } => {
                self.resolve_expr(left)?;
                self.resolve_expr(right)?;
            }
            Expr::Unary { operand, .. } => {
                self.resolve_expr(operand)?;
            }
            Expr::Literal(_, _) => {}
        }
        Ok(())
    }

    fn resolve_type(&mut self, ty: &Type) -> Result<()> {
        match ty {
            Type::Tensor { dims, .. } => {
                for dim in dims {
                    self.resolve_dim_expr(dim)?;
                }
            }
            Type::Named(name) => {
                if self.symbol_table.lookup(name).is_none() {
                    return Err(LumenError::NameError {
                        message: format!("Undefined type '{}'", name),
                    });
                }
            }
            Type::Primitive(_) => {}
        }
        Ok(())
    }

    fn resolve_dim_expr(&mut self, dim: &DimExpr) -> Result<()> {
        match dim {
            DimExpr::Var(name) => {
                if self.symbol_table.lookup(name).is_none() {
                    return Err(LumenError::NameError {
                        message: format!("Undefined dimension '{}'", name),
                    });
                }
            }
            DimExpr::Binary { left, right, .. } => {
                self.resolve_dim_expr(left)?;
                self.resolve_dim_expr(right)?;
            }
            DimExpr::Const(_) => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_table() {
        let mut table = SymbolTable::new();

        let id = table.insert(
            "batch".to_string(),
            Symbol::Dim {
                name: "batch".to_string(),
                value: None,
            },
        ).unwrap();

        assert_eq!(table.lookup("batch"), Some(id));
        assert_eq!(table.lookup("unknown"), None);
    }

    #[test]
    fn test_scopes() {
        let mut table = SymbolTable::new();

        table.insert(
            "outer".to_string(),
            Symbol::Dim {
                name: "outer".to_string(),
                value: Some(10),
            },
        ).unwrap();

        table.enter_scope();

        table.insert(
            "inner".to_string(),
            Symbol::Dim {
                name: "inner".to_string(),
                value: Some(20),
            },
        ).unwrap();

        assert!(table.lookup("outer").is_some());
        assert!(table.lookup("inner").is_some());

        table.exit_scope();

        assert!(table.lookup("outer").is_some());
        assert!(table.lookup("inner").is_none());
    }
}
