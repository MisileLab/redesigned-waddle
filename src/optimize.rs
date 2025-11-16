// Optimization Passes for MIR
// Implements operator fusion, memory optimization, and graph transformations

use crate::mir::*;
use crate::hir::*;
use crate::error::Result;
use std::collections::{HashMap, HashSet};

/// Optimization pipeline that applies multiple passes
pub struct OptimizationPipeline {
    passes: Vec<Box<dyn OptimizationPass>>,
}

impl OptimizationPipeline {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
        }
    }

    /// Standard optimization pipeline
    pub fn standard() -> Self {
        let mut pipeline = Self::new();
        pipeline.add_pass(Box::new(DeadCodeElimination::new()));
        pipeline.add_pass(Box::new(CommonSubexpressionElimination::new()));
        pipeline.add_pass(Box::new(ConstantPropagation::new()));
        pipeline.add_pass(Box::new(OperatorFusion::new()));
        pipeline.add_pass(Box::new(AlgebraicSimplification::new()));
        pipeline
    }

    /// Aggressive optimization (may increase compile time)
    pub fn aggressive() -> Self {
        let mut pipeline = Self::standard();
        pipeline.add_pass(Box::new(MemoryOptimization::new()));
        pipeline
    }

    pub fn add_pass(&mut self, pass: Box<dyn OptimizationPass>) {
        self.passes.push(pass);
    }

    pub fn optimize(&mut self, program: &mut MirProgram) -> Result<()> {
        for pass in &mut self.passes {
            pass.run(program)?;
        }
        Ok(())
    }
}

pub trait OptimizationPass {
    fn run(&mut self, program: &mut MirProgram) -> Result<()>;
    fn name(&self) -> &str;
}

// ============================================================================
// Dead Code Elimination (DCE)
// ============================================================================

pub struct DeadCodeElimination {
    live_vars: HashSet<String>,
}

impl DeadCodeElimination {
    pub fn new() -> Self {
        Self {
            live_vars: HashSet::new(),
        }
    }

    fn mark_live(&mut self, expr: &MirExpr) {
        match expr {
            MirExpr::Var { name, .. } => {
                self.live_vars.insert(name.clone());
            }
            MirExpr::Call { args, .. } => {
                for arg in args {
                    self.mark_live(arg);
                }
            }
            MirExpr::BinOp { left, right, .. } => {
                self.mark_live(left);
                self.mark_live(right);
            }
            MirExpr::UnaryOp { operand, .. } => {
                self.mark_live(operand);
            }
            MirExpr::FieldAccess { object, .. } => {
                self.mark_live(object);
            }
            _ => {}
        }
    }
}

impl OptimizationPass for DeadCodeElimination {
    fn run(&mut self, program: &mut MirProgram) -> Result<()> {
        // Process each function
        for func in &mut program.functions {
            self.live_vars.clear();

            // Mark variables used in return statements
            for stmt in &func.body {
                if let MirStmt::Return { value: Some(expr) } = stmt {
                    self.mark_live(expr);
                }
            }

            // Backward pass to mark live variables
            for stmt in func.body.iter().rev() {
                if let MirStmt::Assign { name, value, .. } = stmt {
                    if self.live_vars.contains(name) {
                        self.mark_live(value);
                    }
                }
            }

            // Remove dead assignments
            func.body.retain(|stmt| {
                match stmt {
                    MirStmt::Assign { name, .. } => self.live_vars.contains(name),
                    _ => true,
                }
            });
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "DeadCodeElimination"
    }
}

// ============================================================================
// Common Subexpression Elimination (CSE)
// ============================================================================

pub struct CommonSubexpressionElimination {
    expr_map: HashMap<String, String>,
}

impl CommonSubexpressionElimination {
    pub fn new() -> Self {
        Self {
            expr_map: HashMap::new(),
        }
    }

    fn expr_key(&self, expr: &MirExpr) -> Option<String> {
        match expr {
            MirExpr::Call { func, args } => {
                let args_str: Vec<String> = args.iter()
                    .filter_map(|arg| self.expr_key(arg))
                    .collect();
                Some(format!("{}({})", func, args_str.join(", ")))
            }
            MirExpr::BinOp { op, left, right } => {
                let left_key = self.expr_key(left)?;
                let right_key = self.expr_key(right)?;
                Some(format!("({:?} {} {})", op, left_key, right_key))
            }
            MirExpr::Var { name, .. } => Some(name.clone()),
            _ => None,
        }
    }
}

impl OptimizationPass for CommonSubexpressionElimination {
    fn run(&mut self, program: &mut MirProgram) -> Result<()> {
        for func in &mut program.functions {
            self.expr_map.clear();

            for stmt in &mut func.body {
                if let MirStmt::Assign { name, value, .. } = stmt {
                    if let Some(key) = self.expr_key(value) {
                        if let Some(existing_var) = self.expr_map.get(&key) {
                            // Replace with existing computation
                            *value = MirExpr::Var {
                                var_id: 0, // Will be renumbered
                                name: existing_var.clone(),
                            };
                        } else {
                            self.expr_map.insert(key, name.clone());
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "CommonSubexpressionElimination"
    }
}

// ============================================================================
// Constant Propagation & Folding
// ============================================================================

pub struct ConstantPropagation {
    constants: HashMap<String, MirExpr>,
}

impl ConstantPropagation {
    pub fn new() -> Self {
        Self {
            constants: HashMap::new(),
        }
    }

    fn is_constant(&self, expr: &MirExpr) -> bool {
        matches!(expr, MirExpr::Literal(_))
    }

    fn fold_constant(&self, expr: &mut MirExpr) {
        match expr {
            MirExpr::Var { name, .. } => {
                if let Some(const_expr) = self.constants.get(name) {
                    *expr = const_expr.clone();
                }
            }
            MirExpr::Call { args, .. } => {
                for arg in args {
                    self.fold_constant(arg);
                }
            }
            MirExpr::BinOp { left, right, .. } => {
                self.fold_constant(left);
                self.fold_constant(right);
            }
            _ => {}
        }
    }
}

impl OptimizationPass for ConstantPropagation {
    fn run(&mut self, program: &mut MirProgram) -> Result<()> {
        for func in &mut program.functions {
            self.constants.clear();

            for stmt in &mut func.body {
                if let MirStmt::Assign { name, value, .. } = stmt {
                    self.fold_constant(value);

                    if self.is_constant(value) {
                        self.constants.insert(name.clone(), value.clone());
                    }
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "ConstantPropagation"
    }
}

// ============================================================================
// Algebraic Simplification
// ============================================================================

pub struct AlgebraicSimplification;

impl AlgebraicSimplification {
    pub fn new() -> Self {
        Self
    }

    fn simplify_expr(&self, expr: &mut MirExpr) {
        match expr {
            MirExpr::BinOp { op, left, right } => {
                // Recursively simplify
                self.simplify_expr(left);
                self.simplify_expr(right);

                // Apply simplification rules
                match op {
                    HirBinOp::Mul => {
                        // x * 1 = x
                        if matches!(**right, MirExpr::Literal(HirLiteral::Int(1))) {
                            *expr = (**left).clone();
                        }
                        // x * 0 = 0
                        else if matches!(**right, MirExpr::Literal(HirLiteral::Int(0))) {
                            *expr = MirExpr::Literal(HirLiteral::Int(0));
                        }
                    }
                    HirBinOp::Add => {
                        // x + 0 = x
                        if matches!(**right, MirExpr::Literal(HirLiteral::Int(0))) {
                            *expr = (**left).clone();
                        }
                    }
                    HirBinOp::Sub => {
                        // x - 0 = x
                        if matches!(**right, MirExpr::Literal(HirLiteral::Int(0))) {
                            *expr = (**left).clone();
                        }
                    }
                    HirBinOp::Div => {
                        // x / 1 = x
                        if matches!(**right, MirExpr::Literal(HirLiteral::Int(1))) {
                            *expr = (**left).clone();
                        }
                    }
                    _ => {}
                }
            }
            MirExpr::Call { args, .. } => {
                for arg in args {
                    self.simplify_expr(arg);
                }
            }
            _ => {}
        }
    }
}

impl OptimizationPass for AlgebraicSimplification {
    fn run(&mut self, program: &mut MirProgram) -> Result<()> {
        for func in &mut program.functions {
            for stmt in &mut func.body {
                if let MirStmt::Assign { value, .. } = stmt {
                    self.simplify_expr(value);
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "AlgebraicSimplification"
    }
}

// ============================================================================
// Operator Fusion
// ============================================================================

pub struct OperatorFusion {
    fusion_candidates: Vec<FusionPattern>,
}

#[derive(Debug)]
#[allow(dead_code)]
enum FusionPattern {
    /// Fuse: matmul + relu -> fused_linear_relu
    LinearReLU { matmul_var: String, relu_var: String },

    /// Fuse: matmul + bias + relu
    LinearBiasReLU { matmul_var: String, add_var: String, relu_var: String },

    /// Fuse element-wise ops: (x + y) * z
    ElementWise { vars: Vec<String> },
}

impl OperatorFusion {
    pub fn new() -> Self {
        Self {
            fusion_candidates: Vec::new(),
        }
    }

    fn detect_patterns(&mut self, func: &MirFunction) {
        // Detect matmul + relu pattern
        for i in 0..func.body.len().saturating_sub(1) {
            if let (
                MirStmt::Assign { name: matmul_var, value: MirExpr::Call { func: f1, .. }, .. },
                MirStmt::Assign { name: relu_var, value: MirExpr::Call { func: f2, args, .. }, .. }
            ) = (&func.body[i], &func.body[i + 1]) {
                if f1 == "matmul" && f2 == "relu" {
                    if let Some(MirExpr::Var { name, .. }) = args.first() {
                        if name == matmul_var {
                            self.fusion_candidates.push(FusionPattern::LinearReLU {
                                matmul_var: matmul_var.clone(),
                                relu_var: relu_var.clone(),
                            });
                        }
                    }
                }
            }
        }
    }
}

impl OptimizationPass for OperatorFusion {
    fn run(&mut self, program: &mut MirProgram) -> Result<()> {
        for func in &mut program.functions {
            self.fusion_candidates.clear();
            self.detect_patterns(func);

            // Apply fusion transformations
            // (Simplified - actual fusion would modify the IR)
            for pattern in &self.fusion_candidates {
                match pattern {
                    FusionPattern::LinearReLU { matmul_var, relu_var } => {
                        // Mark for fusion (actual implementation would combine ops)
                        eprintln!("Fusion opportunity: {} -> {}", matmul_var, relu_var);
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "OperatorFusion"
    }
}

// ============================================================================
// Memory Optimization
// ============================================================================

pub struct MemoryOptimization {
    live_ranges: HashMap<String, (usize, usize)>,
}

impl MemoryOptimization {
    pub fn new() -> Self {
        Self {
            live_ranges: HashMap::new(),
        }
    }

    fn compute_live_ranges(&mut self, func: &MirFunction) {
        self.live_ranges.clear();

        // Compute first and last use of each variable
        for (i, stmt) in func.body.iter().enumerate() {
            match stmt {
                MirStmt::Assign { name, value, .. } => {
                    self.live_ranges.entry(name.clone())
                        .or_insert((i, i))
                        .0 = i;

                    self.mark_uses(value, i);
                }
                MirStmt::Return { value } => {
                    if let Some(expr) = value {
                        self.mark_uses(expr, i);
                    }
                }
            }
        }
    }

    fn mark_uses(&mut self, expr: &MirExpr, pos: usize) {
        match expr {
            MirExpr::Var { name, .. } => {
                if let Some(range) = self.live_ranges.get_mut(name) {
                    range.1 = pos;
                }
            }
            MirExpr::Call { args, .. } => {
                for arg in args {
                    self.mark_uses(arg, pos);
                }
            }
            MirExpr::BinOp { left, right, .. } => {
                self.mark_uses(left, pos);
                self.mark_uses(right, pos);
            }
            _ => {}
        }
    }

    fn can_reuse_memory(&self, var1: &str, var2: &str) -> bool {
        if let (Some(range1), Some(range2)) = (self.live_ranges.get(var1), self.live_ranges.get(var2)) {
            // Non-overlapping ranges can share memory
            range1.1 < range2.0 || range2.1 < range1.0
        } else {
            false
        }
    }
}

impl OptimizationPass for MemoryOptimization {
    fn run(&mut self, program: &mut MirProgram) -> Result<()> {
        for func in &program.functions {
            self.compute_live_ranges(func);

            // Find memory reuse opportunities
            let vars: Vec<String> = self.live_ranges.keys().cloned().collect();
            for i in 0..vars.len() {
                for j in (i + 1)..vars.len() {
                    if self.can_reuse_memory(&vars[i], &vars[j]) {
                        eprintln!("Memory reuse opportunity: {} and {}", vars[i], vars[j]);
                    }
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "MemoryOptimization"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dce() {
        let mut program = MirProgram { models: vec![], 
            models: vec![],
            functions: vec![MirFunction {
                name: "test".to_string(),
                params: vec![],
                return_type: None,
                body: vec![
                    MirStmt::Assign {
                        var_id: 0,
                        name: "dead".to_string(),
                        value: MirExpr::Literal(HirLiteral::Int(42)),
                    },
                    MirStmt::Assign {
                        var_id: 1,
                        name: "live".to_string(),
                        value: MirExpr::Literal(HirLiteral::Int(10)),
                    },
                    MirStmt::Return {
                        value: Some(MirExpr::Var { var_id: 1, name: "live".to_string() }),
                    },
                ],
            }],
        };

        let mut pass = DeadCodeElimination::new();
        pass.run(&mut program).unwrap();

        // Dead code should be eliminated
        assert_eq!(program.functions[0].body.len(), 2);
    }
}
