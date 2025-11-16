// Automatic Differentiation System
// Implements reverse-mode automatic differentiation (backpropagation)

use crate::mir::*;
use crate::hir::*;
use crate::error::Result;
use std::collections::{HashMap, HashSet};

/// Represents a gradient computation in the backward pass
#[derive(Debug, Clone)]
pub struct GradientOp {
    /// Output gradient variable name
    pub output_grad: String,
    /// Input gradient variable name(s)
    pub input_grads: Vec<String>,
    /// Computation to perform
    pub computation: GradientComputation,
}

#[derive(Debug, Clone)]
pub enum GradientComputation {
    /// MatMul gradient: dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
    MatMulGradA { grad_output: String, b: String },
    MatMulGradB { grad_output: String, a: String },

    /// ReLU gradient: dL/dx = dL/dy * (x > 0)
    ReLUGrad { grad_output: String, input: String },

    /// Sigmoid gradient: dL/dx = dL/dy * y * (1 - y)
    SigmoidGrad { grad_output: String, output: String },

    /// Tanh gradient: dL/dx = dL/dy * (1 - y^2)
    TanhGrad { grad_output: String, output: String },

    /// Element-wise Add gradient: dL/dx = dL/dz, dL/dy = dL/dz
    AddGrad { grad_output: String },

    /// Element-wise Mul gradient: dL/dx = dL/dz * y, dL/dy = dL/dz * x
    MulGrad { grad_output: String, other: String },

    /// Cross-entropy gradient
    CrossEntropyGrad { grad_output: String, logits: String, targets: String },

    /// MSE loss gradient
    MSEGrad { grad_output: String, predictions: String, targets: String },

    /// Accumulate gradients (for when a variable is used multiple times)
    Accumulate { grad_vars: Vec<String> },
}

/// Autograd context that tracks the computation graph
pub struct AutogradContext {
    /// Forward operations in topological order
    forward_ops: Vec<MirStmt>,

    /// Backward operations to compute gradients
    backward_ops: Vec<GradientOp>,

    /// Variables that require gradients
    requires_grad: HashSet<String>,

    /// Map from forward var to its gradient var name
    grad_names: HashMap<String, String>,
}

impl AutogradContext {
    pub fn new() -> Self {
        Self {
            forward_ops: Vec::new(),
            backward_ops: Vec::new(),
            requires_grad: HashSet::new(),
            grad_names: HashMap::new(),
        }
    }

    /// Mark parameters as requiring gradients
    pub fn mark_requires_grad(&mut self, var_names: Vec<String>) {
        for name in var_names {
            self.requires_grad.insert(name);
        }
    }

    /// Build backward pass from forward computation graph
    pub fn build_backward(&mut self, forward_stmts: &[MirStmt], loss_var: &str) -> Result<()> {
        self.forward_ops = forward_stmts.to_vec();

        // Initialize gradient of loss as 1.0
        let _loss_grad = self.get_grad_name(loss_var);

        // Process forward operations in reverse order
        for stmt in forward_stmts.iter().rev() {
            if let MirStmt::Assign { name, value, .. } = stmt {
                let output_grad = self.get_grad_name(name);

                match value {
                    MirExpr::Call { func, args } => {
                        self.add_backward_for_call(func, args, name, &output_grad)?;
                    }
                    MirExpr::BinOp { op, left, right } => {
                        self.add_backward_for_binop(op, left, right, name, &output_grad)?;
                    }
                    MirExpr::Var { .. } => {
                        // Simple assignment, gradient flows through
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }

    fn add_backward_for_call(&mut self, func: &str, args: &[MirExpr], output: &str, output_grad: &str) -> Result<()> {
        match func {
            "matmul" => {
                if args.len() == 2 {
                    if let (MirExpr::Var { name: a, .. }, MirExpr::Var { name: b, .. }) = (&args[0], &args[1]) {
                        // dL/dA = dL/dC @ B^T
                        let grad_a = self.get_grad_name(a);
                        self.backward_ops.push(GradientOp {
                            output_grad: output_grad.to_string(),
                            input_grads: vec![grad_a.clone()],
                            computation: GradientComputation::MatMulGradA {
                                grad_output: output_grad.to_string(),
                                b: b.clone(),
                            },
                        });

                        // dL/dB = A^T @ dL/dC
                        let grad_b = self.get_grad_name(b);
                        self.backward_ops.push(GradientOp {
                            output_grad: output_grad.to_string(),
                            input_grads: vec![grad_b.clone()],
                            computation: GradientComputation::MatMulGradB {
                                grad_output: output_grad.to_string(),
                                a: a.clone(),
                            },
                        });
                    }
                }
            }
            "relu" => {
                if let Some(MirExpr::Var { name: input, .. }) = args.first() {
                    let grad_input = self.get_grad_name(input);
                    self.backward_ops.push(GradientOp {
                        output_grad: output_grad.to_string(),
                        input_grads: vec![grad_input],
                        computation: GradientComputation::ReLUGrad {
                            grad_output: output_grad.to_string(),
                            input: input.clone(),
                        },
                    });
                }
            }
            "sigmoid" => {
                if let Some(MirExpr::Var { name: input, .. }) = args.first() {
                    let grad_input = self.get_grad_name(input);
                    self.backward_ops.push(GradientOp {
                        output_grad: output_grad.to_string(),
                        input_grads: vec![grad_input],
                        computation: GradientComputation::SigmoidGrad {
                            grad_output: output_grad.to_string(),
                            output: output.to_string(),
                        },
                    });
                }
            }
            "tanh" => {
                if let Some(MirExpr::Var { name: input, .. }) = args.first() {
                    let grad_input = self.get_grad_name(input);
                    self.backward_ops.push(GradientOp {
                        output_grad: output_grad.to_string(),
                        input_grads: vec![grad_input],
                        computation: GradientComputation::TanhGrad {
                            grad_output: output_grad.to_string(),
                            output: output.to_string(),
                        },
                    });
                }
            }
            "cross_entropy" => {
                if args.len() == 2 {
                    if let (MirExpr::Var { name: logits, .. }, MirExpr::Var { name: targets, .. }) = (&args[0], &args[1]) {
                        let grad_logits = self.get_grad_name(logits);
                        self.backward_ops.push(GradientOp {
                            output_grad: output_grad.to_string(),
                            input_grads: vec![grad_logits],
                            computation: GradientComputation::CrossEntropyGrad {
                                grad_output: output_grad.to_string(),
                                logits: logits.clone(),
                                targets: targets.clone(),
                            },
                        });
                    }
                }
            }
            "mse_loss" => {
                if args.len() == 2 {
                    if let (MirExpr::Var { name: pred, .. }, MirExpr::Var { name: target, .. }) = (&args[0], &args[1]) {
                        let grad_pred = self.get_grad_name(pred);
                        self.backward_ops.push(GradientOp {
                            output_grad: output_grad.to_string(),
                            input_grads: vec![grad_pred],
                            computation: GradientComputation::MSEGrad {
                                grad_output: output_grad.to_string(),
                                predictions: pred.clone(),
                                targets: target.clone(),
                            },
                        });
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn add_backward_for_binop(&mut self, op: &HirBinOp, left: &MirExpr, right: &MirExpr, _output: &str, output_grad: &str) -> Result<()> {
        match op {
            HirBinOp::Add => {
                // dL/dx = dL/dz, dL/dy = dL/dz
                if let MirExpr::Var { name: left_name, .. } = left {
                    let grad_left = self.get_grad_name(left_name);
                    self.backward_ops.push(GradientOp {
                        output_grad: output_grad.to_string(),
                        input_grads: vec![grad_left],
                        computation: GradientComputation::AddGrad {
                            grad_output: output_grad.to_string(),
                        },
                    });
                }
                if let MirExpr::Var { name: right_name, .. } = right {
                    let grad_right = self.get_grad_name(right_name);
                    self.backward_ops.push(GradientOp {
                        output_grad: output_grad.to_string(),
                        input_grads: vec![grad_right],
                        computation: GradientComputation::AddGrad {
                            grad_output: output_grad.to_string(),
                        },
                    });
                }
            }
            HirBinOp::Mul => {
                // dL/dx = dL/dz * y, dL/dy = dL/dz * x
                if let (MirExpr::Var { name: left_name, .. }, MirExpr::Var { name: right_name, .. }) = (left, right) {
                    let grad_left = self.get_grad_name(left_name);
                    self.backward_ops.push(GradientOp {
                        output_grad: output_grad.to_string(),
                        input_grads: vec![grad_left],
                        computation: GradientComputation::MulGrad {
                            grad_output: output_grad.to_string(),
                            other: right_name.clone(),
                        },
                    });

                    let grad_right = self.get_grad_name(right_name);
                    self.backward_ops.push(GradientOp {
                        output_grad: output_grad.to_string(),
                        input_grads: vec![grad_right],
                        computation: GradientComputation::MulGrad {
                            grad_output: output_grad.to_string(),
                            other: left_name.clone(),
                        },
                    });
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn get_grad_name(&mut self, var_name: &str) -> String {
        if let Some(grad) = self.grad_names.get(var_name) {
            grad.clone()
        } else {
            let grad_name = format!("grad_{}", var_name);
            self.grad_names.insert(var_name.to_string(), grad_name.clone());
            grad_name
        }
    }

    pub fn get_backward_ops(&self) -> &[GradientOp] {
        &self.backward_ops
    }

    pub fn get_parameter_gradients(&self) -> HashMap<String, String> {
        let mut param_grads = HashMap::new();
        for param in &self.requires_grad {
            if let Some(grad) = self.grad_names.get(param) {
                param_grads.insert(param.clone(), grad.clone());
            }
        }
        param_grads
    }
}

/// Optimizer implementations
#[derive(Debug, Clone)]
pub enum Optimizer {
    SGD { learning_rate: f32, momentum: f32 },
    Adam { learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32 },
}

impl Optimizer {
    pub fn sgd(lr: f32) -> Self {
        Optimizer::SGD {
            learning_rate: lr,
            momentum: 0.0,
        }
    }

    pub fn sgd_momentum(lr: f32, momentum: f32) -> Self {
        Optimizer::SGD {
            learning_rate: lr,
            momentum,
        }
    }

    pub fn adam(lr: f32) -> Self {
        Optimizer::Adam {
            learning_rate: lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autograd_simple() {
        let mut ctx = AutogradContext::new();
        ctx.mark_requires_grad(vec!["W".to_string()]);

        // Simple forward: y = x @ W
        let forward = vec![
            MirStmt::Assign {
                var_id: 0,
                name: "y".to_string(),
                value: MirExpr::Call {
                    func: "matmul".to_string(),
                    args: vec![
                        MirExpr::Var { var_id: 1, name: "x".to_string() },
                        MirExpr::Var { var_id: 2, name: "W".to_string() },
                    ],
                },
            },
        ];

        ctx.build_backward(&forward, "y").unwrap();

        let backward = ctx.get_backward_ops();
        assert!(!backward.is_empty());
    }
}
