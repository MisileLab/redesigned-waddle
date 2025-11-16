// Graph IR - Higher-level computation graph representation
// Enables advanced graph-level optimizations

use crate::mir::*;
use crate::error::{Result, LumenError};
use std::collections::HashMap;

/// Node in the computation graph
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: usize,
    pub op: GraphOp,
    pub inputs: Vec<usize>,  // Input node IDs
    pub output_shape: Vec<i64>,
    pub dtype: String,
}

/// Graph operations
#[derive(Debug, Clone)]
pub enum GraphOp {
    Input { name: String },
    Constant { value: Vec<f32> },
    MatMul,
    Add,
    Mul,
    ReLU,
    Sigmoid,
    Tanh,
    Conv2d { stride: (usize, usize), padding: (usize, usize) },
    BatchNorm,
    MaxPool2d { kernel_size: (usize, usize) },
    Softmax { dim: i64 },
    LayerNorm { eps: f32 },
    Attention,
    Reshape { new_shape: Vec<i64> },
}

/// Computation graph
#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<GraphNode>,
    pub inputs: Vec<usize>,   // Input node IDs
    pub outputs: Vec<usize>,  // Output node IDs
    pub node_counter: usize,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            node_counter: 0,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, op: GraphOp, inputs: Vec<usize>, output_shape: Vec<i64>, dtype: String) -> usize {
        let id = self.node_counter;
        self.node_counter += 1;

        self.nodes.push(GraphNode {
            id,
            op,
            inputs,
            output_shape,
            dtype,
        });

        id
    }

    /// Build graph from MIR
    pub fn from_mir(mir: &MirProgram) -> Result<Self> {
        let mut graph = Graph::new();
        let mut var_to_node: HashMap<String, usize> = HashMap::new();

        for func in &mir.functions {
            for stmt in &func.body {
                if let MirStmt::Assign { name, value, .. } = stmt {
                    let node_id = Self::expr_to_graph(&mut graph, value, &var_to_node, "f32")?;
                    var_to_node.insert(name.clone(), node_id);
                }
            }
        }

        Ok(graph)
    }

    fn expr_to_graph(
        graph: &mut Graph,
        expr: &MirExpr,
        var_to_node: &HashMap<String, usize>,
        output_type: &str,
    ) -> Result<usize> {
        match expr {
            MirExpr::Var { name, .. } => {
                var_to_node.get(name).copied().ok_or_else(|| LumenError::CodegenError {
                    message: format!("Undefined variable in graph: {}", name),
                })
            }
            MirExpr::Call { func, args } => {
                let input_ids: Vec<usize> = args
                    .iter()
                    .filter_map(|arg| {
                        if let MirExpr::Var { name, .. } = arg {
                            var_to_node.get(name).copied()
                        } else {
                            None
                        }
                    })
                    .collect();

                let op = match func.as_str() {
                    "matmul" => GraphOp::MatMul,
                    "relu" => GraphOp::ReLU,
                    "sigmoid" => GraphOp::Sigmoid,
                    "tanh" => GraphOp::Tanh,
                    "softmax" => GraphOp::Softmax { dim: -1 },
                    _ => GraphOp::MatMul, // Default
                };

                Ok(graph.add_node(op, input_ids, vec![], output_type.to_string()))
            }
            MirExpr::BinOp { op, left, right } => {
                let left_id = Self::expr_to_graph(graph, left, var_to_node, output_type)?;
                let right_id = Self::expr_to_graph(graph, right, var_to_node, output_type)?;

                let graph_op = match op {
                    crate::hir::HirBinOp::Add => GraphOp::Add,
                    crate::hir::HirBinOp::Mul => GraphOp::Mul,
                    _ => GraphOp::Add,
                };

                Ok(graph.add_node(graph_op, vec![left_id, right_id], vec![], output_type.to_string()))
            }
            _ => Ok(graph.add_node(GraphOp::Input { name: "unknown".to_string() }, vec![], vec![], output_type.to_string())),
        }
    }

    /// Optimize the graph
    pub fn optimize(&mut self) -> Result<()> {
        self.eliminate_dead_nodes()?;
        self.fuse_operations()?;
        self.constant_folding()?;
        Ok(())
    }

    /// Remove nodes that don't contribute to outputs
    fn eliminate_dead_nodes(&mut self) -> Result<()> {
        let mut reachable = vec![false; self.nodes.len()];

        // Mark outputs as reachable
        for &output_id in &self.outputs {
            self.mark_reachable(output_id, &mut reachable);
        }

        // Remove unreachable nodes
        self.nodes.retain(|node| reachable[node.id]);

        Ok(())
    }

    fn mark_reachable(&self, node_id: usize, reachable: &mut [bool]) {
        if reachable[node_id] {
            return;
        }

        reachable[node_id] = true;

        if let Some(node) = self.nodes.iter().find(|n| n.id == node_id) {
            for &input_id in &node.inputs {
                self.mark_reachable(input_id, reachable);
            }
        }
    }

    /// Fuse compatible operations
    fn fuse_operations(&mut self) -> Result<()> {
        // Example: Fuse MatMul + ReLU into single operation
        let mut fused_nodes = Vec::new();

        for i in 0..self.nodes.len() {
            if let GraphOp::ReLU = self.nodes[i].op {
                if self.nodes[i].inputs.len() == 1 {
                    let input_id = self.nodes[i].inputs[0];
                    if let Some(input_node) = self.nodes.iter().find(|n| n.id == input_id) {
                        if matches!(input_node.op, GraphOp::MatMul) {
                            // Mark for fusion
                            fused_nodes.push((input_id, self.nodes[i].id));
                        }
                    }
                }
            }
        }

        // Apply fusions (simplified)
        for (_matmul_id, _relu_id) in fused_nodes {
            // In practice, create new fused node and update graph
        }

        Ok(())
    }

    /// Fold constant operations
    fn constant_folding(&mut self) -> Result<()> {
        // Evaluate operations with constant inputs at compile time
        let nodes_copy = self.nodes.clone();
        for node in &mut self.nodes {
            let all_const = node.inputs.iter().all(|&id| {
                nodes_copy.iter()
                    .find(|n| n.id == id)
                    .map(|n| matches!(n.op, GraphOp::Constant { .. }))
                    .unwrap_or(false)
            });

            if all_const && matches!(node.op, GraphOp::Add | GraphOp::Mul) {
                // Fold constant operations
                // In practice, compute the result and replace with Constant node
            }
        }

        Ok(())
    }

    /// Convert back to MIR
    pub fn to_mir(&self) -> Result<MirProgram> {
        // Topological sort of nodes
        let sorted = self.topological_sort()?;

        let mut stmts = Vec::new();

        for node_id in sorted {
            let node = &self.nodes[node_id];
            let var_name = format!("v{}", node.id);

            let expr = match &node.op {
                GraphOp::MatMul => {
                    if node.inputs.len() >= 2 {
                        MirExpr::Call {
                            func: "matmul".to_string(),
                            args: vec![
                                MirExpr::Var { name: format!("v{}", node.inputs[0]), var_id: node.inputs[0] },
                                MirExpr::Var { name: format!("v{}", node.inputs[1]), var_id: node.inputs[1] },
                            ],
                        }
                    } else {
                        continue;
                    }
                }
                GraphOp::ReLU => {
                    if node.inputs.len() >= 1 {
                        MirExpr::Call {
                            func: "relu".to_string(),
                            args: vec![
                                MirExpr::Var { name: format!("v{}", node.inputs[0]), var_id: node.inputs[0] },
                            ],
                        }
                    } else {
                        continue;
                    }
                }
                GraphOp::Add => {
                    if node.inputs.len() >= 2 {
                        MirExpr::BinOp {
                            op: crate::hir::HirBinOp::Add,
                            left: Box::new(MirExpr::Var { name: format!("v{}", node.inputs[0]), var_id: node.inputs[0] }),
                            right: Box::new(MirExpr::Var { name: format!("v{}", node.inputs[1]), var_id: node.inputs[1] }),
                        }
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };

            stmts.push(MirStmt::Assign {
                name: var_name,
                value: expr,
                var_id: node.id,
            });
        }

        Ok(MirProgram {
            models: vec![],
            functions: vec![MirFunction {
                name: "optimized".to_string(),
                params: vec![],
                return_type: None,
                body: stmts,
            }],
        })
    }

    fn topological_sort(&self) -> Result<Vec<usize>> {
        let mut result = Vec::new();
        let mut visited = vec![false; self.nodes.len()];
        let mut temp_mark = vec![false; self.nodes.len()];

        for node in &self.nodes {
            if !visited[node.id] {
                self.visit(node.id, &mut visited, &mut temp_mark, &mut result)?;
            }
        }

        result.reverse();
        Ok(result)
    }

    fn visit(
        &self,
        node_id: usize,
        visited: &mut [bool],
        temp_mark: &mut [bool],
        result: &mut Vec<usize>,
    ) -> Result<()> {
        if temp_mark[node_id] {
            return Err(LumenError::CodegenError {
                message: "Cycle detected in graph".to_string(),
            });
        }

        if visited[node_id] {
            return Ok(());
        }

        temp_mark[node_id] = true;

        if let Some(node) = self.nodes.iter().find(|n| n.id == node_id) {
            for &input_id in &node.inputs {
                self.visit(input_id, visited, temp_mark, result)?;
            }
        }

        temp_mark[node_id] = false;
        visited[node_id] = true;
        result.push(node_id);

        Ok(())
    }
}

