// Profiling & Debugging Tools
// Performance analysis and visualization

use crate::mir::*;
use crate::error::Result;
use std::collections::HashMap;
use std::time::Duration;

/// Operation profile information
#[derive(Debug, Clone)]
pub struct OpProfile {
    pub op_name: String,
    pub count: usize,
    pub total_time_us: u64,
    pub avg_time_us: f64,
    pub memory_bytes: usize,
    pub flops: u64,
}

/// Profiler for analyzing model performance
pub struct Profiler {
    /// Per-operation profiles
    pub op_profiles: HashMap<String, OpProfile>,
    /// Total execution time
    pub total_time: Duration,
    /// Peak memory usage
    pub peak_memory: usize,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            op_profiles: HashMap::new(),
            total_time: Duration::ZERO,
            peak_memory: 0,
        }
    }

    /// Record an operation
    pub fn record_op(&mut self, op_name: String, time_us: u64, memory_bytes: usize, flops: u64) {
        self.op_profiles
            .entry(op_name.clone())
            .and_modify(|p| {
                p.count += 1;
                p.total_time_us += time_us;
                p.avg_time_us = p.total_time_us as f64 / p.count as f64;
                p.memory_bytes += memory_bytes;
                p.flops += flops;
            })
            .or_insert(OpProfile {
                op_name,
                count: 1,
                total_time_us: time_us,
                avg_time_us: time_us as f64,
                memory_bytes,
                flops,
            });

        self.peak_memory = self.peak_memory.max(memory_bytes);
    }

    /// Generate profiling report
    pub fn report(&self) -> String {
        let mut report = String::from("=== Lumen Profiling Report ===\n\n");

        report.push_str(&format!("Total execution time: {:.2}ms\n", self.total_time.as_micros() as f64 / 1000.0));
        report.push_str(&format!("Peak memory usage: {:.2}MB\n\n", self.peak_memory as f64 / 1024.0 / 1024.0));

        report.push_str("Operation breakdown:\n");
        report.push_str(&format!("{:<20} {:>10} {:>15} {:>15} {:>15}\n",
            "Operation", "Count", "Total (ms)", "Avg (us)", "GFLOPS"));
        report.push_str(&"-".repeat(75));
        report.push('\n');

        let mut ops: Vec<_> = self.op_profiles.values().collect();
        ops.sort_by(|a, b| b.total_time_us.cmp(&a.total_time_us));

        for op in ops {
            let gflops = (op.flops as f64 / op.total_time_us as f64) / 1000.0;
            report.push_str(&format!("{:<20} {:>10} {:>15.2} {:>15.2} {:>15.2}\n",
                op.op_name,
                op.count,
                op.total_time_us as f64 / 1000.0,
                op.avg_time_us,
                gflops
            ));
        }

        report.push('\n');
        report
    }

    /// Generate flamegraph data
    pub fn flamegraph(&self) -> String {
        let mut data = String::new();

        for op in self.op_profiles.values() {
            data.push_str(&format!("{};{} {}\n",
                "model",
                op.op_name,
                op.total_time_us
            ));
        }

        data
    }

    /// Estimate FLOPs for operations
    pub fn estimate_flops(op: &str, shapes: &[Vec<i64>]) -> u64 {
        match op {
            "matmul" => {
                if shapes.len() >= 2 {
                    let m = shapes[0][0];
                    let k = shapes[0][1];
                    let n = shapes[1][1];
                    // 2MNK for matmul (multiply-add)
                    (2 * m * n * k) as u64
                } else {
                    0
                }
            }
            "conv2d" => {
                // Approximate: C_out * H_out * W_out * K_h * K_w * C_in
                if shapes.len() >= 1 {
                    let total: i64 = shapes[0].iter().product();
                    (2 * total) as u64
                } else {
                    0
                }
            }
            _ => 0,
        }
    }
}

/// Memory usage analyzer
pub struct MemoryAnalyzer {
    /// Tensor sizes
    pub tensor_sizes: HashMap<String, usize>,
    /// Live tensors at each point
    pub liveness: Vec<Vec<String>>,
}

impl MemoryAnalyzer {
    pub fn new() -> Self {
        Self {
            tensor_sizes: HashMap::new(),
            liveness: Vec::new(),
        }
    }

    /// Analyze memory usage from MIR
    pub fn analyze(&mut self, program: &MirProgram) -> Result<()> {
        for func in &program.functions {
            for stmt in &func.body {
                if let MirStmt::Assign { name, .. } = stmt {
                    // Estimate size (simplified)
                    let size = 1024; // Placeholder
                    self.tensor_sizes.insert(name.clone(), size);
                }
            }
        }

        Ok(())
    }

    /// Get total memory usage
    pub fn total_memory(&self) -> usize {
        self.tensor_sizes.values().sum()
    }

    /// Get peak memory (considering liveness)
    pub fn peak_memory(&self) -> usize {
        self.liveness.iter()
            .map(|live| {
                live.iter()
                    .filter_map(|name| self.tensor_sizes.get(name))
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0)
    }

    /// Generate memory timeline
    pub fn memory_timeline(&self) -> String {
        let mut timeline = String::from("Memory usage over time:\n\n");

        for (i, live) in self.liveness.iter().enumerate() {
            let usage: usize = live.iter()
                .filter_map(|name| self.tensor_sizes.get(name))
                .sum();

            timeline.push_str(&format!("Step {}: {:.2}MB (tensors: {})\n",
                i,
                usage as f64 / 1024.0 / 1024.0,
                live.len()
            ));
        }

        timeline
    }
}

/// Model visualization
pub struct Visualizer {
    /// Dot graph representation
    pub graph: String,
}

impl Visualizer {
    pub fn new() -> Self {
        Self {
            graph: String::new(),
        }
    }

    /// Generate GraphViz dot format
    pub fn visualize(&mut self, program: &MirProgram) -> Result<String> {
        self.graph = String::from("digraph G {\n");
        self.graph.push_str("  rankdir=TB;\n");
        self.graph.push_str("  node [shape=box];\n\n");

        let mut node_id = 0;
        let mut var_to_id: HashMap<String, usize> = HashMap::new();

        for func in &program.functions {
            for stmt in &func.body {
                if let MirStmt::Assign { name, value, .. } = stmt {
                    let current_id = node_id;
                    node_id += 1;
                    var_to_id.insert(name.clone(), current_id);

                    // Add node
                    let op_name = self.get_op_name(value);
                    self.graph.push_str(&format!("  n{} [label=\"{}\\n{}\"];\n",
                        current_id, name, op_name));

                    // Add edges from inputs
                    self.add_edges(value, current_id, &var_to_id);
                }
            }
        }

        self.graph.push_str("}\n");
        Ok(self.graph.clone())
    }

    fn get_op_name(&self, expr: &MirExpr) -> String {
        match expr {
            MirExpr::Call { func, .. } => func.clone(),
            MirExpr::BinOp { op, .. } => format!("{:?}", op),
            _ => "expr".to_string(),
        }
    }

    fn add_edges(&mut self, expr: &MirExpr, current_id: usize, var_to_id: &HashMap<String, usize>) {
        match expr {
            MirExpr::Call { args, .. } => {
                for arg in args {
                    if let MirExpr::Var { name, .. } = arg {
                        if let Some(&input_id) = var_to_id.get(name) {
                            self.graph.push_str(&format!("  n{} -> n{};\n", input_id, current_id));
                        }
                    }
                }
            }
            MirExpr::BinOp { left, right, .. } => {
                if let MirExpr::Var { name, .. } = &**left {
                    if let Some(&input_id) = var_to_id.get(name) {
                        self.graph.push_str(&format!("  n{} -> n{};\n", input_id, current_id));
                    }
                }
                if let MirExpr::Var { name, .. } = &**right {
                    if let Some(&input_id) = var_to_id.get(name) {
                        self.graph.push_str(&format!("  n{} -> n{};\n", input_id, current_id));
                    }
                }
            }
            _ => {}
        }
    }

    /// Save to file
    pub fn save(&self, path: &str) -> Result<()> {
        std::fs::write(path, &self.graph).map_err(|e| {
            crate::error::LumenError::CodegenError {
                message: format!("Failed to save visualization: {}", e),
            }
        })
    }
}

/// Performance comparator
pub struct Comparator {
    /// Baseline measurements
    pub baseline: HashMap<String, f64>,
    /// Current measurements
    pub current: HashMap<String, f64>,
}

impl Comparator {
    pub fn new() -> Self {
        Self {
            baseline: HashMap::new(),
            current: HashMap::new(),
        }
    }

    /// Compare current vs baseline
    pub fn compare(&self) -> String {
        let mut report = String::from("=== Performance Comparison ===\n\n");

        report.push_str(&format!("{:<20} {:>15} {:>15} {:>15}\n",
            "Operation", "Baseline (ms)", "Current (ms)", "Speedup"));
        report.push_str(&"-".repeat(65));
        report.push('\n');

        for (op, baseline_time) in &self.baseline {
            if let Some(current_time) = self.current.get(op) {
                let speedup = baseline_time / current_time;
                let indicator = if speedup > 1.0 { "✓" } else { "✗" };

                report.push_str(&format!("{:<20} {:>15.2} {:>15.2} {:>14.2}x {}\n",
                    op, baseline_time, current_time, speedup, indicator));
            }
        }

        report.push('\n');
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_record() {
        let mut profiler = Profiler::new();
        profiler.record_op("matmul".to_string(), 1000, 1024, 1000000);
        profiler.record_op("matmul".to_string(), 1200, 1024, 1000000);

        assert_eq!(profiler.op_profiles.get("matmul").unwrap().count, 2);
        assert_eq!(profiler.op_profiles.get("matmul").unwrap().total_time_us, 2200);
    }

    #[test]
    fn test_flops_estimation() {
        let flops = Profiler::estimate_flops("matmul", &[vec![10, 20], vec![20, 30]]);
        assert_eq!(flops, 2 * 10 * 30 * 20);
    }
}
