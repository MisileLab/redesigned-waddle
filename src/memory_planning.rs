// Memory Planning & Optimization
// Static memory allocation and in-place operations

use crate::mir::*;
use crate::error::{Result, LumenError};
use std::collections::{HashMap, HashSet};

/// Memory allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Naive: allocate new buffer for each tensor
    Naive,
    /// Reuse: reuse buffers when possible
    Reuse,
    /// InPlace: prefer in-place operations
    InPlace,
    /// Optimal: use graph coloring for minimal memory
    Optimal,
}

/// Tensor buffer allocation
#[derive(Debug, Clone)]
pub struct BufferAllocation {
    pub tensor_name: String,
    pub buffer_id: usize,
    pub offset: usize,
    pub size: usize,
    pub dtype: String,
}

/// Memory planner
pub struct MemoryPlanner {
    strategy: AllocationStrategy,
    /// Buffer allocations
    allocations: HashMap<String, BufferAllocation>,
    /// Live ranges for each tensor
    live_ranges: HashMap<String, (usize, usize)>,  // (first_use, last_use)
    /// Buffer pool
    buffers: Vec<usize>,  // Buffer sizes
}

impl MemoryPlanner {
    pub fn new(strategy: AllocationStrategy) -> Self {
        Self {
            strategy,
            allocations: HashMap::new(),
            live_ranges: HashMap::new(),
            buffers: Vec::new(),
        }
    }

    /// Plan memory allocation for a program
    pub fn plan(&mut self, program: &MirProgram) -> Result<()> {
        // 1. Analyze liveness
        self.analyze_liveness(program)?;

        // 2. Allocate buffers based on strategy
        match self.strategy {
            AllocationStrategy::Naive => self.allocate_naive(program)?,
            AllocationStrategy::Reuse => self.allocate_reuse(program)?,
            AllocationStrategy::InPlace => self.allocate_inplace(program)?,
            AllocationStrategy::Optimal => self.allocate_optimal(program)?,
        }

        Ok(())
    }

    /// Analyze tensor liveness
    fn analyze_liveness(&mut self, program: &MirProgram) -> Result<()> {
        for func in &program.functions {
            let mut position = 0;

            for stmt in &func.body {
                match stmt {
                    MirStmt::Assign { name, value, .. } => {
                        // Define at this position
                        self.live_ranges.entry(name.clone())
                            .or_insert((position, position))
                            .0 = position;

                        // Update uses
                        self.mark_uses(value, position);

                        position += 1;
                    }
                    MirStmt::Return { value } => {
                        if let Some(v) = value {
                            self.mark_uses(v, position);
                        }
                        position += 1;
                    }
                }
            }
        }

        Ok(())
    }

    fn mark_uses(&mut self, expr: &MirExpr, position: usize) {
        match expr {
            MirExpr::Var { name, .. } => {
                self.live_ranges.entry(name.clone())
                    .and_modify(|range| range.1 = range.1.max(position));
            }
            MirExpr::Call { args, .. } => {
                for arg in args {
                    self.mark_uses(arg, position);
                }
            }
            MirExpr::BinOp { left, right, .. } => {
                self.mark_uses(left, position);
                self.mark_uses(right, position);
            }
            _ => {}
        }
    }

    /// Naive allocation: one buffer per tensor
    fn allocate_naive(&mut self, program: &MirProgram) -> Result<()> {
        for func in &program.functions {
            for stmt in &func.body {
                if let MirStmt::Assign { name, var_type, .. } = stmt {
                    let buffer_id = self.buffers.len();
                    self.buffers.push(size);

                    self.allocations.insert(name.clone(), BufferAllocation {
                        tensor_name: name.clone(),
                        buffer_id,
                        offset: 0,
                        size,
                        dtype: var_type.clone(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Reuse allocation: reuse buffers for non-overlapping tensors
    fn allocate_reuse(&mut self, _program: &MirProgram) -> Result<()> {
        // Sort tensors by live range
        let mut tensors: Vec<_> = self.live_ranges.iter().collect();
        tensors.sort_by_key(|(_, range)| range.0);

        // Greedy allocation
        for (tensor_name, &(start, end)) in tensors {
            let size = self.estimate_size("f32");

            // Find a buffer that's free during this range
            let mut buffer_id = None;

            for (bid, &buf_size) in self.buffers.iter().enumerate() {
                if buf_size >= size {
                    // Check if any tensor using this buffer overlaps
                    let overlaps = self.allocations.values().any(|alloc| {
                        if alloc.buffer_id != bid {
                            return false;
                        }

                        if let Some(&(other_start, other_end)) = self.live_ranges.get(&alloc.tensor_name) {
                            // Check overlap
                            !(end < other_start || start > other_end)
                        } else {
                            false
                        }
                    });

                    if !overlaps {
                        buffer_id = Some(bid);
                        break;
                    }
                }
            }

            // Create new buffer if needed
            let bid = buffer_id.unwrap_or_else(|| {
                let id = self.buffers.len();
                self.buffers.push(size);
                id
            });

            self.allocations.insert(tensor_name.clone(), BufferAllocation {
                tensor_name: tensor_name.clone(),
                buffer_id: bid,
                offset: 0,
                size,
                dtype: "f32".to_string(),
            });
        }

        Ok(())
    }

    /// In-place allocation: identify operations that can be done in-place
    fn allocate_inplace(&mut self, program: &MirProgram) -> Result<()> {
        // First do reuse allocation
        self.allocate_reuse(program)?;

        // Then identify in-place opportunities
        for func in &program.functions {
            for stmt in &func.body {
                if let MirStmt::Assign { name, value, .. } = stmt {
                    // Check if operation can be in-place
                    if self.is_inplace_op(value) {
                        if let Some(input_name) = self.get_inplace_input(value) {
                            // Reuse input buffer
                            if let Some(input_alloc) = self.allocations.get(&input_name).cloned() {
                                self.allocations.insert(name.clone(), BufferAllocation {
                                    tensor_name: name.clone(),
                                    buffer_id: input_alloc.buffer_id,
                                    offset: input_alloc.offset,
                                    size: input_alloc.size,
                                    dtype: input_alloc.dtype,
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn is_inplace_op(&self, expr: &MirExpr) -> bool {
        match expr {
            MirExpr::Call { func, .. } => {
                // Operations that can be done in-place
                matches!(func.as_str(), "relu" | "sigmoid" | "tanh")
            }
            _ => false,
        }
    }

    fn get_inplace_input(&self, expr: &MirExpr) -> Option<String> {
        match expr {
            MirExpr::Call { args, .. } => {
                if args.len() == 1 {
                    if let MirExpr::Var { name, .. } = &args[0] {
                        return Some(name.clone());
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Optimal allocation using graph coloring
    fn allocate_optimal(&mut self, program: &MirProgram) -> Result<()> {
        // Build interference graph
        let mut interference: HashMap<String, HashSet<String>> = HashMap::new();

        for (t1, &(start1, end1)) in &self.live_ranges {
            for (t2, &(start2, end2)) in &self.live_ranges {
                if t1 != t2 {
                    // Check if ranges overlap
                    if !(end1 < start2 || start1 > end2) {
                        interference.entry(t1.clone())
                            .or_insert_with(HashSet::new)
                            .insert(t2.clone());
                    }
                }
            }
        }

        // Graph coloring (greedy)
        let mut colors: HashMap<String, usize> = HashMap::new();
        let mut tensors: Vec<_> = self.live_ranges.keys().cloned().collect();

        // Sort by degree (most constrained first)
        tensors.sort_by_key(|t| {
            std::cmp::Reverse(interference.get(t).map(|s| s.len()).unwrap_or(0))
        });

        for tensor in tensors {
            // Find smallest color not used by neighbors
            let neighbor_colors: HashSet<usize> = interference
                .get(&tensor)
                .map(|neighbors| {
                    neighbors.iter()
                        .filter_map(|n| colors.get(n).copied())
                        .collect()
                })
                .unwrap_or_default();

            let mut color = 0;
            while neighbor_colors.contains(&color) {
                color += 1;
            }

            colors.insert(tensor.clone(), color);
        }

        // Create buffers for each color
        let max_color = colors.values().max().copied().unwrap_or(0);
        for _ in 0..=max_color {
            self.buffers.push(4096); // Default size
        }

        // Assign allocations
        for func in &program.functions {
            for stmt in &func.body {
                if let MirStmt::Assign { name, var_type, .. } = stmt {
                    if let Some(&color) = colors.get(name) {

                        self.allocations.insert(name.clone(), BufferAllocation {
                            tensor_name: name.clone(),
                            buffer_id: color,
                            offset: 0,
                            size,
                            dtype: var_type.clone(),
                        });

                        // Update buffer size if needed
                        if self.buffers[color] < size {
                            self.buffers[color] = size;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn estimate_size(&self, dtype: &str) -> usize {
        // Simplified size estimation
        match dtype {
            "f32" => 4096,  // Placeholder
            "i32" => 4096,
            _ => 4096,
        }
    }

    /// Get total memory usage
    pub fn total_memory(&self) -> usize {
        self.buffers.iter().sum()
    }

    /// Get number of buffers
    pub fn num_buffers(&self) -> usize {
        self.buffers.len()
    }

    /// Generate memory allocation report
    pub fn report(&self) -> String {
        let mut report = String::from("=== Memory Allocation Report ===\n\n");

        report.push_str(&format!("Strategy: {:?}\n", self.strategy));
        report.push_str(&format!("Total memory: {:.2} MB\n", self.total_memory() as f64 / 1024.0 / 1024.0));
        report.push_str(&format!("Number of buffers: {}\n\n", self.num_buffers()));

        report.push_str("Buffer allocation:\n");
        for (i, &size) in self.buffers.iter().enumerate() {
            report.push_str(&format!("  Buffer {}: {:.2} KB\n", i, size as f64 / 1024.0));
        }

        report.push('\n');
        report.push_str("Tensor allocations:\n");
        let mut allocs: Vec<_> = self.allocations.values().collect();
        allocs.sort_by_key(|a| a.buffer_id);

        for alloc in allocs {
            report.push_str(&format!("  {}: buffer={}, offset={}, size={:.2}KB\n",
                alloc.tensor_name,
                alloc.buffer_id,
                alloc.offset,
                alloc.size as f64 / 1024.0
            ));
        }

        report
    }
}

