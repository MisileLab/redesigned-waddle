// Mixed Precision Training Support
// FP16, BF16, and automatic mixed precision

use crate::mir::*;
use crate::hir::*;
use crate::error::Result;
use std::collections::HashMap;

/// Precision types supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    FP32,    // 32-bit floating point
    FP16,    // 16-bit floating point (IEEE 754)
    BF16,    // Brain float 16
    TF32,    // TensorFloat-32 (NVIDIA Ampere+)
    FP64,    // 64-bit double precision
}

impl Precision {
    pub fn bits(&self) -> usize {
        match self {
            Precision::FP16 | Precision::BF16 | Precision::TF32 => 16,
            Precision::FP32 => 32,
            Precision::FP64 => 64,
        }
    }

    pub fn to_string(&self) -> &'static str {
        match self {
            Precision::FP32 => "float32",
            Precision::FP16 => "float16",
            Precision::BF16 => "bfloat16",
            Precision::TF32 => "tf32",
            Precision::FP64 => "float64",
        }
    }

    pub fn to_torch_dtype(&self) -> &'static str {
        match self {
            Precision::FP32 => "torch.float32",
            Precision::FP16 => "torch.float16",
            Precision::BF16 => "torch.bfloat16",
            Precision::TF32 => "torch.float32",  // TF32 is used via matmul precision
            Precision::FP64 => "torch.float64",
        }
    }
}

/// Automatic Mixed Precision (AMP) configuration
#[derive(Debug, Clone)]
pub struct AMPConfig {
    /// Enable automatic mixed precision
    pub enabled: bool,

    /// Compute precision (typically FP16 or BF16)
    pub compute_dtype: Precision,

    /// Parameter storage precision (typically FP32)
    pub param_dtype: Precision,

    /// Use dynamic loss scaling
    pub use_loss_scaling: bool,

    /// Initial loss scale
    pub init_scale: f32,

    /// Growth factor for loss scale
    pub growth_factor: f32,

    /// Backoff factor for loss scale
    pub backoff_factor: f32,

    /// Growth interval (steps)
    pub growth_interval: usize,

    /// Operations to keep in FP32
    pub fp32_ops: Vec<String>,
}

impl Default for AMPConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            compute_dtype: Precision::FP16,
            param_dtype: Precision::FP32,
            use_loss_scaling: true,
            init_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            fp32_ops: vec![
                "softmax".to_string(),
                "log_softmax".to_string(),
                "cross_entropy".to_string(),
                "layer_norm".to_string(),
                "batch_norm".to_string(),
            ],
        }
    }
}

impl AMPConfig {
    pub fn fp16() -> Self {
        Self {
            compute_dtype: Precision::FP16,
            ..Default::default()
        }
    }

    pub fn bf16() -> Self {
        Self {
            compute_dtype: Precision::BF16,
            use_loss_scaling: false,  // BF16 doesn't need loss scaling
            ..Default::default()
        }
    }

    /// Check if operation should use FP32
    pub fn should_use_fp32(&self, op: &str) -> bool {
        self.fp32_ops.contains(&op.to_string())
    }
}

/// Mixed precision optimizer wrapper
pub struct MixedPrecisionOptimizer {
    config: AMPConfig,
    loss_scale: f32,
    growth_tracker: usize,
    master_params: HashMap<String, String>,  // param_name -> master_copy_name
}

impl MixedPrecisionOptimizer {
    pub fn new(config: AMPConfig) -> Self {
        Self {
            loss_scale: config.init_scale,
            config,
            growth_tracker: 0,
            master_params: HashMap::new(),
        }
    }

    /// Generate PyTorch AMP code
    pub fn generate_pytorch_amp(&self) -> String {
        let dtype = self.config.compute_dtype.to_torch_dtype();

        format!(
            r#"
# Automatic Mixed Precision Setup
scaler = torch.cuda.amp.GradScaler(
    init_scale={init_scale},
    growth_factor={growth_factor},
    backoff_factor={backoff_factor},
    growth_interval={growth_interval},
    enabled={enabled}
)

# Training step with AMP
def train_step(model, optimizer, data, target):
    with torch.cuda.amp.autocast(dtype={dtype}):
        output = model(data)
        loss = criterion(output, target)

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()

    # Optimizer step with unscaling
    scaler.step(optimizer)
    scaler.update()

    return loss.item()
"#,
            init_scale = self.config.init_scale,
            growth_factor = self.config.growth_factor,
            backoff_factor = self.config.backoff_factor,
            growth_interval = self.config.growth_interval,
            enabled = if self.config.enabled {
                "True"
            } else {
                "False"
            },
            dtype = dtype
        )
    }

    /// Generate CUDA/HIP mixed precision kernel dispatch
    pub fn generate_cuda_dispatch(&self, op: &str) -> String {
        let dtype = match self.config.compute_dtype {
            Precision::FP16 => "__half",
            Precision::BF16 => "__nv_bfloat16",
            Precision::FP32 => "float",
            _ => "float",
        };

        format!(
            r#"
// Mixed precision kernel dispatch
template<typename T = {dtype}>
void {op}_kernel_dispatch(const T* input, T* output, int n) {{
    // Dispatch to appropriate precision kernel
    {op}_kernel<T><<<blocks, threads>>>(input, output, n);
}}
"#,
            dtype = dtype,
            op = op
        )
    }

    /// Check if operation should use FP32
    pub fn should_use_fp32(&self, op: &str) -> bool {
        self.config.fp32_ops.contains(&op.to_string())
    }

    /// Update loss scale after optimizer step
    pub fn update_loss_scale(&mut self, overflow_detected: bool) {
        if !self.config.use_loss_scaling {
            return;
        }

        if overflow_detected {
            self.loss_scale *= self.config.backoff_factor;
            self.growth_tracker = 0;
        } else {
            self.growth_tracker += 1;
            if self.growth_tracker >= self.config.growth_interval {
                self.loss_scale *= self.config.growth_factor;
                self.growth_tracker = 0;
            }
        }
    }
}

/// Precision analysis pass - determines optimal precision for each operation
pub struct PrecisionAnalyzer {
    op_precisions: HashMap<String, Precision>,
}

impl PrecisionAnalyzer {
    pub fn new() -> Self {
        Self {
            op_precisions: HashMap::new(),
        }
    }

    /// Analyze program and assign precisions
    pub fn analyze(&mut self, program: &MirProgram, config: &AMPConfig) -> Result<()> {
        for func in &program.functions {
            for stmt in &func.body {
                if let MirStmt::Assign { name, value, .. } = stmt {
                    let precision = self.infer_precision(value, config);
                    self.op_precisions.insert(name.clone(), precision);
                }
            }
        }

        Ok(())
    }

    fn infer_precision(&self, expr: &MirExpr, config: &AMPConfig) -> Precision {
        match expr {
            MirExpr::Call { func, .. } => {
                if config.should_use_fp32(func) {
                    config.param_dtype
                } else {
                    config.compute_dtype
                }
            }
            MirExpr::BinOp { .. } => config.compute_dtype,
            _ => config.param_dtype,
        }
    }

    pub fn get_precision(&self, var: &str) -> Option<Precision> {
        self.op_precisions.get(var).copied()
    }
}

/// Cast insertion pass - adds explicit casts between precisions
pub struct CastInsertion {
    casts: Vec<(String, Precision, Precision)>,  // (var, from, to)
}

impl CastInsertion {
    pub fn new() -> Self {
        Self { casts: Vec::new() }
    }

    /// Insert casts where precision changes
    pub fn insert_casts(&mut self, _program: &mut MirProgram, _analyzer: &PrecisionAnalyzer) -> Result<()> {
        // Scan for precision mismatches and insert casts
        // Implementation details...

        Ok(())
    }

    /// Generate cast operation
    pub fn generate_cast(&self, var: &str, from: Precision, to: Precision) -> String {
        format!("{} = {}.to({})", var, var, to.to_torch_dtype())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_types() {
        assert_eq!(Precision::FP16.bits(), 16);
        assert_eq!(Precision::FP32.bits(), 32);
        assert_eq!(Precision::BF16.to_string(), "bfloat16");
    }

    #[test]
    fn test_amp_config() {
        let config = AMPConfig::fp16();
        assert_eq!(config.compute_dtype, Precision::FP16);
        assert!(config.use_loss_scaling);

        let bf16_config = AMPConfig::bf16();
        assert_eq!(bf16_config.compute_dtype, Precision::BF16);
        assert!(!bf16_config.use_loss_scaling);
    }

    #[test]
    fn test_mixed_precision_optimizer() {
        let config = AMPConfig::fp16();
        let optimizer = MixedPrecisionOptimizer::new(config);

        let code = optimizer.generate_pytorch_amp();
        assert!(code.contains("GradScaler"));
        assert!(code.contains("autocast"));
    }

    #[test]
    fn test_precision_analyzer() {
        let analyzer = PrecisionAnalyzer::new();
        assert!(analyzer.op_precisions.is_empty());
    }
}
