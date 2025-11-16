// Quantization Support
// INT8, INT4 quantization for model compression and acceleration

use crate::mir::*;
use crate::error::Result;
use std::collections::HashMap;

/// Quantization schemes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationScheme {
    /// Symmetric quantization
    Symmetric,
    /// Asymmetric quantization (with zero-point)
    Asymmetric,
    /// Per-channel quantization
    PerChannel,
}

/// Quantization bit width
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantBits {
    Int4,
    Int8,
    Int16,
}

impl QuantBits {
    pub fn bits(&self) -> i32 {
        match self {
            QuantBits::Int4 => 4,
            QuantBits::Int8 => 8,
            QuantBits::Int16 => 16,
        }
    }

    pub fn min_value(&self) -> i32 {
        match self {
            QuantBits::Int4 => -8,
            QuantBits::Int8 => -128,
            QuantBits::Int16 => -32768,
        }
    }

    pub fn max_value(&self) -> i32 {
        match self {
            QuantBits::Int4 => 7,
            QuantBits::Int8 => 127,
            QuantBits::Int16 => 32767,
        }
    }
}

/// Quantization configuration
#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub scheme: QuantizationScheme,
    pub bits: QuantBits,
    /// Layers to quantize (empty = all)
    pub quantize_layers: Vec<String>,
    /// Layers to keep in FP32
    pub skip_layers: Vec<String>,
    /// Quantize activations
    pub quantize_activations: bool,
    /// Quantize weights
    pub quantize_weights: bool,
}

impl QuantConfig {
    pub fn int8_default() -> Self {
        Self {
            scheme: QuantizationScheme::Symmetric,
            bits: QuantBits::Int8,
            quantize_layers: vec![],
            skip_layers: vec!["input".to_string(), "output".to_string()],
            quantize_activations: true,
            quantize_weights: true,
        }
    }

    pub fn int4_weights() -> Self {
        Self {
            scheme: QuantizationScheme::Symmetric,
            bits: QuantBits::Int4,
            quantize_layers: vec![],
            skip_layers: vec![],
            quantize_activations: false,
            quantize_weights: true,
        }
    }

    pub fn should_quantize(&self, layer_name: &str) -> bool {
        // Skip if in skip list
        if self.skip_layers.contains(&layer_name.to_string()) {
            return false;
        }

        // If quantize_layers is specified, only quantize those
        if !self.quantize_layers.is_empty() {
            return self.quantize_layers.contains(&layer_name.to_string());
        }

        true
    }
}

/// Quantization parameters for a tensor
#[derive(Debug, Clone)]
pub struct QuantParams {
    pub scale: f32,
    pub zero_point: i32,
    pub min: f32,
    pub max: f32,
}

impl QuantParams {
    /// Compute quantization parameters from data statistics
    pub fn from_minmax(min: f32, max: f32, bits: QuantBits, scheme: QuantizationScheme) -> Self {
        match scheme {
            QuantizationScheme::Symmetric => {
                // Symmetric: zero_point = 0
                let abs_max = min.abs().max(max.abs());
                let scale = abs_max / (bits.max_value() as f32);

                Self {
                    scale,
                    zero_point: 0,
                    min,
                    max,
                }
            }
            QuantizationScheme::Asymmetric => {
                // Asymmetric: full range utilization
                let range = max - min;
                let scale = range / ((bits.max_value() - bits.min_value()) as f32);
                let zero_point = ((-min / scale) + bits.min_value() as f32).round() as i32;

                Self {
                    scale,
                    zero_point,
                    min,
                    max,
                }
            }
            QuantizationScheme::PerChannel => {
                // For now, same as symmetric
                let abs_max = min.abs().max(max.abs());
                let scale = abs_max / (bits.max_value() as f32);

                Self {
                    scale,
                    zero_point: 0,
                    min,
                    max,
                }
            }
        }
    }

    /// Quantize a float value
    pub fn quantize(&self, value: f32, bits: QuantBits) -> i32 {
        let quantized = (value / self.scale).round() as i32 + self.zero_point;
        quantized.clamp(bits.min_value(), bits.max_value())
    }

    /// Dequantize an integer value
    pub fn dequantize(&self, quantized: i32) -> f32 {
        ((quantized - self.zero_point) as f32) * self.scale
    }
}

/// Quantization-aware training (QAT) support
pub struct QuantAwareTraining {
    config: QuantConfig,
    /// Quantization parameters for each layer
    quant_params: HashMap<String, QuantParams>,
}

impl QuantAwareTraining {
    pub fn new(config: QuantConfig) -> Self {
        Self {
            config,
            quant_params: HashMap::new(),
        }
    }

    /// Insert fake quantization operations
    pub fn insert_fake_quant(&mut self, program: &mut MirProgram) -> Result<()> {
        for func in &mut program.functions {
            let mut new_stmts = Vec::new();

            for stmt in &func.body {
                new_stmts.push(stmt.clone());

                // Insert fake quant after matmul/conv operations
                if let MirStmt::Assign { name, value, .. } = stmt {
                    if let MirExpr::Call { func, .. } = value {
                        if func == "matmul" || func == "conv2d" {
                            if self.config.should_quantize(name) {
                                // Add fake quantization
                                let fake_quant = MirStmt::Assign {
                                    name: format!("{}_quant", name),
                                    value: MirExpr::Call {
                                        func: "fake_quantize".to_string(),
                                        args: vec![MirExpr::Var {
                                            name: name.clone(),
                                            var_id: 0,
                                        }],
                                    },
                                    var_id: 0,
                                };
                                new_stmts.push(fake_quant);
                            }
                        }
                    }
                }
            }

            func.body = new_stmts;
        }

        Ok(())
    }

    /// Register quantization parameters for a layer
    pub fn register_quant_params(&mut self, layer_name: String, params: QuantParams) {
        self.quant_params.insert(layer_name, params);
    }

    /// Generate quantized model code
    pub fn generate_quantized_code(&self) -> String {
        let mut code = String::from("// Quantized model\n\n");

        code.push_str(&format!("// Quantization: {} bits, {:?} scheme\n",
            self.config.bits.bits(),
            self.config.scheme
        ));

        code.push_str("struct QuantizedTensor {\n");
        code.push_str("    int8_t* data;\n");
        code.push_str("    float scale;\n");
        code.push_str("    int32_t zero_point;\n");
        code.push_str("};\n\n");

        code.push_str("// Quantized matmul\n");
        code.push_str("Tensor* quantized_matmul(QuantizedTensor* A, QuantizedTensor* B) {\n");
        code.push_str("    // INT8 matrix multiplication\n");
        code.push_str("    // Result = (A_int @ B_int) * scale_A * scale_B\n");
        code.push_str("    // ...\n");
        code.push_str("}\n\n");

        code
    }
}

/// Post-training quantization (PTQ)
pub struct PostTrainingQuant {
    config: QuantConfig,
    /// Calibration data statistics
    activation_stats: HashMap<String, (f32, f32)>,  // (min, max)
}

impl PostTrainingQuant {
    pub fn new(config: QuantConfig) -> Self {
        Self {
            config,
            activation_stats: HashMap::new(),
        }
    }

    /// Calibrate on sample data
    pub fn calibrate(&mut self, layer_name: String, min: f32, max: f32) {
        self.activation_stats.insert(layer_name, (min, max));
    }

    /// Quantize a trained model
    pub fn quantize_model(&self, program: &MirProgram) -> Result<MirProgram> {
        let mut quantized = program.clone();

        for func in &mut quantized.functions {
            for stmt in &mut func.body {
                if let MirStmt::Assign { value, .. } = stmt {
                    // Convert FP32 operations to INT8
                    match value {
                        MirExpr::Call { func, .. } => {
                            if func == "matmul" && self.config.quantize_weights {
                                // Replace with quantized version
                                *var_type = "int8".to_string();
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(quantized)
    }

    /// Get quantization parameters for a layer
    pub fn get_quant_params(&self, layer_name: &str) -> Option<QuantParams> {
        self.activation_stats.get(layer_name).map(|(min, max)| {
            QuantParams::from_minmax(*min, *max, self.config.bits, self.config.scheme)
        })
    }
}

/// GPTQ (Generative Pre-trained Transformer Quantization) style quantization
pub struct GPTQQuant {
    bits: QuantBits,
    block_size: usize,
}

impl GPTQQuant {
    pub fn new(bits: QuantBits) -> Self {
        Self {
            bits,
            block_size: 128,
        }
    }

    /// Quantize weights with Hessian approximation
    pub fn quantize_layer(&self, _weights: &[f32]) -> Vec<i8> {
        // Simplified implementation
        // In practice: use Hessian inverse for optimal quantization
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_params_symmetric() {
        let params = QuantParams::from_minmax(-10.0, 10.0, QuantBits::Int8, QuantizationScheme::Symmetric);
        assert_eq!(params.zero_point, 0);
        assert!((params.scale - 10.0 / 127.0).abs() < 0.01);
    }

    #[test]
    fn test_quantize_dequantize() {
        let params = QuantParams::from_minmax(-1.0, 1.0, QuantBits::Int8, QuantizationScheme::Symmetric);
        let value = 0.5f32;
        let quantized = params.quantize(value, QuantBits::Int8);
        let dequantized = params.dequantize(quantized);
        assert!((value - dequantized).abs() < 0.1);
    }

    #[test]
    fn test_config_should_quantize() {
        let config = QuantConfig::int8_default();
        assert!(!config.should_quantize("input"));
        assert!(config.should_quantize("layer1"));
    }
}
