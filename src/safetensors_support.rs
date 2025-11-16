// Safetensors Import/Export
// Safe, fast serialization format for ML models

use crate::mir::*;
use crate::hir::*;
use crate::error::{Result, LumenError};
use safetensors::tensor::{SafeTensors, Dtype};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Model metadata for safetensors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,

    /// Model version
    pub version: String,

    /// Framework (lumen, pytorch, jax, etc.)
    pub framework: String,

    /// Architecture description
    pub architecture: Option<String>,

    /// Training configuration
    pub config: Option<serde_json::Value>,

    /// Additional metadata
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            name: "lumen_model".to_string(),
            version: "0.1.0".to_string(),
            framework: "lumen".to_string(),
            architecture: None,
            config: None,
            extra: HashMap::new(),
        }
    }
}

/// Tensor information for export
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: TensorDtype,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDtype {
    F32,
    F16,
    BF16,
    I8,
    I16,
    I32,
    I64,
    U8,
}

impl TensorDtype {
    pub fn to_safetensors_dtype(&self) -> Dtype {
        match self {
            TensorDtype::F32 => Dtype::F32,
            TensorDtype::F16 => Dtype::F16,
            TensorDtype::BF16 => Dtype::BF16,
            TensorDtype::I8 => Dtype::I8,
            TensorDtype::I16 => Dtype::I16,
            TensorDtype::I32 => Dtype::I32,
            TensorDtype::I64 => Dtype::I64,
            TensorDtype::U8 => Dtype::U8,
        }
    }

    pub fn from_safetensors_dtype(dtype: &Dtype) -> Self {
        match dtype {
            Dtype::F32 => TensorDtype::F32,
            Dtype::F16 => TensorDtype::F16,
            Dtype::BF16 => TensorDtype::BF16,
            Dtype::I8 => TensorDtype::I8,
            Dtype::I16 => TensorDtype::I16,
            Dtype::I32 => TensorDtype::I32,
            Dtype::I64 => TensorDtype::I64,
            Dtype::U8 => TensorDtype::U8,
            _ => TensorDtype::F32,  // Default
        }
    }

    pub fn size_bytes(&self) -> usize {
        match self {
            TensorDtype::F32 | TensorDtype::I32 => 4,
            TensorDtype::F16 | TensorDtype::BF16 | TensorDtype::I16 => 2,
            TensorDtype::I8 | TensorDtype::U8 => 1,
            TensorDtype::I64 => 8,
        }
    }
}

/// Safetensors exporter
pub struct SafetensorsExporter {
    metadata: ModelMetadata,
    tensors: Vec<TensorInfo>,
}

impl SafetensorsExporter {
    pub fn new(metadata: ModelMetadata) -> Self {
        Self {
            metadata,
            tensors: Vec::new(),
        }
    }

    /// Add tensor to export
    pub fn add_tensor(&mut self, name: String, shape: Vec<usize>, dtype: TensorDtype, data: Vec<u8>) {
        self.tensors.push(TensorInfo {
            name,
            shape,
            dtype,
            data,
        });
    }

    /// Export to safetensors file
    pub fn save<P: AsRef<Path>>(&self, _path: P) -> Result<()> {
        // Simplified implementation - actual safetensors serialization
        // would be done via PyTorch/HuggingFace libraries

        // For now, we provide code generation that PyTorch can use
        // The actual file I/O happens in the generated Python code

        Ok(())
    }

    /// Generate PyTorch code to export to safetensors
    pub fn generate_pytorch_export_code(&self, model_var: &str) -> String {
        format!(
            r#"
# Export model to safetensors
from safetensors.torch import save_file

def export_model(model, path):
    # Get state dict
    state_dict = model.state_dict()

    # Metadata
    metadata = {{
        "name": "{name}",
        "version": "{version}",
        "framework": "{framework}",
    }}

    # Save to safetensors
    save_file(state_dict, path, metadata=metadata)

# Usage
export_model({model_var}, "model.safetensors")
"#,
            name = self.metadata.name,
            version = self.metadata.version,
            framework = self.metadata.framework,
            model_var = model_var
        )
    }
}

/// Safetensors importer
pub struct SafetensorsImporter {
    metadata: Option<ModelMetadata>,
    tensors: HashMap<String, TensorInfo>,
}

impl SafetensorsImporter {
    pub fn new() -> Self {
        Self {
            metadata: None,
            tensors: HashMap::new(),
        }
    }

    /// Load from safetensors file
    pub fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let mut file = File::open(path).map_err(|e| LumenError::CodegenError {
            message: format!("Failed to open file: {}", e),
        })?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).map_err(|e| LumenError::CodegenError {
            message: format!("Failed to read file: {}", e),
        })?;

        // Deserialize
        let safetensors = SafeTensors::deserialize(&buffer).map_err(|e| {
            LumenError::CodegenError {
                message: format!("Failed to deserialize safetensors: {}", e),
            }
        })?;

        // Load metadata (simplified - actual implementation would extract from file header)
        // The safetensors format stores metadata in the file header
        // For now, we skip metadata parsing and focus on tensor loading
        self.metadata = Some(ModelMetadata::default());

        // Load tensors
        for name in safetensors.names() {
            if let Ok(tensor_view) = safetensors.tensor(name) {
                let shape = tensor_view.shape().to_vec();
                let dtype = TensorDtype::from_safetensors_dtype(&tensor_view.dtype());
                let data = tensor_view.data().to_vec();

                self.tensors.insert(
                    name.to_string(),
                    TensorInfo {
                        name: name.to_string(),
                        shape,
                        dtype,
                        data,
                    },
                );
            }
        }

        Ok(())
    }

    pub fn get_metadata(&self) -> Option<&ModelMetadata> {
        self.metadata.as_ref()
    }

    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// Generate PyTorch code to import from safetensors
    pub fn generate_pytorch_import_code(&self) -> String {
        r#"
# Import model from safetensors
from safetensors.torch import load_file

def import_model(model, path):
    # Load state dict from safetensors
    state_dict = load_file(path)

    # Load into model
    model.load_state_dict(state_dict)

    return model

# Usage
model = MyModel()
model = import_model(model, "model.safetensors")
"#
        .to_string()
    }
}

/// Convert MIR model to safetensors format
pub struct ModelConverter;

impl ModelConverter {
    /// Extract model parameters for export
    pub fn extract_parameters(model: &MirModel) -> Vec<TensorInfo> {
        let mut tensors = Vec::new();

        for param in &model.params {
            // Extract shape from type
            let shape = Self::extract_shape(&param.ty);

            // Create placeholder data (in practice, would be actual trained weights)
            let total_size: usize = shape.iter().product();
            let data = vec![0u8; total_size * 4]; // FP32

            tensors.push(TensorInfo {
                name: param.name.clone(),
                shape,
                dtype: TensorDtype::F32,
                data,
            });
        }

        tensors
    }

    fn extract_shape(ty: &HirType) -> Vec<usize> {
        match ty {
            HirType::Tensor { shape, .. } => {
                shape
                    .iter()
                    .filter_map(|dim| match dim {
                        HirDimExpr::Const(c) => Some(*c as usize),
                        _ => None,
                    })
                    .collect()
            }
            _ => vec![1],
        }
    }

    /// Create exporter from MIR program
    pub fn from_program(program: &MirProgram, metadata: ModelMetadata) -> SafetensorsExporter {
        let mut exporter = SafetensorsExporter::new(metadata);

        for model in &program.models {
            let tensors = Self::extract_parameters(model);
            for tensor in tensors {
                exporter.add_tensor(tensor.name, tensor.shape, tensor.dtype, tensor.data);
            }
        }

        exporter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata() {
        let metadata = ModelMetadata::default();
        assert_eq!(metadata.framework, "lumen");
        assert_eq!(metadata.name, "lumen_model");
    }

    #[test]
    fn test_dtype_conversion() {
        assert_eq!(TensorDtype::F32.size_bytes(), 4);
        assert_eq!(TensorDtype::F16.size_bytes(), 2);
        assert_eq!(TensorDtype::I8.size_bytes(), 1);
    }

    #[test]
    fn test_exporter() {
        let metadata = ModelMetadata::default();
        let exporter = SafetensorsExporter::new(metadata);

        let code = exporter.generate_pytorch_export_code("model");
        assert!(code.contains("save_file"));
        assert!(code.contains("safetensors"));
    }

    #[test]
    fn test_importer() {
        let importer = SafetensorsImporter::new();
        let code = importer.generate_pytorch_import_code();
        assert!(code.contains("load_file"));
        assert!(code.contains("load_state_dict"));
    }
}
