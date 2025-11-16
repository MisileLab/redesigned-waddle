// Dynamic Shapes Support
// Allows symbolic dimensions like `dim batch = ?`

use crate::error::{Result, LumenError};
use std::collections::HashMap;

/// Symbolic dimension
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolicDim {
    /// Static dimension with known size
    Static(i64),
    /// Dynamic dimension with symbolic name
    Dynamic(String),
    /// Any size (completely unconstrained)
    Any,
}

impl SymbolicDim {
    /// Check if dimension is static
    pub fn is_static(&self) -> bool {
        matches!(self, SymbolicDim::Static(_))
    }

    /// Check if dimension is dynamic
    pub fn is_dynamic(&self) -> bool {
        !self.is_static()
    }

    /// Get static value if available
    pub fn static_value(&self) -> Option<i64> {
        match self {
            SymbolicDim::Static(v) => Some(*v),
            _ => None,
        }
    }
}

/// Symbolic shape with mix of static and dynamic dimensions
#[derive(Debug, Clone)]
pub struct SymbolicShape {
    pub dims: Vec<SymbolicDim>,
}

impl SymbolicShape {
    pub fn new(dims: Vec<SymbolicDim>) -> Self {
        Self { dims }
    }

    /// Create from static dimensions
    pub fn from_static(dims: Vec<i64>) -> Self {
        Self {
            dims: dims.into_iter().map(SymbolicDim::Static).collect(),
        }
    }

    /// Check if shape is fully static
    pub fn is_static(&self) -> bool {
        self.dims.iter().all(|d| d.is_static())
    }

    /// Get static shape if available
    pub fn to_static(&self) -> Option<Vec<i64>> {
        if self.is_static() {
            Some(self.dims.iter().filter_map(|d| d.static_value()).collect())
        } else {
            None
        }
    }

    /// Number of dimensions
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Check shape compatibility for broadcasting
    pub fn is_broadcastable_with(&self, other: &SymbolicShape) -> bool {
        let max_ndim = self.ndim().max(other.ndim());

        for i in 0..max_ndim {
            let dim1 = self.dims.get(self.ndim() - 1 - i);
            let dim2 = other.dims.get(other.ndim() - 1 - i);

            match (dim1, dim2) {
                (None, _) | (_, None) => continue,
                (Some(d1), Some(d2)) => {
                    match (d1, d2) {
                        (SymbolicDim::Static(1), _) | (_, SymbolicDim::Static(1)) => continue,
                        (SymbolicDim::Static(a), SymbolicDim::Static(b)) if a == b => continue,
                        (SymbolicDim::Dynamic(s1), SymbolicDim::Dynamic(s2)) if s1 == s2 => continue,
                        (SymbolicDim::Any, _) | (_, SymbolicDim::Any) => continue,
                        _ => return false,
                    }
                }
            }
        }

        true
    }
}

/// Shape inference engine for dynamic shapes
pub struct DynamicShapeInference {
    /// Mapping from symbolic dimension names to their constraints
    pub symbolic_dims: HashMap<String, SymbolicDimConstraint>,
    /// Mapping from variable names to their symbolic shapes
    pub var_shapes: HashMap<String, SymbolicShape>,
}

#[derive(Debug, Clone)]
pub struct SymbolicDimConstraint {
    pub name: String,
    /// Known value if resolved
    pub value: Option<i64>,
    /// Minimum value constraint
    pub min: Option<i64>,
    /// Maximum value constraint
    pub max: Option<i64>,
}

impl DynamicShapeInference {
    pub fn new() -> Self {
        Self {
            symbolic_dims: HashMap::new(),
            var_shapes: HashMap::new(),
        }
    }

    /// Register a symbolic dimension
    pub fn register_symbolic_dim(&mut self, name: String, constraint: SymbolicDimConstraint) {
        self.symbolic_dims.insert(name, constraint);
    }

    /// Infer output shape for matmul
    pub fn infer_matmul(&self, a_shape: &SymbolicShape, b_shape: &SymbolicShape) -> Result<SymbolicShape> {
        if a_shape.ndim() < 2 || b_shape.ndim() < 2 {
            return Err(LumenError::TypeError {
                message: "MatMul requires at least 2D tensors".to_string(),
            });
        }

        // Get last two dimensions
        let a_m = &a_shape.dims[a_shape.ndim() - 2];
        let a_k = &a_shape.dims[a_shape.ndim() - 1];
        let b_k = &b_shape.dims[b_shape.ndim() - 2];
        let b_n = &b_shape.dims[b_shape.ndim() - 1];

        // Check inner dimensions match
        if !self.dims_compatible(a_k, b_k) {
            return Err(LumenError::TypeError {
                message: format!("MatMul dimension mismatch: {:?} vs {:?}", a_k, b_k),
            });
        }

        // Result shape: (..., M, N)
        let mut result_dims = a_shape.dims[..a_shape.ndim() - 2].to_vec();
        result_dims.push(a_m.clone());
        result_dims.push(b_n.clone());

        Ok(SymbolicShape::new(result_dims))
    }

    /// Infer output shape for element-wise operation
    pub fn infer_elementwise(&self, a_shape: &SymbolicShape, b_shape: &SymbolicShape) -> Result<SymbolicShape> {
        if !a_shape.is_broadcastable_with(b_shape) {
            return Err(LumenError::TypeError {
                message: format!("Shapes not broadcastable: {:?} vs {:?}", a_shape.dims, b_shape.dims),
            });
        }

        // Broadcast to common shape
        let max_ndim = a_shape.ndim().max(b_shape.ndim());
        let mut result_dims = Vec::new();

        for i in 0..max_ndim {
            let a_dim = a_shape.dims.get(a_shape.ndim() - max_ndim + i);
            let b_dim = b_shape.dims.get(b_shape.ndim() - max_ndim + i);

            let result_dim = match (a_dim, b_dim) {
                (None, Some(d)) | (Some(d), None) => d.clone(),
                (Some(SymbolicDim::Static(1)), Some(d)) | (Some(d), Some(SymbolicDim::Static(1))) => d.clone(),
                (Some(d1), Some(d2)) => {
                    if self.dims_compatible(d1, d2) {
                        d1.clone()
                    } else {
                        return Err(LumenError::TypeError {
                            message: format!("Incompatible dimensions: {:?} vs {:?}", d1, d2),
                        });
                    }
                }
                (None, None) => unreachable!(),
            };

            result_dims.push(result_dim);
        }

        Ok(SymbolicShape::new(result_dims))
    }

    /// Check if two dimensions are compatible
    fn dims_compatible(&self, d1: &SymbolicDim, d2: &SymbolicDim) -> bool {
        match (d1, d2) {
            (SymbolicDim::Static(a), SymbolicDim::Static(b)) => a == b,
            (SymbolicDim::Dynamic(s1), SymbolicDim::Dynamic(s2)) => s1 == s2,
            (SymbolicDim::Any, _) | (_, SymbolicDim::Any) => true,
            _ => false,
        }
    }

    /// Specialize graph for concrete input shapes
    pub fn specialize(&self, symbolic_shape: &SymbolicShape, concrete_dims: &HashMap<String, i64>) -> Result<Vec<i64>> {
        let mut result = Vec::new();

        for dim in &symbolic_shape.dims {
            match dim {
                SymbolicDim::Static(v) => result.push(*v),
                SymbolicDim::Dynamic(name) => {
                    let value = concrete_dims.get(name).ok_or_else(|| LumenError::TypeError {
                        message: format!("No concrete value for dynamic dimension '{}'", name),
                    })?;
                    result.push(*value);
                }
                SymbolicDim::Any => {
                    return Err(LumenError::TypeError {
                        message: "Cannot specialize 'Any' dimension without concrete value".to_string(),
                    });
                }
            }
        }

        Ok(result)
    }
}

/// Parse dimension expression from AST
pub fn parse_dim_expr(expr: &str) -> SymbolicDim {
    if expr == "?" {
        SymbolicDim::Any
    } else if let Ok(value) = expr.parse::<i64>() {
        SymbolicDim::Static(value)
    } else {
        SymbolicDim::Dynamic(expr.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_shape() {
        let shape = SymbolicShape::from_static(vec![2, 3, 4]);
        assert!(shape.is_static());
        assert_eq!(shape.to_static(), Some(vec![2, 3, 4]));
    }

    #[test]
    fn test_dynamic_shape() {
        let shape = SymbolicShape::new(vec![
            SymbolicDim::Dynamic("batch".to_string()),
            SymbolicDim::Static(784),
        ]);
        assert!(!shape.is_static());
        assert_eq!(shape.ndim(), 2);
    }

    #[test]
    fn test_broadcasting() {
        let shape1 = SymbolicShape::from_static(vec![2, 1, 4]);
        let shape2 = SymbolicShape::from_static(vec![2, 3, 4]);
        assert!(shape1.is_broadcastable_with(&shape2));
    }

    #[test]
    fn test_matmul_inference() {
        let inference = DynamicShapeInference::new();
        let a = SymbolicShape::from_static(vec![2, 3]);
        let b = SymbolicShape::from_static(vec![3, 4]);

        let result = inference.infer_matmul(&a, &b).unwrap();
        assert_eq!(result.to_static(), Some(vec![2, 4]));
    }
}
