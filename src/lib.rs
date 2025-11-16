// Lumen Compiler Library

pub mod ast;
pub mod parser;
pub mod error;
pub mod resolve;
pub mod typeck;
pub mod hir;
pub mod mir;
pub mod codegen;
pub mod autograd;
pub mod advanced_autograd;
pub mod optimize;
pub mod schedule;
pub mod mixed_precision;
pub mod distributed;
pub mod safetensors_support;

// Re-exports
pub use ast::*;
pub use parser::parse;
pub use error::*;
pub use resolve::{NameResolver, SymbolTable};
pub use typeck::{TypeChecker, TypeContext};
pub use autograd::{AutogradContext, Optimizer};
pub use advanced_autograd::AdvancedAutogradContext;
pub use optimize::OptimizationPipeline;
pub use schedule::{Schedule, AutoScheduler};
pub use mixed_precision::{AMPConfig, MixedPrecisionOptimizer, Precision};
pub use distributed::{DistributedConfig, Strategy, DeviceMesh};
pub use safetensors_support::{SafetensorsExporter, SafetensorsImporter, ModelMetadata};
