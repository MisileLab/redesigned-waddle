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

// New advanced features
pub mod graph_ir;
pub mod dynamic_shapes;
pub mod quantization;
pub mod profiling;
pub mod memory_planning;
pub mod advanced_ops;
pub mod benchmark;

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

// New advanced features
pub use graph_ir::Graph;
pub use dynamic_shapes::{SymbolicShape, DynamicShapeInference};
pub use quantization::{QuantConfig, QuantizationScheme};
pub use profiling::{Profiler, MemoryAnalyzer, Visualizer};
pub use memory_planning::{MemoryPlanner, AllocationStrategy};
pub use advanced_ops::AdvancedOpsLibrary;
pub use benchmark::{BenchmarkSuite, BenchmarkConfig};
