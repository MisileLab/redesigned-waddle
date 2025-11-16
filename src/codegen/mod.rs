// Code generation backends

pub mod triton;
pub mod cuda;
pub mod rocm;
pub mod llvm_backend;
pub mod webgpu;

pub use triton::TritonCodegen;
pub use cuda::CudaCodegen;
pub use rocm::RocmCodegen;
pub use llvm_backend::LLVMCodegen;
pub use webgpu::WebGPUCodegen;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Triton,
    Cuda,
    Rocm,
    LLVM,
    WebGPU,
}
