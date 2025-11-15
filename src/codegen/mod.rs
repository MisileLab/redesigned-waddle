// Code generation backends

pub mod pytorch;
pub mod triton;
pub mod cuda;
pub mod rocm;
pub mod llvm_backend;

pub use pytorch::PyTorchCodegen;
pub use triton::TritonCodegen;
pub use cuda::CudaCodegen;
pub use rocm::RocmCodegen;
pub use llvm_backend::LLVMCodegen;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    PyTorch,
    Triton,
    Cuda,
    Rocm,
    LLVM,
}
