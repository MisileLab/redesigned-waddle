// Code generation backends

pub mod pytorch;
pub mod triton;
pub mod cuda;

pub use pytorch::PyTorchCodegen;
pub use triton::TritonCodegen;
pub use cuda::CudaCodegen;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    PyTorch,
    Triton,
    Cuda,
}
