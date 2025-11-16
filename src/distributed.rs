// Distributed Training Framework
// Multi-GPU data parallelism, tensor parallelism, and pipeline parallelism

use crate::mir::*;
use crate::error::Result;
use std::collections::HashMap;

/// Distributed training strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Data parallelism - replicate model, shard data
    DataParallel,

    /// Distributed Data Parallel (DDP)
    DistributedDataParallel,

    /// Tensor parallelism - shard model tensors
    TensorParallel,

    /// Pipeline parallelism - split model across devices
    PipelineParallel,

    /// Fully Sharded Data Parallel (FSDP/ZeRO)
    FullySharded,
}

/// Device mesh configuration
#[derive(Debug, Clone)]
pub struct DeviceMesh {
    /// Number of devices
    pub world_size: usize,

    /// Device IDs
    pub devices: Vec<usize>,

    /// Mesh dimensions for multi-dimensional parallelism
    pub mesh_shape: Vec<usize>,
}

impl DeviceMesh {
    pub fn new(world_size: usize) -> Self {
        Self {
            world_size,
            devices: (0..world_size).collect(),
            mesh_shape: vec![world_size],
        }
    }

    pub fn new_2d(data_parallel: usize, tensor_parallel: usize) -> Self {
        Self {
            world_size: data_parallel * tensor_parallel,
            devices: (0..data_parallel * tensor_parallel).collect(),
            mesh_shape: vec![data_parallel, tensor_parallel],
        }
    }
}

/// Communication primitives
#[derive(Debug, Clone)]
pub enum CollectiveOp {
    /// All-reduce: sum gradients across devices
    AllReduce { tensor: String, op: ReduceOp },

    /// All-gather: gather tensors from all devices
    AllGather { tensor: String },

    /// Reduce-scatter: reduce and scatter result
    ReduceScatter { tensor: String, op: ReduceOp },

    /// Broadcast: send tensor to all devices
    Broadcast { tensor: String, src: usize },

    /// Send/Recv: point-to-point communication
    Send { tensor: String, dst: usize },
    Recv { tensor: String, src: usize },
}

#[derive(Debug, Clone, Copy)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
}

/// Distributed training configuration
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Training strategy
    pub strategy: Strategy,

    /// Device mesh
    pub mesh: DeviceMesh,

    /// Backend (nccl, gloo, mpi)
    pub backend: String,

    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,

    /// Find unused parameters (for DDP)
    pub find_unused_parameters: bool,

    /// Bucket size (MB) for gradient allreduce
    pub bucket_cap_mb: f32,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            strategy: Strategy::DistributedDataParallel,
            mesh: DeviceMesh::new(1),
            backend: "nccl".to_string(),
            gradient_accumulation_steps: 1,
            find_unused_parameters: false,
            bucket_cap_mb: 25.0,
        }
    }
}

/// Distributed code generator
pub struct DistributedCodegen {
    config: DistributedConfig,
    collective_ops: Vec<CollectiveOp>,
}

impl DistributedCodegen {
    pub fn new(config: DistributedConfig) -> Self {
        Self {
            config,
            collective_ops: Vec::new(),
        }
    }

    /// Generate PyTorch DDP code
    pub fn generate_ddp_code(&self) -> String {
        format!(
            r#"
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
def setup(rank, world_size):
    dist.init_process_group(
        backend='{backend}',
        init_method='env://',
        world_size={world_size},
        rank=rank
    )

# Cleanup
def cleanup():
    dist.destroy_process_group()

# Wrap model with DDP
def create_ddp_model(model, device_id):
    model = model.to(device_id)
    ddp_model = DDP(
        model,
        device_ids=[device_id],
        find_unused_parameters={find_unused},
        bucket_cap_mb={bucket_cap}
    )
    return ddp_model

# Training step with gradient accumulation
def train_step(model, optimizer, data, target, step, accum_steps={accum_steps}):
    output = model(data)
    loss = criterion(output, target) / accum_steps

    loss.backward()

    # Only step optimizer every accum_steps
    if (step + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    return loss.item() * accum_steps
"#,
            backend = self.config.backend,
            world_size = self.config.mesh.world_size,
            find_unused = if self.config.find_unused_parameters {
                "True"
            } else {
                "False"
            },
            bucket_cap = self.config.bucket_cap_mb,
            accum_steps = self.config.gradient_accumulation_steps
        )
    }

    /// Generate FSDP code (Fully Sharded Data Parallel)
    pub fn generate_fsdp_code(&self) -> String {
        r#"
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

# Auto-wrap policy
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=1e8  # 100M parameters
)

# Create FSDP model
def create_fsdp_model(model, device_id):
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=device_id,
        sync_module_states=True,
        sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
    )
    return model
"#
        .to_string()
    }

    /// Generate tensor parallel code
    pub fn generate_tensor_parallel_code(&self) -> String {
        r#"
# Tensor Parallelism (Megatron-style)
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        self.world_size = world_size
        self.rank = rank

        # Shard output dimension
        assert out_features % world_size == 0
        self.out_features_per_partition = out_features // world_size

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )

    def forward(self, input):
        # Each device computes partial output
        output_parallel = F.linear(input, self.weight)
        # No communication needed for column parallel
        return output_parallel

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        self.world_size = world_size
        self.rank = rank

        # Shard input dimension
        assert in_features % world_size == 0
        self.in_features_per_partition = in_features // world_size

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )

    def forward(self, input):
        # Input is already partitioned
        output_parallel = F.linear(input, self.weight)

        # All-reduce to get final output
        dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)

        return output_parallel
"#
        .to_string()
    }

    /// Generate pipeline parallel code
    pub fn generate_pipeline_parallel_code(&self) -> String {
        r#"
# Pipeline Parallelism
from torch.distributed.pipeline.sync import Pipe

def create_pipeline_model(model, chunks=4):
    # Split model into stages
    # Example: model has 4 stages, 2 devices
    # Stages 0-1 on device 0, stages 2-3 on device 1

    # Wrap with Pipe
    model = Pipe(
        model,
        chunks=chunks,  # Micro-batch size
        checkpoint='except_last'  # Gradient checkpointing
    )

    return model

# Training with pipeline parallelism
def train_step_pipeline(model, optimizer, data, target):
    # Pipe handles forward/backward automatically
    output = model(data)
    loss = criterion(output.local_value(), target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
"#
        .to_string()
    }

    /// Generate NCCL collective operation
    pub fn generate_collective_op(&self, op: &CollectiveOp) -> String {
        match op {
            CollectiveOp::AllReduce { tensor, op } => {
                let reduce_op = match op {
                    ReduceOp::Sum => "dist.ReduceOp.SUM",
                    ReduceOp::Mean => "dist.ReduceOp.AVG",
                    ReduceOp::Max => "dist.ReduceOp.MAX",
                    ReduceOp::Min => "dist.ReduceOp.MIN",
                };
                format!("dist.all_reduce({}, op={})", tensor, reduce_op)
            }
            CollectiveOp::AllGather { tensor } => {
                format!("dist.all_gather_into_tensor(output, {})", tensor)
            }
            CollectiveOp::ReduceScatter { tensor, op } => {
                let reduce_op = match op {
                    ReduceOp::Sum => "dist.ReduceOp.SUM",
                    ReduceOp::Mean => "dist.ReduceOp.AVG",
                    ReduceOp::Max => "dist.ReduceOp.MAX",
                    ReduceOp::Min => "dist.ReduceOp.MIN",
                };
                format!("dist.reduce_scatter_tensor(output, {}, op={})", tensor, reduce_op)
            }
            CollectiveOp::Broadcast { tensor, src } => {
                format!("dist.broadcast({}, src={})", tensor, src)
            }
            CollectiveOp::Send { tensor, dst } => {
                format!("dist.send({}, dst={})", tensor, dst)
            }
            CollectiveOp::Recv { tensor, src } => {
                format!("dist.recv({}, src={})", tensor, src)
            }
        }
    }

    /// Insert gradient allreduce
    pub fn insert_gradient_sync(&mut self, param: &str) {
        self.collective_ops.push(CollectiveOp::AllReduce {
            tensor: format!("{}.grad", param),
            op: ReduceOp::Mean,
        });
    }
}

/// Sharding plan for tensors
#[derive(Debug, Clone)]
pub struct ShardingPlan {
    shards: HashMap<String, ShardSpec>,
}

#[derive(Debug, Clone)]
pub struct ShardSpec {
    pub tensor_name: String,
    pub shard_dim: usize,
    pub num_shards: usize,
}

impl ShardingPlan {
    pub fn new() -> Self {
        Self {
            shards: HashMap::new(),
        }
    }

    pub fn shard_tensor(&mut self, tensor: &str, dim: usize, num_shards: usize) {
        self.shards.insert(
            tensor.to_string(),
            ShardSpec {
                tensor_name: tensor.to_string(),
                shard_dim: dim,
                num_shards,
            },
        );
    }

    pub fn get_shard_spec(&self, tensor: &str) -> Option<&ShardSpec> {
        self.shards.get(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_mesh() {
        let mesh = DeviceMesh::new(4);
        assert_eq!(mesh.world_size, 4);
        assert_eq!(mesh.devices.len(), 4);

        let mesh_2d = DeviceMesh::new_2d(2, 2);
        assert_eq!(mesh_2d.world_size, 4);
        assert_eq!(mesh_2d.mesh_shape, vec![2, 2]);
    }

    #[test]
    fn test_distributed_config() {
        let config = DistributedConfig::default();
        assert_eq!(config.strategy, Strategy::DistributedDataParallel);
        assert_eq!(config.backend, "nccl");
    }

    #[test]
    fn test_ddp_codegen() {
        let config = DistributedConfig::default();
        let codegen = DistributedCodegen::new(config);

        let code = codegen.generate_ddp_code();
        assert!(code.contains("DistributedDataParallel"));
        assert!(code.contains("init_process_group"));
    }

    #[test]
    fn test_sharding_plan() {
        let mut plan = ShardingPlan::new();
        plan.shard_tensor("weight", 0, 4);

        let spec = plan.get_shard_spec("weight").unwrap();
        assert_eq!(spec.shard_dim, 0);
        assert_eq!(spec.num_shards, 4);
    }
}
