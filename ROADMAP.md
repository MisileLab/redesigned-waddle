# Lumen Compiler Roadmap

## Current Status (v0.1 - Prototype)

**Completed**:
- ✅ Language specification document
- ✅ Compiler architecture design
- ✅ AST definitions
- ✅ Parser implementation (pest-based)
- ✅ Basic project structure
- ✅ CLI tool (`lumenc`)
- ✅ Example programs

**Current Limitations**:
- Parser only (no type checking yet)
- No code generation
- Limited operation support
- No training loop implementation
- No optimizer/autograd

---

## Phase 1: Core Compiler Infrastructure (v0.2-0.4)

### v0.2: Type System & Shape Inference

**Goals**: Implement static type checking and shape inference

**Tasks**:
1. **Symbol Table & Name Resolution**
   - Build symbol table during AST traversal
   - Resolve variable/function/model references
   - Detect undefined names and duplicate declarations

2. **Type Checker**
   - Implement bidirectional type checking
   - Primitive type checking
   - Tensor type checking with shape inference
   - Function type checking

3. **Dimension Constraint System**
   - Symbolic dimension tracking (`dim batch`, `dim hidden = 256`)
   - Constraint generation during type checking
   - Simple constraint solver (unification + arithmetic simplification)
   - Error reporting for shape mismatches

4. **Built-in Operations Type Rules**
   - Element-wise operations (broadcasting)
   - Matrix multiplication
   - Reductions (sum, mean, etc.)
   - Reshape, transpose

**Success Criteria**:
- Type check example programs
- Catch shape errors at compile-time
- Clear error messages with source locations

**Estimated Time**: 3-4 weeks

---

### v0.3: IR & Lowering

**Goals**: Convert typed AST to intermediate representation

**Tasks**:
1. **High-Level IR (HIR) Design**
   - Type-annotated, desugared AST
   - Model instantiation representation
   - Function call graph

2. **HIR Lowering Pass**
   - AST → HIR transformation
   - Model method resolution
   - Constant folding

3. **Mid-Level IR (MIR) Design**
   - SSA-based computation graph
   - Basic blocks and control flow
   - Explicit memory allocation

4. **MIR Lowering Pass**
   - HIR → MIR transformation
   - Control flow graph construction
   - Variable lifetime analysis

**Success Criteria**:
- Convert example programs to MIR
- Visualize computation graphs (dot/graphviz)
- Validate IR correctness

**Estimated Time**: 3-4 weeks

---

### v0.4: Basic Code Generation

**Goals**: Generate executable code from MIR

**Strategy**: Start with **PyTorch interop backend** for rapid validation

**Tasks**:
1. **PyTorch Backend**
   - Generate Python code from MIR
   - Map Lumen operations to PyTorch ops
   - Handle model parameters and initialization
   - Generate training loop code

2. **Runtime Library (Python)**
   - Dataset loaders
   - Optimizer implementations (SGD, Adam)
   - Training utilities

3. **End-to-End Compilation**
   - `lumenc compile` generates .py file
   - Execute generated code
   - Compare results with manual PyTorch code

**Success Criteria**:
- Compile MLP example to PyTorch
- Run training loop successfully
- Verify numerical correctness

**Estimated Time**: 2-3 weeks

---

## Phase 2: Automatic Differentiation (v0.5-0.6)

### v0.5: Forward Mode AD

**Goals**: Implement basic automatic differentiation

**Tasks**:
1. **Computation Graph Construction**
   - Build forward graph from MIR
   - Track parameter nodes
   - Identify loss node

2. **Reverse-Mode AD Implementation**
   - Topological sort of computation graph
   - VJP (Vector-Jacobian Product) for each operation
   - Gradient accumulation

3. **Gradient Code Generation**
   - Generate backward pass code
   - Parameter update code
   - Integration with optimizers

**Operations to Support**:
- Arithmetic ops: +, -, *, /
- MatMul
- Activations: ReLU, Sigmoid, Tanh, Softmax
- Loss functions: Cross-entropy, MSE

**Success Criteria**:
- Automatically compute gradients for MLP
- Verify gradients match PyTorch (gradient checking)
- Train simple models end-to-end

**Estimated Time**: 3-4 weeks

---

### v0.6: Advanced AD Features

**Goals**: Support complex differentiation scenarios

**Tasks**:
1. **Higher-Order Operations**
   - Conv2d forward/backward
   - Pooling operations
   - Batch normalization

2. **Control Flow Differentiation**
   - If statements in differentiable code
   - For loops (unrolling or dynamic)

3. **Checkpointing**
   - Memory-efficient backpropagation
   - Gradient checkpointing for large models

**Estimated Time**: 4-5 weeks

---

## Phase 3: Optimization (v0.7-0.9)

### v0.7: Operator Fusion

**Goals**: Fuse operations to reduce kernel launches and memory traffic

**Tasks**:
1. **Element-wise Fusion**
   - Fuse chains of element-wise operations
   - Example: `(x + y) * z` → single kernel

2. **Linear Layer Fusion**
   - Fuse `matmul + bias + activation`
   - Example: `relu(matmul(x, W) + b)` → `fused_linear_relu`

3. **Fusion Pass Implementation**
   - Pattern matching on MIR
   - Rewrite rules for common patterns
   - Cost model for fusion decisions

**Success Criteria**:
- Reduce kernel count by 30-50%
- Improve training speed by 10-20%

**Estimated Time**: 3-4 weeks

---

### v0.8: Memory Optimization

**Goals**: Minimize memory footprint and optimize memory access patterns

**Tasks**:
1. **Memory Planning**
   - Liveness analysis
   - Memory reuse for temporary tensors
   - In-place operations where safe

2. **Layout Optimization**
   - Choose optimal tensor layouts (NCHW vs NHWC)
   - Minimize transpose operations
   - Cache-friendly access patterns

3. **Memory Pooling**
   - Allocate from pre-allocated pools
   - Reduce allocation overhead

**Estimated Time**: 3-4 weeks

---

### v0.9: Advanced Graph Optimizations

**Goals**: High-level computation graph transformations

**Tasks**:
1. **Common Subexpression Elimination (CSE)**
   - Detect duplicate computations
   - Reuse results

2. **Dead Code Elimination (DCE)**
   - Remove unused computations
   - Prune unreachable branches

3. **Constant Propagation**
   - Evaluate constant expressions at compile-time
   - Fold constants through operations

4. **Algebraic Simplification**
   - `x * 1` → `x`
   - `x + 0` → `x`
   - `x * 0` → `0`

**Estimated Time**: 2-3 weeks

---

## Phase 4: Alternative Backends (v1.0+)

### v1.0: LLVM Backend

**Goals**: Native code generation via LLVM

**Tasks**:
1. **LLVM IR Generation**
   - Map MIR operations to LLVM IR
   - Use external libraries for tensor ops (Eigen, oneDNN, etc.)

2. **Tensor Runtime Library (C++)**
   - Implement tensor data structure
   - CPU kernel library
   - Memory management

3. **Calling Convention**
   - Interface between Lumen and runtime
   - Parameter passing
   - Result return

**Benefits**:
- No Python dependency
- Better CPU performance
- Ahead-of-time compilation

**Estimated Time**: 8-10 weeks

---

### v1.1: CUDA Backend

**Goals**: Direct GPU code generation

**Tasks**:
1. **CUDA Kernel Generation**
   - Generate CUDA C++ from MIR operations
   - Template-based kernel generation
   - PTX compilation

2. **GPU Memory Management**
   - Device memory allocation
   - Host-device transfers
   - Unified memory support

3. **Kernel Fusion & Tiling**
   - Fuse operations into single kernels
   - Tile for shared memory
   - Thread block configuration

**Benefits**:
- Maximum GPU performance
- Fine-grained control over execution

**Challenges**:
- Complexity of kernel generation
- Optimization required for performance

**Estimated Time**: 10-12 weeks

---

## Phase 5: Advanced Features (v1.2+)

### v1.2: Schedule DSL

**Goals**: Separate algorithm from schedule (Halide/TVM-style)

**Concept**:
```lumen
fn matmul_schedule(M: dim, N: dim, K: dim) {
  tile(M, 64)
  tile(N, 64)
  tile(K, 16)
  vectorize(N, 8)
  parallel(M)
  cache(A, shared)
  cache(B, shared)
}

apply_schedule(matmul, matmul_schedule)
```

**Tasks**:
1. **Schedule Language Design**
   - Tiling, fusion, reordering
   - Vectorization, parallelization
   - Memory hierarchy control

2. **Schedule Compiler**
   - Apply schedule transformations to IR
   - Verify schedule legality
   - Generate optimized code

3. **Auto-Scheduler Integration**
   - Search space exploration
   - Cost model
   - ML-based tuning (AutoTVM-style)

**Estimated Time**: 12-16 weeks

---

### v1.3: Distributed & Parallel Execution

**Goals**: Multi-device training

**Features**:
1. **Data Parallelism**
   ```lumen
   mesh devices[gpus: 8] {
     shard dim batch across gpus
     train { ... }
   }
   ```

2. **Tensor Parallelism**
   - Shard tensors across devices
   - SPMD (Single Program Multiple Data)

3. **Pipeline Parallelism**
   - Split model across stages
   - Micro-batching

4. **Communication Primitives**
   - AllReduce, AllGather, ReduceScatter
   - NCCL/Gloo backend integration

**Estimated Time**: 16-20 weeks

---

### v1.4: Mixed Precision Training

**Goals**: Automatic FP16/BF16 support

**Tasks**:
1. **Type System Extension**
   - Mixed precision type annotations
   - Automatic casting rules

2. **Loss Scaling**
   - Dynamic loss scaling
   - Gradient clipping

3. **Precision Selection**
   - Heuristics for operation precision
   - User annotations

**Estimated Time**: 6-8 weeks

---

### v1.5: Model Quantization

**Goals**: INT8/INT4 quantization for inference

**Tasks**:
1. **Quantization-Aware Training**
   - Fake quantization nodes
   - Learned quantization parameters

2. **Post-Training Quantization**
   - Calibration on sample data
   - Weight/activation quantization

3. **Quantized Kernels**
   - INT8 matmul, conv
   - Dequantize-compute-quantize fusion

**Estimated Time**: 8-10 weeks

---

## Phase 6: Interoperability & Ecosystem (v2.0+)

### v2.0: Framework Interop

**Goals**: Import/export models from other frameworks

**Tasks**:
1. **ONNX Import/Export**
   - Read ONNX models into Lumen IR
   - Export Lumen models to ONNX

2. **PyTorch Model Import**
   - Convert torch.nn.Module to Lumen
   - Parameter transfer

3. **JAX Interop**
   - Call JAX functions from Lumen
   - Leverage XLA backend

**Estimated Time**: 8-10 weeks

---

### v2.1: Standard Library

**Goals**: Comprehensive library of layers, models, utilities

**Components**:
1. **Core Layers**
   - Linear, Conv2d, BatchNorm, LayerNorm, Dropout, Embedding

2. **Pretrained Models**
   - ResNet, VGG, Transformer, BERT, GPT

3. **Utilities**
   - Data augmentation
   - Learning rate schedulers
   - Metrics and evaluation

**Estimated Time**: 12-16 weeks (ongoing)

---

## Long-Term Vision (v3.0+)

### Dynamic Shapes

- Support runtime-variable shapes
- JIT compilation for dynamic cases

### Model Serving

- Efficient inference runtime
- Batching, caching, quantization
- Server deployment

### Debugging & Profiling

- Step-through debugging
- Performance profiling
- Memory leak detection

### IDE Integration

- Language server protocol (LSP)
- Syntax highlighting
- IntelliSense / autocomplete

### Formal Verification

- Prove shape correctness
- Verify numerical stability
- Security guarantees

---

## Development Priorities

**Short-term (next 6 months)**:
1. Type system & shape checking (v0.2)
2. IR design & lowering (v0.3)
3. PyTorch backend (v0.4)
4. Automatic differentiation (v0.5-0.6)

**Mid-term (6-12 months)**:
1. Optimization passes (v0.7-0.9)
2. LLVM backend (v1.0)
3. CUDA backend (v1.1)

**Long-term (1-2 years)**:
1. Schedule DSL (v1.2)
2. Distributed training (v1.3)
3. Mixed precision & quantization (v1.4-1.5)
4. Ecosystem building (v2.0+)

---

## Success Metrics

### Performance Benchmarks

Compare against PyTorch on:
- ResNet-50 training (ImageNet)
- BERT-base training (text)
- GPT-2 training (language modeling)

**Target**: Match or exceed PyTorch performance

### Developer Experience

- Compilation time < 5s for medium models
- Clear error messages
- Comprehensive documentation
- Active community

---

## Contributing

We welcome contributions! Priority areas:
1. Type system implementation
2. Operation library expansion
3. Optimization passes
4. Documentation and examples
5. Testing and benchmarking

See `CONTRIBUTING.md` for guidelines.

---

**Last Updated**: 2025-11-15
**Version**: 0.1 (Prototype)
