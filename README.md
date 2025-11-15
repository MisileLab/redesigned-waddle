# Lumen: A Statically-Typed Deep Learning DSL

**Lumen** is an ahead-of-time compiled, statically-typed domain-specific language for deep learning. It aims to provide PyTorch-level expressiveness with superior performance through aggressive compile-time optimization and static shape verification.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Version](https://img.shields.io/badge/version-0.1.0-orange)]()

---

## 🎯 Goals

1. **Static Shape Verification**: Catch dimension mismatches at compile-time
2. **Performance**: Match or exceed PyTorch through AOT compilation and optimization
3. **Expressiveness**: First-class support for models, training loops, and optimizers
4. **Safety**: No runtime shape errors, memory safety through Rust

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lumen.git
cd lumen

# Build the compiler
cargo build --release

# Add to PATH (optional)
export PATH="$PWD/target/release:$PATH"
```

### Example: Multi-Layer Perceptron

Create a file `mlp.lumen`:

```lumen
// Dimension declarations
dim batch
dim features = 784
dim hidden = 256
dim classes = 10

// Model definition
model MLP {
  param W1: Tensor[f32, features, hidden] ~ xavier()
  param b1: Tensor[f32, hidden] ~ zeros()

  param W2: Tensor[f32, hidden, classes] ~ xavier()
  param b2: Tensor[f32, classes] ~ zeros()

  fn forward(x: Tensor[f32, batch, features]) -> Tensor[f32, batch, classes] {
    let h = relu(matmul(x, W1) + b1)
    let out = matmul(h, W2) + b2
    return out
  }
}

// Loss function
fn classification_loss(model: MLP,
                       x: Tensor[f32, batch, features],
                       y: Tensor[i64, batch]) -> Tensor[f32] {
  let logits = model.forward(x)
  return cross_entropy(logits, y)
}
```

### Compile and Run

```bash
# Parse the file (current v0.1 capability)
lumenc parse mlp.lumen

# Compile (coming in v0.4)
lumenc compile mlp.lumen -o mlp.py

# Run training (coming in v0.4)
python mlp.py
```

---

## 🌟 Features

### Current (v0.4) - **Working Compiler!** 🎉

- ✅ **Parser**: Full syntax parsing with pest
- ✅ **AST**: Complete abstract syntax tree representation
- ✅ **CLI**: Command-line interface (`lumenc compile`)
- ✅ **Symbol Table**: Name resolution with scopes
- ✅ **Type Checking**: Static type and shape inference
- ✅ **IR Pipeline**: Multi-level IR (HIR → MIR)
- ✅ **PyTorch Backend**: Full code generation to executable Python
- ✅ **Built-in Operations**: matmul, relu, sigmoid, tanh, etc.

**You can now compile Lumen code to PyTorch!**

```bash
lumenc compile examples/standalone.lumen
# Generates executable PyTorch code in examples/standalone.py
```

### In Progress (v0.5-0.6)

- 🔨 **Model Classes**: Full support for model definitions
- 🔨 **Training Loops**: Native training DSL
- 🔨 **Automatic Differentiation**: Reverse-mode AD with gradient computation

### Planned (v0.7+)

- 📋 **Optimization**: Operator fusion, memory planning, CSE/DCE
- 📋 **LLVM Backend**: Native code generation
- 📋 **CUDA Backend**: Direct GPU kernel generation
- 📋 **Distributed Training**: Multi-GPU support
- 📋 **Mixed Precision**: Automatic FP16/BF16

See [ROADMAP.md](ROADMAP.md) for detailed development plan.

---

## 📖 Language Features

### 1. Dimension System

Dimensions are first-class citizens:

```lumen
dim batch               // Symbolic dimension
dim seq = 512          // Concrete dimension
dim hidden where hidden % 8 == 0  // Constrained dimension
```

### 2. Tensor Types

Tensors have explicit dtype and shape:

```lumen
Tensor[f32, batch, seq, hidden]     // 3D tensor
Tensor[i64, batch]                   // 1D tensor (labels)
Tensor[bool, height, width]          // 2D boolean mask
```

### 3. Static Shape Checking

The compiler verifies shape compatibility:

```lumen
// ✓ Valid: shapes match
let A: Tensor[f32, batch, 784]
let W: Tensor[f32, 784, 256]
let result = matmul(A, W)  // → Tensor[f32, batch, 256]

// ✗ Error: shape mismatch detected at compile-time
let B: Tensor[f32, batch, 512]
let result = matmul(B, W)  // Compile error: 512 ≠ 784
```

### 4. Model Declaration

Models encapsulate parameters and methods:

```lumen
model Transformer {
  param W_qkv: Tensor[f32, embed, 3 * embed] ~ xavier()
  param W_out: Tensor[f32, embed, embed] ~ xavier()

  fn attention(x: Tensor[f32, batch, seq, embed])
      -> Tensor[f32, batch, seq, embed] {
    // ... attention logic
  }
}
```

### 5. Training Loops (Coming in v0.4)

Training is part of the language:

```lumen
train {
  use model MLP as net
  dataset mnist(batch=128, shuffle=true) as (images, labels)

  optimizer adam(params=net.params, lr=1e-3)

  for step in 0..10000 {
    let loss = classification_loss(net, images, labels)
    minimize loss

    if step % 100 == 0 {
      print("step =", step, " loss =", loss)
    }
  }
}
```

---

## 🏗️ Architecture

```
┌─────────────┐
│ Lumen Source│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Parser    │ (pest grammar)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   AST       │ (syntax tree)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Type Checker│ (shape inference)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   HIR       │ (high-level IR)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Autograd   │ (gradient generation)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   MIR       │ (mid-level IR)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Optimizer   │ (fusion, CSE, DCE)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   LIR       │ (low-level IR)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Codegen    │ (PyTorch/LLVM/CUDA)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Executable  │
└─────────────┘
```

See [COMPILER_ARCHITECTURE.md](COMPILER_ARCHITECTURE.md) for details.

---

## 📚 Documentation

- [Language Specification](LANGUAGE_SPEC.md) - Syntax, semantics, type system
- [Compiler Architecture](COMPILER_ARCHITECTURE.md) - Internal design
- [Roadmap](ROADMAP.md) - Development plan and future features
- [Examples](examples/) - Sample programs

---

## 🛠️ Development

### Prerequisites

- Rust 1.70+ (2021 edition)
- Cargo

### Building from Source

```bash
git clone https://github.com/yourusername/lumen.git
cd lumen
cargo build
```

### Running Tests

```bash
cargo test
```

### Project Structure

```
lumen/
├── src/
│   ├── ast.rs           # AST definitions
│   ├── parser/          # Parser (pest-based)
│   ├── error.rs         # Error types
│   ├── lib.rs           # Library entry point
│   └── main.rs          # CLI
├── examples/            # Example .lumen files
├── tests/               # Integration tests
├── docs/                # Documentation
│   ├── LANGUAGE_SPEC.md
│   ├── COMPILER_ARCHITECTURE.md
│   └── ROADMAP.md
└── Cargo.toml
```

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

1. **Type System**: Implement shape inference and constraint solving
2. **IR Design**: Design and implement intermediate representations
3. **Optimization**: Add fusion, CSE, DCE passes
4. **Code Generation**: PyTorch/LLVM/CUDA backends
5. **Documentation**: Improve docs and examples
6. **Testing**: Add test cases and benchmarks

See `CONTRIBUTING.md` for guidelines (coming soon).

---

## 📊 Comparison with Existing Systems

| Feature | PyTorch | JAX | Triton | **Lumen** |
|---------|---------|-----|--------|-----------|
| **Type System** | Dynamic | Dynamic (hints) | Weak | **Static** |
| **Shape Checking** | Runtime | Runtime | Runtime | **Compile-time** |
| **Abstraction Level** | Eager Op | Function | Kernel | **Full Training** |
| **Compilation** | JIT | JIT (XLA) | JIT | **AOT** |
| **Training Loop** | Python | Python | Python | **Native DSL** |
| **Optimization** | Runtime | XLA | Auto | **Multi-pass AOT** |

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Inspired by:
- **PyTorch**: Ease of use and flexibility
- **JAX**: Functional transformations and XLA compilation
- **Triton**: GPU programming abstraction
- **Halide**: Schedule separation
- **TVM**: IR design and auto-scheduling
- **Rust**: Memory safety and performance

---

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/lumen/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/lumen/discussions)

---

**Status**: 🚧 Prototype (v0.1) - Active Development

See [ROADMAP.md](ROADMAP.md) for the path to v1.0 and beyond!
