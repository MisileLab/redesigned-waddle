# Lumen Language Specification v0.1

## 1. Philosophy and Goals

### 1.1 Core Philosophy

Lumen is a **statically-typed, ahead-of-time compiled domain-specific language** for deep learning that prioritizes:

1. **Safety through static analysis**: Catch shape and type errors at compile-time
2. **Performance through aggressive optimization**: AOT compilation enables optimizations impossible in dynamic frameworks
3. **Expressiveness at the right level**: Model architecture, training loops, and optimization strategies as first-class language constructs
4. **Explicit over implicit**: Clear semantics for dimensions, parallelism, and memory layout

### 1.2 Design Goals

1. **Static Shape System**: All tensor shapes known at compile-time (with support for symbolic dimensions)
2. **Unified Training DSL**: Model definition, loss functions, optimizers, and training loops in one language
3. **Separation of Concerns**: Algorithm (what to compute) vs Schedule (how to compute) can be separated
4. **Performance Target**: Match or exceed PyTorch/JAX performance through:
   - Kernel fusion
   - Static memory planning
   - Automatic mixed precision
   - Graph-level optimizations
5. **Future-proof**: Designed for distributed/parallel execution from the ground up

### 1.3 Differentiation from Existing Systems

| Aspect | Triton | JAX | PyTorch | **Lumen** |
|--------|--------|-----|---------|-----------|
| Abstraction Level | GPU Kernel | Function | Eager Op | Full Training Pipeline |
| Type System | Weak | Python hints | Dynamic | **Static + Shape** |
| Shape Checking | Runtime | Runtime | Runtime | **Compile-time** |
| Training Loop | External (Python) | External (Python) | External (Python) | **Native DSL** |
| Compilation | JIT | JIT (XLA) | JIT/Eager | **AOT** |

---

## 2. Core Concepts

### 2.1 Dimension Declarations

Dimensions are first-class symbolic values that can be:
- **Concrete constants**: `dim hidden = 512`
- **Symbolic (runtime-bound)**: `dim batch`
- **Constrained**: `dim seq where seq <= 1024`

### 2.2 Type System

**Primitive Types**:
- Integers: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
- Floats: `f16`, `bf16`, `f32`, `f64`
- Boolean: `bool`

**Tensor Types**:
- `Tensor[dtype, d1, d2, ..., dn]` where each `di` is a dimension expression
- Examples:
  - `Tensor[f32, batch, 784]`
  - `Tensor[f32, 3, 224, 224]`
  - `Tensor[i64, batch]`

**Function Types**:
- `fn(T1, T2, ..., Tn) -> R`

### 2.3 Model Declaration

Models are containers for:
- **Parameters** (`param`): Trainable tensors with initializers
- **Buffers** (`buffer`): Non-trainable state (e.g., batch norm running stats)
- **Functions** (`fn`): Computation logic

### 2.4 Training Constructs

- `train { ... }`: Training loop block
- `dataset`: Data source specification
- `optimizer`: Optimization algorithm
- `minimize`: Automatic differentiation + gradient update

---

## 3. Grammar (EBNF)

```ebnf
(* Top-level program *)
program = { declaration } ;

declaration = dim_decl
            | const_decl
            | fn_decl
            | model_decl
            | train_decl
            ;

(* Dimension declarations *)
dim_decl = "dim" IDENT [ "=" expr ] [ "where" constraint ] ;

constraint = expr comparison_op expr
           | constraint "and" constraint
           | constraint "or" constraint
           ;

comparison_op = "==" | "!=" | "<" | "<=" | ">" | ">=" | "%" ;

(* Constant declarations *)
const_decl = "const" IDENT ":" type "=" expr ;

(* Function declarations *)
fn_decl = "fn" IDENT "(" param_list ")" [ "->" type ] block ;

param_list = [ param { "," param } ] ;
param = IDENT ":" type ;

(* Model declarations *)
model_decl = "model" IDENT "{" { model_member } "}" ;

model_member = param_decl
             | buffer_decl
             | fn_decl
             ;

param_decl = "param" IDENT ":" type [ "~" initializer ] ;
buffer_decl = "buffer" IDENT ":" type [ "~" initializer ] ;

initializer = "zeros" "(" ")"
            | "ones" "(" ")"
            | "normal" "(" expr "," expr ")"
            | "uniform" "(" expr "," expr ")"
            | "xavier" "(" ")"
            | "kaiming" "(" ")"
            | "constant" "(" expr ")"
            ;

(* Training declarations *)
train_decl = "train" "{" { train_stmt } "}" ;

train_stmt = use_stmt
           | dataset_stmt
           | optimizer_stmt
           | stmt
           ;

use_stmt = "use" "model" type "as" IDENT ;
dataset_stmt = "dataset" IDENT "(" dataset_args ")" "as" pattern ;
optimizer_stmt = "optimizer" IDENT "(" optimizer_args ")" ;
dataset_args = IDENT "=" expr { "," IDENT "=" expr } ;
optimizer_args = IDENT "=" expr { "," IDENT "=" expr } ;

(* Statements *)
stmt = let_stmt
     | assign_stmt
     | if_stmt
     | for_stmt
     | while_stmt
     | minimize_stmt
     | return_stmt
     | expr_stmt
     ;

let_stmt = "let" pattern [ ":" type ] "=" expr ;
assign_stmt = expr "=" expr ;
if_stmt = "if" expr block [ "else" ( if_stmt | block ) ] ;
for_stmt = "for" IDENT "in" expr block ;
while_stmt = "while" expr block ;
minimize_stmt = "minimize" expr ;
return_stmt = "return" [ expr ] ;
expr_stmt = expr ;

block = "{" { stmt } "}" ;

pattern = IDENT
        | "(" IDENT { "," IDENT } ")"
        ;

(* Types *)
type = primitive_type
     | tensor_type
     | function_type
     | IDENT
     ;

primitive_type = "i8" | "i16" | "i32" | "i64"
               | "u8" | "u16" | "u32" | "u64"
               | "f16" | "bf16" | "f32" | "f64"
               | "bool"
               ;

tensor_type = "Tensor" "[" type { "," dim_expr } "]" ;

dim_expr = IDENT
         | NUMBER
         | dim_expr ( "+" | "-" | "*" | "/" | "%" ) dim_expr
         | "(" dim_expr ")"
         ;

function_type = "fn" "(" [ type { "," type } ] ")" [ "->" type ] ;

(* Expressions *)
expr = literal
     | IDENT
     | expr "." IDENT                    (* field access *)
     | expr "(" [ expr { "," expr } ] ")" (* function call *)
     | expr "[" expr { "," expr } "]"    (* indexing *)
     | "(" expr ")"
     | expr binary_op expr
     | unary_op expr
     | if_expr
     | block_expr
     ;

if_expr = "if" expr block "else" ( if_expr | block ) ;
block_expr = block ;

binary_op = "+" | "-" | "*" | "/" | "%" | "**"
          | "==" | "!=" | "<" | "<=" | ">" | ">="
          | "and" | "or"
          | "@"  (* matrix multiplication *)
          ;

unary_op = "-" | "!" | "not" ;

literal = NUMBER | FLOAT | STRING | BOOL | tensor_literal ;

tensor_literal = "[" [ expr { "," expr } ] "]" ;

(* Lexical elements *)
IDENT = LETTER { LETTER | DIGIT | "_" } ;
NUMBER = DIGIT { DIGIT } ;
FLOAT = DIGIT { DIGIT } "." DIGIT { DIGIT } [ EXPONENT ]
      | DIGIT { DIGIT } EXPONENT
      ;
EXPONENT = ( "e" | "E" ) [ "+" | "-" ] DIGIT { DIGIT } ;
STRING = '"' { ANY_CHAR - '"' } '"' ;
BOOL = "true" | "false" ;
LETTER = "a" .. "z" | "A" .. "Z" ;
DIGIT = "0" .. "9" ;
```

---

## 4. Type and Shape System

### 4.1 Type Inference Rules

Lumen uses bidirectional type checking:
- **Checking mode**: Type flows from context to expression
- **Synthesis mode**: Type flows from expression to context

### 4.2 Tensor Operation Type Rules

#### 4.2.1 Element-wise Operations

For binary operations `op ∈ {+, -, *, /, %}`:

```
Γ ⊢ e₁ : Tensor[T, d₁, d₂, ..., dₙ]
Γ ⊢ e₂ : Tensor[T, d₁, d₂, ..., dₙ]
─────────────────────────────────────
Γ ⊢ e₁ op e₂ : Tensor[T, d₁, d₂, ..., dₙ]
```

**Broadcasting**: When shapes differ, NumPy-style broadcasting applies:
- Dimensions are aligned from the right
- Dimensions of size 1 can broadcast to any size
- Missing dimensions are treated as 1

```
Tensor[f32, batch, 1, hidden] + Tensor[f32, hidden]
→ Tensor[f32, batch, 1, hidden] + Tensor[f32, 1, 1, hidden]  (broadcast)
→ Tensor[f32, batch, 1, hidden]
```

#### 4.2.2 Matrix Multiplication

```
Γ ⊢ A : Tensor[T, ..., M, K]
Γ ⊢ B : Tensor[T, ..., K, N]
─────────────────────────────
Γ ⊢ matmul(A, B) : Tensor[T, ..., M, N]
```

Batch dimensions must broadcast-compatible.

#### 4.2.3 Reductions

```
Γ ⊢ x : Tensor[T, d₁, ..., dᵢ, ..., dₙ]
axis = i
─────────────────────────────────────────
Γ ⊢ sum(x, axis=i) : Tensor[T, d₁, ..., dᵢ₋₁, dᵢ₊₁, ..., dₙ]
```

Same for `mean`, `max`, `min`, etc.

#### 4.2.4 Reshape

```
Γ ⊢ x : Tensor[T, d₁, d₂, ..., dₙ]
∏dᵢ = ∏sⱼ  (element count preserved)
─────────────────────────────────────
Γ ⊢ reshape(x, [s₁, s₂, ..., sₘ]) : Tensor[T, s₁, s₂, ..., sₘ]
```

The compiler verifies `∏dᵢ = ∏sⱼ` at compile-time when possible.

#### 4.2.5 Convolution (2D)

```
Γ ⊢ input : Tensor[T, batch, C_in, H_in, W_in]
Γ ⊢ kernel : Tensor[T, C_out, C_in, K_h, K_w]
stride = (s_h, s_w), padding = (p_h, p_w)
H_out = ⌊(H_in + 2·p_h - K_h) / s_h⌋ + 1
W_out = ⌊(W_in + 2·p_w - K_w) / s_w⌋ + 1
─────────────────────────────────────────────────
Γ ⊢ conv2d(input, kernel, stride, padding) : Tensor[T, batch, C_out, H_out, W_out]
```

### 4.3 Dimension Constraint Solving

The compiler maintains a constraint system:
- **Equality constraints**: `d1 == d2`
- **Arithmetic constraints**: `d3 == d1 + d2`, `d4 == d1 * 2`
- **Divisibility constraints**: `d1 % 8 == 0`

Constraint solving uses:
1. **Unification** for equality constraints
2. **Symbolic arithmetic** for expressions
3. **SMT solver** (optional, for complex constraints) or conservative approximation

**Error Reporting**: When constraints cannot be satisfied, the compiler reports:
- Which operation caused the conflict
- Expected vs. actual shapes
- Source location

Example error:
```
error: shape mismatch in matrix multiplication
  ┌─ examples/mlp.lumen:12:15
  │
12│     let h = matmul(x, W1) + b1
  │              ^^^^^^^^^^^^
  │              │      │
  │              │      expected shape [..., 784, 256]
  │              │      actual shape [..., 512, 256]
  │              │
  │              note: 'x' has shape [batch, 512] (defined at line 10)
  │              note: 'W1' has shape [784, 256] (defined at line 6)
```

---

## 5. Built-in Operations

### 5.1 Tensor Creation

- `zeros(shape: [dim]) -> Tensor`
- `ones(shape: [dim]) -> Tensor`
- `randn(shape: [dim], mean, std) -> Tensor`
- `uniform(shape: [dim], low, high) -> Tensor`

### 5.2 Linear Algebra

- `matmul(a: Tensor, b: Tensor) -> Tensor`
- `transpose(x: Tensor, dim0, dim1) -> Tensor`
- `dot(a: Tensor, b: Tensor) -> Tensor` (vector dot product)

### 5.3 Activation Functions

- `relu(x: Tensor) -> Tensor`
- `sigmoid(x: Tensor) -> Tensor`
- `tanh(x: Tensor) -> Tensor`
- `gelu(x: Tensor) -> Tensor`
- `softmax(x: Tensor, axis) -> Tensor`

### 5.4 Convolution & Pooling

- `conv2d(input, kernel, stride, padding) -> Tensor`
- `max_pool2d(input, kernel_size, stride, padding) -> Tensor`
- `avg_pool2d(input, kernel_size, stride, padding) -> Tensor`

### 5.5 Normalization

- `batch_norm(x, scale, bias, running_mean, running_var, momentum, eps) -> Tensor`
- `layer_norm(x, normalized_shape, weight, bias, eps) -> Tensor`

### 5.6 Loss Functions

- `cross_entropy(logits: Tensor[f32, batch, classes], labels: Tensor[i64, batch]) -> Tensor[f32]`
- `mse_loss(pred, target) -> Tensor[f32]`
- `binary_cross_entropy(pred, target) -> Tensor[f32]`

### 5.7 Reductions

- `sum(x: Tensor, axis: ?int, keepdim: bool) -> Tensor`
- `mean(x: Tensor, axis: ?int, keepdim: bool) -> Tensor`
- `max(x: Tensor, axis: ?int, keepdim: bool) -> Tensor`
- `min(x: Tensor, axis: ?int, keepdim: bool) -> Tensor`

### 5.8 Shape Manipulation

- `reshape(x: Tensor, shape: [dim]) -> Tensor`
- `flatten(x: Tensor, start_dim, end_dim) -> Tensor`
- `unsqueeze(x: Tensor, dim) -> Tensor`
- `squeeze(x: Tensor, dim) -> Tensor`
- `concat(tensors: [Tensor], axis) -> Tensor`

---

## 6. Example Programs

### 6.1 Multi-Layer Perceptron (MLP)

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
  let loss = cross_entropy(logits, y)
  return loss
}

// Training loop
train {
  use model MLP as net
  dataset mnist(batch=128, shuffle=true) as (images, labels)

  optimizer adam(params=net.params, lr=1e-3, betas=(0.9, 0.999))

  for step in 0..10000 {
    let loss = classification_loss(net, images, labels)
    minimize loss

    if step % 100 == 0 {
      print("step =", step, " loss =", loss)
    }
  }
}
```

### 6.2 Convolutional Neural Network (CNN)

```lumen
dim batch
dim channels = 3
dim height = 224
dim width = 224
dim classes = 1000

model SimpleCNN {
  // Conv layer 1: 3 -> 64 channels
  param conv1_weight: Tensor[f32, 64, 3, 7, 7] ~ kaiming()
  param conv1_bias: Tensor[f32, 64] ~ zeros()

  // Conv layer 2: 64 -> 128 channels
  param conv2_weight: Tensor[f32, 128, 64, 3, 3] ~ kaiming()
  param conv2_bias: Tensor[f32, 128] ~ zeros()

  // Fully connected
  param fc_weight: Tensor[f32, 128 * 56 * 56, classes] ~ xavier()
  param fc_bias: Tensor[f32, classes] ~ zeros()

  fn forward(x: Tensor[f32, batch, channels, height, width])
      -> Tensor[f32, batch, classes] {
    // First conv block: 224x224 -> 112x112
    let c1 = conv2d(x, conv1_weight, stride=(2, 2), padding=(3, 3))
    let c1 = c1 + unsqueeze(unsqueeze(conv1_bias, 0), 2)  // broadcast bias
    let c1 = relu(c1)
    let p1 = max_pool2d(c1, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

    // Second conv block: 56x56 -> 56x56
    let c2 = conv2d(p1, conv2_weight, stride=(1, 1), padding=(1, 1))
    let c2 = c2 + unsqueeze(unsqueeze(conv2_bias, 0), 2)
    let c2 = relu(c2)

    // Flatten and FC
    let flat = flatten(c2, start_dim=1, end_dim=-1)
    let out = matmul(flat, fc_weight) + fc_bias

    return out
  }
}

fn train_cnn(model: SimpleCNN,
             x: Tensor[f32, batch, channels, height, width],
             y: Tensor[i64, batch]) -> Tensor[f32] {
  let logits = model.forward(x)
  return cross_entropy(logits, y)
}

train {
  use model SimpleCNN as net
  dataset imagenet(batch=32, shuffle=true, augment=true) as (images, labels)

  optimizer sgd(params=net.params, lr=0.1, momentum=0.9, weight_decay=1e-4)

  for epoch in 0..90 {
    for step in 0..1000 {
      let loss = train_cnn(net, images, labels)
      minimize loss
    }
    print("epoch =", epoch, " completed")
  }
}
```

### 6.3 Transformer Attention Block

```lumen
dim batch
dim seq = 512
dim embed = 768
dim heads = 12
dim head_dim = 64  // embed / heads

model MultiHeadAttention {
  param W_q: Tensor[f32, embed, heads * head_dim] ~ xavier()
  param W_k: Tensor[f32, embed, heads * head_dim] ~ xavier()
  param W_v: Tensor[f32, embed, heads * head_dim] ~ xavier()
  param W_o: Tensor[f32, heads * head_dim, embed] ~ xavier()

  fn forward(x: Tensor[f32, batch, seq, embed]) -> Tensor[f32, batch, seq, embed] {
    // Linear projections
    let Q = matmul(x, W_q)  // [batch, seq, heads * head_dim]
    let K = matmul(x, W_k)
    let V = matmul(x, W_v)

    // Reshape to separate heads: [batch, seq, heads, head_dim]
    let Q = reshape(Q, [batch, seq, heads, head_dim])
    let K = reshape(K, [batch, seq, heads, head_dim])
    let V = reshape(V, [batch, seq, heads, head_dim])

    // Transpose to [batch, heads, seq, head_dim]
    let Q = transpose(Q, 1, 2)
    let K = transpose(K, 1, 2)
    let V = transpose(V, 1, 2)

    // Scaled dot-product attention
    // scores = Q @ K^T / sqrt(head_dim)
    let K_t = transpose(K, 2, 3)  // [batch, heads, head_dim, seq]
    let scores = matmul(Q, K_t)   // [batch, heads, seq, seq]
    let scores = scores / sqrt(head_dim as f32)
    let attn_weights = softmax(scores, axis=-1)

    // Apply attention to values
    let attn_out = matmul(attn_weights, V)  // [batch, heads, seq, head_dim]

    // Transpose back and reshape
    let attn_out = transpose(attn_out, 1, 2)  // [batch, seq, heads, head_dim]
    let attn_out = reshape(attn_out, [batch, seq, heads * head_dim])

    // Output projection
    let out = matmul(attn_out, W_o)

    return out
  }
}
```

---

## 7. Semantic Rules

### 7.1 Model Instantiation

- Models are **types**, not values
- `use model MLP as net` creates an instance with initialized parameters
- Parameters are initialized according to their `~` initializer

### 7.2 Automatic Differentiation

- `minimize expr` triggers reverse-mode AD:
  1. Compute forward pass of `expr`
  2. Generate backward pass computing gradients
  3. Apply optimizer update to all parameters

- Only expressions with scalar output (or reduced to scalar) can be minimized

### 7.3 Dataset Iteration

- `dataset name(args) as pattern` binds a dataset iterator
- Each iteration of training loop automatically fetches next batch
- Pattern destructuring: `as (x, y)` unpacks tuple

### 7.4 Optimizer Semantics

- Optimizers maintain internal state (momentum buffers, etc.)
- `minimize` applies the update rule configured by optimizer

---

## 8. Memory Model (Future)

### 8.1 Ownership (Rust-inspired)

- Tensors have **ownership** semantics
- Pass-by-value transfers ownership (move)
- Pass-by-reference borrows (immutable or mutable)

```lumen
fn consume(x: Tensor[f32, 10]) { ... }  // takes ownership
fn borrow(x: &Tensor[f32, 10]) { ... }   // borrows immutably
fn mutate(x: &mut Tensor[f32, 10]) { ... }  // borrows mutably
```

### 8.2 In-place Operations

- Operations can be marked `inplace` for memory efficiency
- Type system ensures no aliasing conflicts

---

## 9. Future Extensions

### 9.1 Schedule DSL

Separate algorithm from schedule:

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

### 9.2 Distributed Parallelism

```lumen
mesh devices[gpus: 8] {
  shard dim batch across gpus

  train {
    // training code automatically distributed
  }
}
```

### 9.3 Mixed Precision

```lumen
with mixed_precision(compute=f16, params=f32) {
  // automatic casting and loss scaling
}
```

---

## 10. Standard Library Organization

```
lumen::core         // Tensor, dim, basic ops
lumen::nn           // Neural network layers
lumen::optim        // Optimizers (SGD, Adam, etc.)
lumen::data         // Dataset utilities
lumen::transforms   // Data augmentation
lumen::distributed  // Multi-device support (future)
```

---

**Version**: 0.1 (Draft)
**Last Updated**: 2025-11-15
**Status**: Initial Design Phase
