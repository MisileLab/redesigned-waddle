use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use lumen::benchmark::StandardBenchmarks;
use lumen::mir::*;
use lumen::hir::{HirBinOp, HirType, HirLiteral, HirDimExpr};

// Simple MIR programs for benchmarking
fn create_simple_matmul() -> MirProgram {
    MirProgram {
        models: vec![],
        functions: vec![MirFunction {
            name: "matmul".to_string(),
            params: vec![
                MirFunctionParam {
                    var_id: 0,
                    name: "A".to_string(),
                    ty: HirType::Tensor {
                        dtype: Box::new(HirType::F32),
                        shape: vec![HirDimExpr::Const(1024), HirDimExpr::Const(1024)],
                    },
                },
                MirFunctionParam {
                    var_id: 1,
                    name: "B".to_string(),
                    ty: HirType::Tensor {
                        dtype: Box::new(HirType::F32),
                        shape: vec![HirDimExpr::Const(1024), HirDimExpr::Const(1024)],
                    },
                },
            ],
            return_type: None,
            body: vec![
                MirStmt::Assign {
                    var_id: 2,
                    name: "C".to_string(),
                    value: MirExpr::BinOp {
                        op: HirBinOp::MatMul,
                        left: Box::new(MirExpr::Var {
                            var_id: 0,
                            name: "A".to_string(),
                        }),
                        right: Box::new(MirExpr::Var {
                            var_id: 1,
                            name: "B".to_string(),
                        }),
                    },
                },
            ],
        }],
    }
}

fn create_simple_relu() -> MirProgram {
    MirProgram {
        models: vec![],
        functions: vec![MirFunction {
            name: "relu".to_string(),
            params: vec![
                MirFunctionParam {
                    var_id: 0,
                    name: "x".to_string(),
                    ty: HirType::Tensor {
                        dtype: Box::new(HirType::F32),
                        shape: vec![HirDimExpr::Const(1024), HirDimExpr::Const(1024)],
                    },
                },
            ],
            return_type: None,
            body: vec![
                MirStmt::Assign {
                    var_id: 1,
                    name: "zero".to_string(),
                    value: MirExpr::Literal(HirLiteral::Float(0.0)),
                },
                MirStmt::Assign {
                    var_id: 2,
                    name: "result".to_string(),
                    value: MirExpr::BinOp {
                        op: HirBinOp::Mul,
                        left: Box::new(MirExpr::Var {
                            var_id: 0,
                            name: "x".to_string(),
                        }),
                        right: Box::new(MirExpr::Var {
                            var_id: 0,
                            name: "x".to_string(),
                        }),
                    },
                },
            ],
        }],
    }
}

fn create_conv2d() -> MirProgram {
    MirProgram {
        models: vec![],
        functions: vec![MirFunction {
            name: "conv2d".to_string(),
            params: vec![
                MirFunctionParam {
                    var_id: 0,
                    name: "input".to_string(),
                    ty: HirType::Tensor {
                        dtype: Box::new(HirType::F32),
                        shape: vec![
                            HirDimExpr::Const(32),
                            HirDimExpr::Const(3),
                            HirDimExpr::Const(224),
                            HirDimExpr::Const(224),
                        ],
                    },
                },
                MirFunctionParam {
                    var_id: 1,
                    name: "weight".to_string(),
                    ty: HirType::Tensor {
                        dtype: Box::new(HirType::F32),
                        shape: vec![
                            HirDimExpr::Const(64),
                            HirDimExpr::Const(3),
                            HirDimExpr::Const(7),
                            HirDimExpr::Const(7),
                        ],
                    },
                },
            ],
            return_type: None,
            body: vec![],
        }],
    }
}

fn bench_mir_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("MIR Compilation");

    let matmul = create_simple_matmul();
    let relu = create_simple_relu();
    let conv = create_conv2d();

    group.bench_function("matmul_1024x1024", |b| {
        b.iter(|| {
            let program = black_box(&matmul);
            // Simulate compilation overhead
            format!("{:?}", program)
        })
    });

    group.bench_function("relu_1024x1024", |b| {
        b.iter(|| {
            let program = black_box(&relu);
            format!("{:?}", program)
        })
    });

    group.bench_function("conv2d_resnet", |b| {
        b.iter(|| {
            let program = black_box(&conv);
            format!("{:?}", program)
        })
    });

    group.finish();
}

fn bench_standard_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("Standard Models");

    group.bench_function("ResNet-50", |b| {
        b.iter(|| {
            let program = StandardBenchmarks::resnet50();
            black_box(program)
        })
    });

    group.bench_function("GPT-2", |b| {
        b.iter(|| {
            let program = StandardBenchmarks::gpt2();
            black_box(program)
        })
    });

    group.bench_function("BERT", |b| {
        b.iter(|| {
            let program = StandardBenchmarks::bert();
            black_box(program)
        })
    });

    group.finish();
}

fn bench_matrix_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Multiplication");

    for size in [64, 128, 256, 512, 1024].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let program = StandardBenchmarks::matmul(size, size, size);
                black_box(program)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_mir_compilation, bench_standard_models, bench_matrix_sizes);
criterion_main!(benches);
