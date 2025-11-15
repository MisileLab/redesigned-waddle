// Lumen Compiler CLI

use clap::{Parser as ClapParser, Subcommand};
use std::fs;
use std::path::PathBuf;

#[derive(ClapParser)]
#[command(name = "lumenc")]
#[command(about = "Lumen compiler for deep learning DSL", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse a Lumen source file
    Parse {
        /// Input file path
        #[arg(value_name = "FILE")]
        input: PathBuf,
    },
    /// Compile a Lumen source file
    Compile {
        /// Input file path
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output file path
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Parse { input } => {
            let result = parse_file(&input);
            match result {
                Ok(()) => println!("✓ Parse successful"),
                Err(e) => {
                    eprintln!("✗ Parse failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Compile { input, output } => {
            let result = compile_file(&input, output);
            match result {
                Ok(()) => println!("✓ Compilation successful"),
                Err(e) => {
                    eprintln!("✗ Compilation failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
    }
}

fn parse_file(path: &PathBuf) -> lumen::Result<()> {
    let source = fs::read_to_string(path)
        .map_err(|e| lumen::LumenError::ParseError {
            message: format!("Failed to read file: {}", e),
            line: 0,
            column: 0,
        })?;

    let program = lumen::parse(&source)?;

    println!("Parsed program:");
    println!("{:#?}", program);

    Ok(())
}

fn compile_file(path: &PathBuf, output: Option<PathBuf>) -> lumen::Result<()> {
    println!("🔍 Reading source file...");
    let source = fs::read_to_string(path)
        .map_err(|e| lumen::LumenError::ParseError {
            message: format!("Failed to read file: {}", e),
            line: 0,
            column: 0,
        })?;

    println!("📝 Parsing...");
    let program = lumen::parse(&source)?;

    println!("🔎 Resolving names...");
    let mut resolver = lumen::NameResolver::new();
    let symbol_table = resolver.resolve(&program)?;

    println!("🎯 Type checking...");
    let mut type_checker = lumen::TypeChecker::new(symbol_table);
    let type_context = type_checker.check(&program)?;

    println!("🏗️  Lowering to HIR...");
    let mut hir_lowering = lumen::hir::HirLowering::new(type_context);
    let hir_program = hir_lowering.lower(&program)?;

    println!("⚙️  Lowering to MIR...");
    let mir_lowering = lumen::mir::MirLowering::new();
    let mir_program = mir_lowering.lower(&hir_program)?;

    println!("🐍 Generating PyTorch code...");
    let mut codegen = lumen::codegen::PyTorchCodegen::new();
    let python_code = codegen.generate(&mir_program)?;

    // Write output
    let output_path = output.unwrap_or_else(|| {
        let mut p = path.clone();
        p.set_extension("py");
        p
    });

    println!("💾 Writing to {:?}...", output_path);
    fs::write(&output_path, python_code)
        .map_err(|e| lumen::LumenError::CodegenError {
            message: format!("Failed to write output: {}", e),
        })?;

    println!("✅ Compilation successful!");
    println!("   Output: {:?}", output_path);

    Ok(())
}

