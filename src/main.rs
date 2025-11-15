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

fn compile_file(path: &PathBuf, _output: Option<PathBuf>) -> lumen::Result<()> {
    let source = fs::read_to_string(path)
        .map_err(|e| lumen::LumenError::ParseError {
            message: format!("Failed to read file: {}", e),
            line: 0,
            column: 0,
        })?;

    let program = lumen::parse(&source)?;

    println!("Compilation not yet implemented");
    println!("Parsed AST:");
    println!("{:#?}", program);

    Ok(())
}

