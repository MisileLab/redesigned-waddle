// Lumen Compiler Library

pub mod ast;
pub mod parser;
pub mod error;
pub mod resolve;
pub mod typeck;
pub mod hir;
pub mod mir;
pub mod codegen;

// Re-exports
pub use ast::*;
pub use parser::parse;
pub use error::*;
pub use resolve::{NameResolver, SymbolTable};
pub use typeck::{TypeChecker, TypeContext};
