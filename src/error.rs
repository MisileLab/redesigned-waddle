// Error types for the Lumen compiler

use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum LumenError {
    #[error("Parse error at line {line}, column {column}: {message}")]
    ParseError {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("Type error: {message}")]
    TypeError {
        message: String,
    },

    #[error("Name resolution error: {message}")]
    NameError {
        message: String,
    },

    #[error("Code generation error: {message}")]
    CodegenError {
        message: String,
    },
}

pub type Result<T> = std::result::Result<T, LumenError>;
