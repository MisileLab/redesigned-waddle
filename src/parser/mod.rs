// Parser implementation using pest

use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use crate::ast::*;
use crate::error::{LumenError, Result};

#[derive(Parser)]
#[grammar = "parser/grammar.pest"]
pub struct LumenParser;

pub fn parse(source: &str) -> Result<Program> {
    let pairs = LumenParser::parse(Rule::program, source)
        .map_err(|e| {
            let (line, column) = match e.line_col {
                pest::error::LineColLocation::Pos((l, c)) => (l, c),
                pest::error::LineColLocation::Span((l, c), _) => (l, c),
            };
            LumenError::ParseError {
                message: e.to_string(),
                line,
                column,
            }
        })?;

    let mut declarations = Vec::new();

    for pair in pairs {
        match pair.as_rule() {
            Rule::program => {
                for inner in pair.into_inner() {
                    match inner.as_rule() {
                        Rule::declaration => {
                            declarations.push(parse_declaration(inner)?);
                        }
                        Rule::EOI => break,
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    Ok(Program { declarations })
}

fn parse_declaration(pair: Pair<Rule>) -> Result<Declaration> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::dim_decl => Ok(Declaration::Dim(parse_dim_decl(inner)?)),
        Rule::model_decl => Ok(Declaration::Model(parse_model_decl(inner)?)),
        Rule::function_decl => Ok(Declaration::Function(parse_function_decl(inner)?)),
        _ => Err(LumenError::ParseError {
            message: format!("Unexpected rule: {:?}", inner.as_rule()),
            line: 0,
            column: 0,
        }),
    }
}

fn parse_dim_decl(pair: Pair<Rule>) -> Result<DimDecl> {
    let span = Span {
        start: pair.as_span().start(),
        end: pair.as_span().end(),
    };

    let mut name = String::new();
    let mut value = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::ident => name = inner.as_str().to_string(),
            Rule::int_literal => value = Some(inner.as_str().parse().unwrap()),
            _ => {}
        }
    }

    Ok(DimDecl { name, value, span })
}

fn parse_model_decl(pair: Pair<Rule>) -> Result<ModelDecl> {
    let span = Span {
        start: pair.as_span().start(),
        end: pair.as_span().end(),
    };

    let mut name = String::new();
    let mut members = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::ident => name = inner.as_str().to_string(),
            Rule::model_member => {
                members.push(parse_model_member(inner)?);
            }
            _ => {}
        }
    }

    Ok(ModelDecl { name, members, span })
}

fn parse_model_member(pair: Pair<Rule>) -> Result<ModelMember> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::param_decl => Ok(ModelMember::Param(parse_param_decl(inner)?)),
        Rule::function_decl => Ok(ModelMember::Function(parse_function_decl(inner)?)),
        _ => Err(LumenError::ParseError {
            message: format!("Unexpected model member: {:?}", inner.as_rule()),
            line: 0,
            column: 0,
        }),
    }
}

fn parse_param_decl(pair: Pair<Rule>) -> Result<ParamDecl> {
    let span = Span {
        start: pair.as_span().start(),
        end: pair.as_span().end(),
    };

    let mut name = String::new();
    let mut ty = None;
    let mut initializer = Initializer::Zeros;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::ident => name = inner.as_str().to_string(),
            Rule::type_expr => ty = Some(parse_type(inner)?),
            Rule::initializer => initializer = parse_initializer(inner)?,
            _ => {}
        }
    }

    Ok(ParamDecl {
        name,
        ty: ty.unwrap(),
        initializer,
        span,
    })
}

fn parse_initializer(pair: Pair<Rule>) -> Result<Initializer> {
    let text = pair.as_str().trim();

    if text.starts_with("zeros") {
        Ok(Initializer::Zeros)
    } else if text.starts_with("ones") {
        Ok(Initializer::Ones)
    } else if text.starts_with("xavier") {
        Ok(Initializer::Xavier)
    } else if text.starts_with("kaiming") {
        Ok(Initializer::Kaiming)
    } else if text.starts_with("normal") {
        // Parse normal(mean, std)
        let mut floats = Vec::new();
        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::float_literal {
                floats.push(inner.as_str().parse().unwrap());
            }
        }
        if floats.len() >= 2 {
            Ok(Initializer::Normal { mean: floats[0], std: floats[1] })
        } else {
            Ok(Initializer::Normal { mean: 0.0, std: 1.0 })
        }
    } else if text.starts_with("uniform") {
        // Parse uniform(low, high)
        let mut floats = Vec::new();
        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::float_literal {
                floats.push(inner.as_str().parse().unwrap());
            }
        }
        if floats.len() >= 2 {
            Ok(Initializer::Uniform { low: floats[0], high: floats[1] })
        } else {
            Ok(Initializer::Uniform { low: 0.0, high: 1.0 })
        }
    } else {
        Ok(Initializer::Zeros)
    }
}

fn parse_type(pair: Pair<Rule>) -> Result<Type> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::primitive_type => Ok(Type::Primitive(parse_primitive_type(inner)?)),
        Rule::tensor_type => parse_tensor_type(inner),
        Rule::ident => Ok(Type::Named(inner.as_str().to_string())),
        _ => Err(LumenError::ParseError {
            message: format!("Unexpected type: {:?}", inner.as_rule()),
            line: 0,
            column: 0,
        }),
    }
}

fn parse_primitive_type(pair: Pair<Rule>) -> Result<PrimitiveType> {
    Ok(match pair.as_str() {
        "i32" => PrimitiveType::I32,
        "i64" => PrimitiveType::I64,
        "f32" => PrimitiveType::F32,
        "f64" => PrimitiveType::F64,
        "bool" => PrimitiveType::Bool,
        _ => {
            return Err(LumenError::ParseError {
                message: format!("Unknown primitive type: {}", pair.as_str()),
                line: 0,
                column: 0,
            })
        }
    })
}

fn parse_tensor_type(pair: Pair<Rule>) -> Result<Type> {
    let mut dtype = None;
    let mut dims = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::primitive_type => {
                dtype = Some(Box::new(Type::Primitive(parse_primitive_type(inner)?)));
            }
            Rule::dim_expr => {
                dims.push(parse_dim_expr(inner)?);
            }
            _ => {}
        }
    }

    Ok(Type::Tensor {
        dtype: dtype.unwrap(),
        dims,
    })
}

fn parse_dim_expr(pair: Pair<Rule>) -> Result<DimExpr> {
    let mut terms = Vec::new();
    let mut ops = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::dim_term => terms.push(parse_dim_term(inner)?),
            Rule::add_op => ops.push(BinaryOp::Add),
            Rule::sub_op => ops.push(BinaryOp::Sub),
            _ => {}
        }
    }

    let mut result = terms.into_iter();
    let mut expr = result.next().unwrap();

    for (op, right) in ops.into_iter().zip(result) {
        expr = DimExpr::Binary {
            op,
            left: Box::new(expr),
            right: Box::new(right),
        };
    }

    Ok(expr)
}

fn parse_dim_term(pair: Pair<Rule>) -> Result<DimExpr> {
    let mut factors = Vec::new();
    let mut ops = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::dim_factor => factors.push(parse_dim_factor(inner)?),
            Rule::mul_op => ops.push(BinaryOp::Mul),
            Rule::div_op => ops.push(BinaryOp::Div),
            _ => {}
        }
    }

    let mut result = factors.into_iter();
    let mut expr = result.next().unwrap();

    for (op, right) in ops.into_iter().zip(result) {
        expr = DimExpr::Binary {
            op,
            left: Box::new(expr),
            right: Box::new(right),
        };
    }

    Ok(expr)
}

fn parse_dim_factor(pair: Pair<Rule>) -> Result<DimExpr> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::int_literal => Ok(DimExpr::Const(inner.as_str().parse().unwrap())),
        Rule::ident => Ok(DimExpr::Var(inner.as_str().to_string())),
        Rule::dim_expr => parse_dim_expr(inner),
        _ => Err(LumenError::ParseError {
            message: format!("Unexpected dim factor: {:?}", inner.as_rule()),
            line: 0,
            column: 0,
        }),
    }
}

fn parse_function_decl(pair: Pair<Rule>) -> Result<FunctionDecl> {
    let span = Span {
        start: pair.as_span().start(),
        end: pair.as_span().end(),
    };

    let mut name = String::new();
    let mut params = Vec::new();
    let mut return_type = None;
    let mut body = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::ident => name = inner.as_str().to_string(),
            Rule::param_list => {
                for param_pair in inner.into_inner() {
                    params.push(parse_param(param_pair)?);
                }
            }
            Rule::type_expr => return_type = Some(parse_type(inner)?),
            Rule::block => body = Some(parse_block(inner)?),
            _ => {}
        }
    }

    Ok(FunctionDecl {
        name,
        params,
        return_type,
        body: body.unwrap(),
        span,
    })
}

fn parse_param(pair: Pair<Rule>) -> Result<Param> {
    let span = Span {
        start: pair.as_span().start(),
        end: pair.as_span().end(),
    };

    let mut name = String::new();
    let mut ty = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::ident => name = inner.as_str().to_string(),
            Rule::type_expr => ty = Some(parse_type(inner)?),
            _ => {}
        }
    }

    Ok(Param {
        name,
        ty: ty.unwrap(),
        span,
    })
}

fn parse_block(pair: Pair<Rule>) -> Result<Block> {
    let span = Span {
        start: pair.as_span().start(),
        end: pair.as_span().end(),
    };

    let mut stmts = Vec::new();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::stmt {
            stmts.push(parse_stmt(inner)?);
        }
    }

    Ok(Block { stmts, span })
}

fn parse_stmt(pair: Pair<Rule>) -> Result<Stmt> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::let_stmt => parse_let_stmt(inner),
        Rule::return_stmt => parse_return_stmt(inner),
        Rule::expr_stmt => Ok(Stmt::Expr(parse_expr_stmt(inner)?)),
        _ => Err(LumenError::ParseError {
            message: format!("Unexpected statement: {:?}", inner.as_rule()),
            line: 0,
            column: 0,
        }),
    }
}

fn parse_let_stmt(pair: Pair<Rule>) -> Result<Stmt> {
    let span = Span {
        start: pair.as_span().start(),
        end: pair.as_span().end(),
    };

    let mut name = String::new();
    let mut ty = None;
    let mut value = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::ident => name = inner.as_str().to_string(),
            Rule::type_expr => ty = Some(parse_type(inner)?),
            Rule::expr => value = Some(parse_expr(inner)?),
            _ => {}
        }
    }

    Ok(Stmt::Let {
        name,
        ty,
        value: value.unwrap(),
        span,
    })
}

fn parse_return_stmt(pair: Pair<Rule>) -> Result<Stmt> {
    let span = Span {
        start: pair.as_span().start(),
        end: pair.as_span().end(),
    };

    let mut value = None;

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::expr {
            value = Some(parse_expr(inner)?);
        }
    }

    Ok(Stmt::Return { value, span })
}

fn parse_expr_stmt(pair: Pair<Rule>) -> Result<Expr> {
    let inner = pair.into_inner().next().unwrap();
    parse_expr(inner)
}

fn parse_expr(pair: Pair<Rule>) -> Result<Expr> {
    parse_or_expr(pair.into_inner().next().unwrap())
}

fn parse_or_expr(pair: Pair<Rule>) -> Result<Expr> {
    let mut terms = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::and_expr => terms.push(parse_and_expr(inner)?),
            _ => {}
        }
    }

    let mut result = terms.into_iter();
    let mut expr = result.next().unwrap();

    for right in result {
        let span = Span {
            start: expr.span().start,
            end: right.span().end,
        };
        expr = Expr::Binary {
            op: BinaryOp::Or,
            left: Box::new(expr),
            right: Box::new(right),
            span,
        };
    }

    Ok(expr)
}

fn parse_and_expr(pair: Pair<Rule>) -> Result<Expr> {
    let mut terms = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::cmp_expr => terms.push(parse_cmp_expr(inner)?),
            _ => {}
        }
    }

    let mut result = terms.into_iter();
    let mut expr = result.next().unwrap();

    for right in result {
        let span = Span {
            start: expr.span().start,
            end: right.span().end,
        };
        expr = Expr::Binary {
            op: BinaryOp::And,
            left: Box::new(expr),
            right: Box::new(right),
            span,
        };
    }

    Ok(expr)
}

fn parse_cmp_expr(pair: Pair<Rule>) -> Result<Expr> {
    let mut terms = Vec::new();
    let mut ops = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::add_expr => terms.push(parse_add_expr(inner)?),
            Rule::eq_op => ops.push(BinaryOp::Eq),
            Rule::ne_op => ops.push(BinaryOp::Ne),
            Rule::lt_op => ops.push(BinaryOp::Lt),
            Rule::le_op => ops.push(BinaryOp::Le),
            Rule::gt_op => ops.push(BinaryOp::Gt),
            Rule::ge_op => ops.push(BinaryOp::Ge),
            _ => {}
        }
    }

    let mut result = terms.into_iter();
    let mut expr = result.next().unwrap();

    for (op, right) in ops.into_iter().zip(result) {
        let span = Span {
            start: expr.span().start,
            end: right.span().end,
        };
        expr = Expr::Binary {
            op,
            left: Box::new(expr),
            right: Box::new(right),
            span,
        };
    }

    Ok(expr)
}

fn parse_add_expr(pair: Pair<Rule>) -> Result<Expr> {
    let mut terms = Vec::new();
    let mut ops = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::mul_expr => terms.push(parse_mul_expr(inner)?),
            Rule::add_op => ops.push(BinaryOp::Add),
            Rule::sub_op => ops.push(BinaryOp::Sub),
            _ => {}
        }
    }

    let mut result = terms.into_iter();
    let mut expr = result.next().unwrap();

    for (op, right) in ops.into_iter().zip(result) {
        let span = Span {
            start: expr.span().start,
            end: right.span().end,
        };
        expr = Expr::Binary {
            op,
            left: Box::new(expr),
            right: Box::new(right),
            span,
        };
    }

    Ok(expr)
}

fn parse_mul_expr(pair: Pair<Rule>) -> Result<Expr> {
    let mut terms = Vec::new();
    let mut ops = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::unary_expr => terms.push(parse_unary_expr(inner)?),
            Rule::mul_op => ops.push(BinaryOp::Mul),
            Rule::div_op => ops.push(BinaryOp::Div),
            Rule::mod_op => ops.push(BinaryOp::Mod),
            _ => {}
        }
    }

    let mut result = terms.into_iter();
    let mut expr = result.next().unwrap();

    for (op, right) in ops.into_iter().zip(result) {
        let span = Span {
            start: expr.span().start,
            end: right.span().end,
        };
        expr = Expr::Binary {
            op,
            left: Box::new(expr),
            right: Box::new(right),
            span,
        };
    }

    Ok(expr)
}

fn parse_unary_expr(pair: Pair<Rule>) -> Result<Expr> {
    let mut ops = Vec::new();
    let mut operand = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::neg_op => ops.push(UnaryOp::Neg),
            Rule::not_op => ops.push(UnaryOp::Not),
            Rule::postfix_expr => operand = Some(parse_postfix_expr(inner)?),
            _ => {}
        }
    }

    let mut expr = operand.unwrap();

    for op in ops.into_iter().rev() {
        let span = expr.span();
        expr = Expr::Unary {
            op,
            operand: Box::new(expr),
            span,
        };
    }

    Ok(expr)
}

fn parse_postfix_expr(pair: Pair<Rule>) -> Result<Expr> {
    let mut inner_iter = pair.into_inner();
    let mut expr = parse_primary_expr(inner_iter.next().unwrap())?;

    for inner in inner_iter {
        match inner.as_rule() {
            Rule::ident => {
                let span = Span {
                    start: expr.span().start,
                    end: inner.as_span().end(),
                };
                expr = Expr::FieldAccess {
                    object: Box::new(expr),
                    field: inner.as_str().to_string(),
                    span,
                };
            }
            Rule::arg_list => {
                let args = parse_arg_list(inner)?;
                let span = Span {
                    start: expr.span().start,
                    end: args.last().map(|e| e.span().end).unwrap_or(expr.span().end),
                };
                expr = Expr::Call {
                    func: Box::new(expr),
                    args,
                    span,
                };
            }
            _ => {}
        }
    }

    Ok(expr)
}

fn parse_arg_list(pair: Pair<Rule>) -> Result<Vec<Expr>> {
    let mut args = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::expr {
            args.push(parse_expr(inner)?);
        }
    }
    Ok(args)
}

fn parse_primary_expr(pair: Pair<Rule>) -> Result<Expr> {
    let span = Span {
        start: pair.as_span().start(),
        end: pair.as_span().end(),
    };

    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::int_literal => Ok(Expr::Literal(
            Literal::Int(inner.as_str().parse().unwrap()),
            span,
        )),
        Rule::float_literal => Ok(Expr::Literal(
            Literal::Float(inner.as_str().parse().unwrap()),
            span,
        )),
        Rule::bool_literal => {
            let val = inner.as_str() == "true";
            Ok(Expr::Literal(Literal::Bool(val), span))
        }
        Rule::ident => Ok(Expr::Ident(inner.as_str().to_string(), span)),
        Rule::expr => parse_expr(inner),
        _ => Err(LumenError::ParseError {
            message: format!("Unexpected primary expression: {:?}", inner.as_rule()),
            line: 0,
            column: 0,
        }),
    }
}
