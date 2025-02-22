use spans::SpanMaker;
use std::fmt::Display;

pub(crate) mod ast;
pub(crate) mod spans;
pub use spans::SpannedError;

use crate::parser::language::ChunkParser;
use crate::parser::spans::SpanManager;
use lalrpop_util::{lalrpop_mod, ParseError};
lalrpop_mod!(pub language, "/parser/language.rs");

fn convert_parse_error<T: Display>(
    mut sm: SpanMaker,
    e: ParseError<usize, T, &'static str>,
) -> SpannedError {
    match e {
        ParseError::InvalidToken { location } => {
            SpannedError::new1("SyntaxError: Invalid token", sm.span(location, location))
        }
        ParseError::UnrecognizedEof { location, expected } => SpannedError::new1(
            format!(
                "SyntaxError: Unexpected end of input.\nNote: expected tokens: [{}]\nParse error occurred here:",
                expected.join(", ")
            ),
            sm.span(location, location),
        ),
        ParseError::UnrecognizedToken { token, expected } => SpannedError::new1(
            format!(
                "SyntaxError: Unexpected token {}\nNote: expected tokens: [{}]\nParse error occurred here:",
                token.1,
                expected.join(", ")
            ),
            sm.span(token.0, token.2),
        ),
        ParseError::ExtraToken { token } => {
            SpannedError::new1("SyntaxError: Unexpected extra token", sm.span(token.0, token.2))
        }
        ParseError::User { error: _msg } => unreachable!(),
    }
}

pub struct Parser {
    parser: ChunkParser,
    spans: SpanManager,
}

impl Parser {
    pub fn new() -> Parser {
        Parser {
            parser: ChunkParser::new(),
            spans: SpanManager::default(),
        }
    }

    pub fn parse(&mut self, content: &str) -> Result<ast::TopLevel, SpannedError> {
        let mut span_maker = self.spans.add_source(content.to_string());

        self.parser
            .parse(&mut span_maker, content)
            .map_err(|e| convert_parse_error(span_maker, e))
    }

    pub fn get_manager(&self) -> &SpanManager {
        &self.spans
    }
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}
