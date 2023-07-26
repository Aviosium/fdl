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
        ParseError::User { error: msg } => unreachable!(),
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

    pub fn parse(
        &mut self,
        source_path: &str,
        content: &str,
    ) -> Result<ast::TopLevel, SpannedError> {
        let mut span_maker = self.spans.add_source(source_path.to_string());

        self.parser
            .parse(&mut span_maker, content)
            .map_err(|e| convert_parse_error(span_maker, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::spans::SpanManager;

    #[test]
    fn test_basic_script() {
        // This is just to test the language parser, and doesn't make sense
        let script = "\
            let a = 5;
            b=6;
            if a then
                b=8;
            elseif b then
                a=true;
            else
                $a()
            end;
            let c = {a=1,b=true,c=-1};
            let b = [1,2,3,4,5,6,7,8,9,10];
            let a = [1;20];
            a[0];
            b.c;
            function $a(b,)
                c
            end;";
        let mut parser = Parser::new();
        let ast = parser.parse("temp", script).unwrap();
    }
}
