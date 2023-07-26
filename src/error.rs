use crate::parser::SpannedError;
use std::io;
use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FdlError {
    #[error("file error")]
    File(#[from] io::Error),
    #[error("parsing error")]
    Parser(#[from] SpannedError),
    /// This error is only triggered if the
    #[error("wrong arguments")]
    Runtime(),
}
