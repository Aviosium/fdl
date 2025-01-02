use std::time::Instant;

use fdl::{Parser, TypeckState};

pub fn main() {
    let start = Instant::now();
    let mut parser = Parser::new();
    let path = std::env::args().nth(1).expect("Please provide a path");
    let content = std::fs::read_to_string(path).expect("Unable to read file");
    let result = match parser.parse(&content) {
        Ok(ast) => ast,
        Err(error) => {
            print!("{}", error.print(parser.get_manager()));
            return;
        }
    };
    let mut typecheck = TypeckState::new();
    match typecheck.check_script(&[result]) {
        Ok(_) => {}
        Err(error) => {
            print!("{}", error.print(parser.get_manager()));
            return;
        }
    }
    println!(
        "Sucessfully checked file in {:0.3}s",
        start.elapsed().as_secs_f64()
    );
}
