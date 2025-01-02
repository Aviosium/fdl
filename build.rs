use std::{
    fs::{rename, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
};

const TARGET: &str = "#[allow(clippy::too_many_arguments";

fn main() {
    println!("cargo::rerun-if-changed=src/parser/language.lalrpop");
    lalrpop::process_root().unwrap();
    let base_path = PathBuf::from(std::env::var_os("OUT_DIR").expect("missing OUT_DIR variable"));
    let path = base_path.join("parser/language.rs");
    let out_path = base_path.join("parser/language_out.rs");
    let input_file = File::open(&path).expect("Unable to open input file");
    let reader = BufReader::new(input_file);
    let output_file = File::create(&out_path).expect("unable to create output file");
    let mut writer = BufWriter::new(output_file);
    for line_result in reader.lines() {
        let mut line = line_result.expect("invalid line");

        // Check if the line matches the pattern
        if line.starts_with(TARGET) {
            // Modify the line as needed
            line.insert_str(TARGET.len(), ", clippy::type_complexity, clippy::vec_box")
        }

        if line.starts_with("mod") {
            writeln!(
                writer,
                "#[allow(clippy::vec_box, clippy::mixed_attributes_style)]"
            )
            .expect("Unabel to write file");
        }

        // Write the line to the output file
        writeln!(writer, "{}", line).expect("Unable to write file");
    }
    rename(out_path, path).expect("unable to rename file");
}
