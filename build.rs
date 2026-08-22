extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    // These paths were originally hardcoded to match one specific Docker
    // layout (TASO_HOME=/usr/TASO installed to /usr/local, protobuf from
    // a conda env at /opt/conda/lib). That's neither this project's local
    // dev layout (tensat/, taso/, egg/ as plain sibling checkouts) nor
    // guaranteed to be the eventual GPU/Docker layout either, so they're
    // now overridable via env vars, defaulting to what actually matches a
    // sibling checkout of yycdavid/taso built with `cmake .. && make`
    // (see taso/CMakeLists.txt) -- i.e. taso/build, not an installed
    // prefix.
    let taso_lib_dir =
        env::var("TASO_LIB_DIR").unwrap_or_else(|_| "../taso/build".to_string());
    let taso_include_dir =
        env::var("TASO_INCLUDE_DIR").unwrap_or_else(|_| "../taso/include".to_string());
    // No cross-platform default for this one -- protobuf's install
    // location varies too much by machine/package manager. Falls back to
    // the original Docker image's conda path.
    let protobuf_lib_dir =
        env::var("PROTOBUF_LIB_DIR").unwrap_or_else(|_| "/opt/conda/lib".to_string());

    // Tell cargo to tell rustc to link the libraries.
    println!("cargo:rustc-link-search={}", protobuf_lib_dir);
    println!("cargo:rustc-link-lib=protobuf");
    println!("cargo:rustc-link-search={}", taso_lib_dir);
    println!("cargo:rustc-link-lib=taso_runtime");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        .enable_cxx_namespaces()
        .header("wrapper.h")
        .clang_args(&["-x", "c++", "-std=c++11", "-I", &taso_include_dir])
        .allowlist_type("std::map")
        .allowlist_type("std::set")
        .allowlist_type("std::vector")
        .allowlist_type("taso::Graph")
        .allowlist_type("taso::Tensor")
        .opaque_type("std::.*")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
