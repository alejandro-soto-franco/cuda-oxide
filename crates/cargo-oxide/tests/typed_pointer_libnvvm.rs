/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integration tests for typed-pointer NVVM export (issue #98).
//!
//! These require a CUDA toolkit (libNVVM) and the cuda-oxide codegen backend,
//! so they are `#[ignore]` by default. Run locally with:
//!
//! ```bash
//! env -u CARGO_TARGET_DIR cargo test -p cargo-oxide \
//!     --test typed_pointer_libnvvm -- --ignored
//! ```
//!
//! They shell out to the prebuilt `cargo-oxide` binary to emit NVVM IR, then
//! drive libNVVM through `libnvvm-sys`.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn cargo_oxide_bin() -> PathBuf {
    repo_root().join("target/debug/cargo-oxide")
}

fn example_ll(example: &str) -> PathBuf {
    repo_root()
        .join("crates/rustc-codegen-cuda/examples")
        .join(example)
        .join(format!("{example}.ll"))
}

/// Emit NVVM IR for `example` at `arch` in the given pointer mode (`typed` or
/// `opaque`), returning the path of the generated `.ll`.
fn emit_nvvm_ir(example: &str, arch: &str, ptr_mode: &str) -> PathBuf {
    let status = Command::new(cargo_oxide_bin())
        .args(["build", example, "--emit-nvvm-ir", "--arch", arch])
        .env_remove("CARGO_TARGET_DIR")
        .env("CUDA_OXIDE_PTR_MODE", ptr_mode)
        .current_dir(repo_root())
        .status()
        .expect("run cargo-oxide build");
    assert!(
        status.success(),
        "emit NVVM IR failed for {example} {arch} {ptr_mode}"
    );
    example_ll(example)
}

/// Compile NVVM IR text through libNVVM. `gen_lto` selects LTOIR output;
/// otherwise PTX. Returns the produced bytes, or the libNVVM error string.
fn libnvvm_compile(ll: &PathBuf, compute_arch: &str, gen_lto: bool) -> Result<Vec<u8>, String> {
    let ir = std::fs::read(ll).map_err(|e| e.to_string())?;
    let nvvm = libnvvm_sys::LibNvvm::load().map_err(|e| e.to_string())?;
    let mut prog = libnvvm_sys::Program::new(&nvvm).map_err(|e| e.to_string())?;
    prog.add_module(&ir, "test").map_err(|e| e.to_string())?;
    let arch_opt = format!("-arch={compute_arch}");
    let opts: Vec<&str> = if gen_lto {
        vec![&arch_opt, "-gen-lto"]
    } else {
        vec![&arch_opt]
    };
    prog.compile(&opts).map_err(|e| e.to_string())
}

/// Strip PTX comment lines (the only nondeterministic / cosmetic content).
fn strip_ptx_comments(ptx: &[u8]) -> String {
    String::from_utf8_lossy(ptx)
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Typed-pointer NVVM IR is accepted by libNVVM on pre-Blackwell targets, where
/// opaque pointers are rejected (the #98 floor). sm_70 is excluded: CUDA 13
/// dropped it regardless of pointer mode.
#[test]
#[ignore]
fn pre_blackwell_typed_ir_compiles() {
    for arch in ["sm_75", "sm_80", "sm_86", "sm_90"] {
        let ll = emit_nvvm_ir("array_index", arch, "typed");
        let compute = format!("compute_{}", arch.strip_prefix("sm_").unwrap());
        let result = libnvvm_compile(&ll, &compute, true);
        assert!(
            result.is_ok(),
            "libNVVM rejected typed IR for {arch}: {}",
            result.unwrap_err()
        );
    }
}

/// Typed-pointer export is semantically equivalent to opaque export: on a
/// Blackwell target that accepts both, the optimized PTX is identical (libNVVM
/// lowers away the synthesized bitcasts). LTOIR is pre-link-optimization and so
/// retains the bitcasts; the equivalence check is therefore at the PTX level.
#[test]
#[ignore]
fn typed_and_opaque_ptx_match_on_blackwell() {
    let typed_ll = emit_nvvm_ir("array_index", "sm_120", "typed");
    let typed_ptx = libnvvm_compile(&typed_ll, "compute_120", false).expect("typed PTX");

    let opaque_ll = emit_nvvm_ir("array_index", "sm_120", "opaque");
    let opaque_ptx = libnvvm_compile(&opaque_ll, "compute_120", false).expect("opaque PTX");

    assert_eq!(
        strip_ptx_comments(&typed_ptx),
        strip_ptx_comments(&opaque_ptx),
        "typed and opaque PTX differ on sm_120"
    );
}
