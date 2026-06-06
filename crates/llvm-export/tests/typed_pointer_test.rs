/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for typed-pointer NVVM export (pre-Blackwell, issue #98).

use llvm_export::export::{
    export_module_to_string_with_config, ExportBackendConfig, NvvmExportConfig, PtxExportConfig,
};
use llvm_export::ops::{FuncOp, ReturnOp};
use llvm_export::types::{FuncType, PointerType, VoidType};
use pliron::{
    basic_block::BasicBlock,
    builtin::ops::ModuleOp,
    context::Context,
    linked_list::ContainsLinkedList,
    op::Op,
};

/// Get the module's first block, creating one if the module is empty.
fn module_block(ctx: &mut Context, module: &ModuleOp) -> pliron::context::Ptr<BasicBlock> {
    let region = module.get_operation().deref(ctx).get_region(0);
    if let Some(block) = region.deref(ctx).iter(ctx).next() {
        return block;
    }
    let block = BasicBlock::new(ctx, None, vec![]);
    block.insert_at_back(region, ctx);
    block
}

#[test]
fn config_typed_pointers_flag_round_trips() {
    assert!(!PtxExportConfig.typed_pointers());
    assert!(!NvvmExportConfig::default().typed_pointers());
    assert!(
        NvvmExportConfig {
            typed_pointers: true
        }
        .typed_pointers()
    );
}

/// A function taking one i32 pointer renders that param as a typed pointer in
/// typed mode and as opaque `ptr` in opaque mode.
#[test]
fn typed_mode_renders_pointer_param_as_typed() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "m".try_into().unwrap());
    let mblock = module_block(&mut ctx, &module);

    let void_ty = VoidType::get(&ctx);
    let ptr_ty = PointerType::get(&mut ctx, 0).into();
    let func_ty = FuncType::get(&mut ctx, void_ty.to_ptr(), vec![ptr_ty], false);
    let func = FuncOp::new(&mut ctx, "f".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(mblock, &ctx);

    let typed = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig {
            typed_pointers: true,
        },
    )
    .unwrap();
    assert!(
        typed.contains("i8*"),
        "typed mode should emit i8*, got:\n{typed}"
    );

    let opaque =
        export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::default()).unwrap();
    assert!(
        opaque.contains("ptr"),
        "opaque mode should still emit ptr, got:\n{opaque}"
    );
    assert!(
        !opaque.contains("i8*"),
        "opaque mode should not emit typed i8*, got:\n{opaque}"
    );
}
