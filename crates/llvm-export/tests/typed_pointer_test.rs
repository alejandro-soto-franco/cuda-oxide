/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for typed-pointer NVVM export (pre-Blackwell, issue #98).

use llvm_export::export::{
    ExportBackendConfig, NvvmExportConfig, PtxExportConfig, export_module_to_string_with_config,
};
use llvm_export::ops::{
    AllocaOp, ConstantOp, FuncOp, GepIndex, GetElementPtrOp, LoadOp, ReturnOp, StoreOp,
};
use llvm_export::types::{FuncType, PointerType, VoidType};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        ops::ModuleOp,
        types::{IntegerType, Signedness},
    },
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    value::Value,
};

/// Build `define void @f(<ptr addrspace(A)> %arg)` and return the pieces a test
/// needs to attach a memory op. The caller inserts ops into `entry`, then a
/// `ReturnOp`, then inserts `func` into `mblock` before exporting.
#[allow(clippy::type_complexity)]
fn ptr_param_fn(
    ctx: &mut Context,
    addrspace: u32,
) -> (ModuleOp, Ptr<BasicBlock>, FuncOp, Ptr<BasicBlock>, Value) {
    let module = ModuleOp::new(ctx, "m".try_into().unwrap());
    let mblock = module_block(ctx, &module);
    let void_ty = VoidType::get(ctx);
    let ptr_ty = PointerType::get(ctx, addrspace).into();
    let func_ty = FuncType::get(ctx, void_ty.to_ptr(), vec![ptr_ty], false);
    let func = FuncOp::new(ctx, "f".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(ctx);
    let ptr_val = entry.deref(ctx).get_argument(0);
    (module, mblock, func, entry, ptr_val)
}

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

#[test]
fn typed_mode_load_bitcasts_pointer_operand() {
    let mut ctx = Context::new();
    let (module, mblock, func, entry, ptr_val) = ptr_param_fn(&mut ctx, 0);
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    LoadOp::new(&mut ctx, ptr_val, i32_ty.to_ptr())
        .get_operation()
        .insert_at_back(entry, &ctx);
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
        typed.contains("bitcast i8* ") && typed.contains(" to i32*"),
        "typed load should bitcast the pointer to i32*, got:\n{typed}"
    );
    assert!(
        typed.contains("load i32, i32* %__ptrcast."),
        "typed load should load through the typed pointer, got:\n{typed}"
    );

    let opaque =
        export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::default()).unwrap();
    assert!(
        opaque.contains("load i32, ptr"),
        "opaque load should use ptr, got:\n{opaque}"
    );
}

#[test]
fn typed_mode_store_bitcasts_pointer_operand() {
    let mut ctx = Context::new();
    let (module, mblock, func, entry, ptr_val) = ptr_param_fn(&mut ctx, 0);
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let load = LoadOp::new(&mut ctx, ptr_val, i32_ty.to_ptr());
    let loaded = load.get_operation().deref(&ctx).get_result(0);
    load.get_operation().insert_at_back(entry, &ctx);
    StoreOp::new(&mut ctx, loaded, ptr_val)
        .get_operation()
        .insert_at_back(entry, &ctx);
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
    let store_line = typed
        .lines()
        .find(|l| l.trim_start().starts_with("store "))
        .expect("a store line");
    assert!(
        store_line.contains("i32* %__ptrcast."),
        "typed store should store through a typed pointer, got store line:\n{store_line}\nfull:\n{typed}"
    );

    let opaque =
        export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::default()).unwrap();
    let store_line_o = opaque
        .lines()
        .find(|l| l.trim_start().starts_with("store "))
        .expect("a store line");
    assert!(
        store_line_o.contains(", ptr"),
        "opaque store should use ptr, got store line:\n{store_line_o}"
    );
}

#[test]
fn typed_mode_alloca_yields_i8_pointer() {
    let mut ctx = Context::new();
    let (module, mblock, func, entry, _ptr_val) = ptr_param_fn(&mut ctx, 0);
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&mut ctx, 64, Signedness::Signless);
    let one = {
        let apint =
            pliron::utils::apint::APInt::from_i64(1, std::num::NonZeroUsize::new(64).unwrap());
        let attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, apint);
        let c = ConstantOp::new(&mut ctx, attr.into());
        c.get_operation().insert_at_back(entry, &ctx);
        c.get_operation().deref(&ctx).get_result(0)
    };
    let alloca = AllocaOp::new(&mut ctx, i32_ty.to_ptr(), one);
    alloca.get_operation().insert_at_back(entry, &ctx);
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
        typed.contains("%__ptrcast.") && typed.contains("alloca i32"),
        "typed alloca should allocate into a raw temp, got:\n{typed}"
    );
    assert!(
        typed.contains("bitcast i32* %__ptrcast.") && typed.contains(" to i8*"),
        "typed alloca should bitcast its result to i8*, got:\n{typed}"
    );

    let opaque =
        export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::default()).unwrap();
    let alloca_line = opaque
        .lines()
        .find(|l| l.contains("alloca i32"))
        .expect("an alloca line");
    assert!(
        !alloca_line.contains("bitcast"),
        "opaque alloca should not bitcast, got:\n{alloca_line}"
    );
}

#[test]
fn typed_mode_gep_casts_base_and_result() {
    let mut ctx = Context::new();
    let (module, mblock, func, entry, ptr_val) = ptr_param_fn(&mut ctx, 0);
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let gep = GetElementPtrOp::new(
        &mut ctx,
        ptr_val,
        vec![GepIndex::Constant(0)],
        i32_ty.to_ptr(),
    );
    gep.get_operation().insert_at_back(entry, &ctx);
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
        typed.contains("getelementptr inbounds i32, i32* %__ptrcast."),
        "typed GEP should index through a typed base pointer, got:\n{typed}"
    );
    assert!(
        typed.contains("bitcast i32* %__ptrcast.") && typed.contains(" to i8*"),
        "typed GEP should cast its result back to i8*, got:\n{typed}"
    );

    let opaque =
        export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::default()).unwrap();
    assert!(
        opaque.contains("getelementptr inbounds i32, ptr"),
        "opaque GEP should index through ptr, got:\n{opaque}"
    );
}
