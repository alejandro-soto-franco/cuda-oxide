/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Vectorised global-memory load lowering: `dialect-nvvm` → inline PTX.
//!
//! The default scalar path emits a generic `ld.b32` per `f32`, which the
//! profiler shows moving 32 bits per 128-byte sector (~22% sector utilisation)
//! on streaming kernels. `LdGlobalV4F32Op` instead emits one 128-bit
//! `ld.global.v4.f32`, four times the payload per transaction.

use dialect_llvm::ops as llvm;
use dialect_llvm::types as llvm_types;
use pliron::builtin::types::FP32Type;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::TypeObj;

/// Convert `nvvm.ld_global_v4_f32` to inline PTX.
///
/// Emits `cvta.to.global.u64` so a generic kernel-argument pointer resolves to
/// the global window, then `ld.global.v4.f32` reading four contiguous f32 in
/// one 128-bit transaction. Returns the four lanes as separate SSA values; the
/// importer has already wrapped them back into the `CuSimd<f32, 4>` struct.
///
/// PTX: `cvta.to.global.u64 %gp, $4; ld.global.v4.f32 {$0,$1,$2,$3}, [%gp];`
///
/// This is a per-thread load (NOT warp-collective), so the inline asm is
/// non-convergent. The `~{memory}` clobber keeps it ordered with respect to
/// other memory operations and prevents the optimiser from reusing a stale
/// result across an aliasing store.
pub(crate) fn convert_ld_global_v4_f32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let ptr = op.deref(ctx).get_operand(0);

    let f32_ty = FP32Type::get(ctx);
    let field_types: Vec<Ptr<TypeObj>> = (0..4).map(|_| f32_ty.into()).collect();
    let struct_ty = llvm_types::StructType::get_unnamed(ctx, field_types);

    let inline_asm = llvm::InlineAsmOp::new(
        ctx,
        struct_ty.into(),
        vec![ptr],
        concat!(
            "{ ",
            ".reg .u64 %gp; ",
            "cvta.to.global.u64 %gp, $4; ",
            "ld.global.v4.f32 {$0,$1,$2,$3}, [%gp]; ",
            "}"
        ),
        "=f,=f,=f,=f,l,~{memory}",
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);

    let struct_result = asm_op.deref(ctx).get_result(0);
    let mut extracted_values = Vec::with_capacity(4);
    for i in 0..4u32 {
        let extract_op = llvm::ExtractValueOp::new(ctx, struct_result, vec![i])
            .map_err(|e| pliron::input_error_noloc!("{}", e))?;
        rewriter.insert_operation(ctx, extract_op.get_operation());
        let field_val = extract_op.get_operation().deref(ctx).get_result(0);
        extracted_values.push(field_val);
    }
    rewriter.replace_operation_with_values(ctx, op, extracted_values);

    Ok(())
}
