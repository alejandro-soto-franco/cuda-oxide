/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Vectorised global-memory load operations.
//!
//! These ops back the [`cuda_device::vmem`] intrinsics. They exist so that a
//! 128-bit `ld.global.v4.f32` is expressible without the default scalar
//! `ld.b32` codegen path, which is the dominant throughput limiter for
//! streaming kernels (see `cuda-device/src/vmem.rs`).
//!
//! | Operation        | PTX                                        | Min SM |
//! |------------------|--------------------------------------------|--------|
//! | `LdGlobalV4F32Op` | `cvta.to.global.u64` + `ld.global.v4.f32` | All    |

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

// =============================================================================
// Vectorised global load
// =============================================================================

/// Vectorised global load: `ld.global.v4.f32`.
///
/// Reads four contiguous `f32` from global memory in a single 128-bit
/// transaction and returns them in four registers. The single operand is a
/// (generic) pointer; the lowering inserts `cvta.to.global.u64` so a generic
/// kernel-argument pointer resolves to the global window before the load.
///
/// # Operands
///
/// - `ptr` (pointer): base address, must be 16-byte aligned.
///
/// # Results
///
/// - 4 × f32 values (lane 0..3 of the loaded vector).
#[pliron_op(
    name = "nvvm.ld_global_v4_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<4>],
)]
pub struct LdGlobalV4F32Op;

impl LdGlobalV4F32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        LdGlobalV4F32Op { op }
    }
}

/// Register vectorised-memory operations with the context.
pub(super) fn register(ctx: &mut Context) {
    LdGlobalV4F32Op::register(ctx);
}
