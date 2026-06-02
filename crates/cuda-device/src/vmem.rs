/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Vectorised global-memory loads.
//!
//! The default codegen path lowers an indexed `&[f32]` read to a scalar
//! generic `ld.b32`, which moves 32 bits per memory transaction and leaves
//! roughly three quarters of every 128-byte sector unused. For throughput-
//! bound kernels (large-batch FFT, stencils, any streaming read) the fix is a
//! single 128-bit vector transaction per thread.
//!
//! [`ld_global_v4_f32`] is that transaction: it lowers to
//! `cvta.to.global.u64` + `ld.global.v4.f32`, returning four contiguous f32
//! values in registers as a [`CuSimd<f32, 4>`]. Compared with the u64-pair
//! trick (two complex values bitcast through one `ld.global.v2`) it doubles
//! the per-instruction payload and keeps the kernel source in plain `&[f32]`
//! terms with no `from_bits` juggling.
//!
//! # Alignment
//!
//! `ld.global.v4.f32` requires the address to be 16-byte aligned. The caller
//! is responsible for this: index a slice at a multiple of 4, or load from the
//! base of a `[f32; 4]`-aligned region. A misaligned address is undefined
//! behaviour on the device (it does NOT trap to a scalar fallback).
//!
//! # Example
//!
//! ```rust,ignore
//! use cuda_device::{vmem, thread};
//!
//! #[kernel]
//! pub fn copy4(src: &[f32], mut dst: DisjointSlice<f32>) {
//!     let base = thread::index_1d().get() * 4;
//!     // One 128-bit transaction instead of four ld.b32.
//!     let v = vmem::ld_global_v4_f32(unsafe { src.as_ptr().add(base) });
//!     let (a, b, c, d) = v.xyzw();
//!     // ...store a, b, c, d...
//! }
//! ```

use crate::cusimd::CuSimd;

/// Load four contiguous `f32` from global memory in one 128-bit transaction.
///
/// Lowers to `cvta.to.global.u64` followed by `ld.global.v4.f32`, producing
/// `CuSimd<f32, 4>` `{ptr[0], ptr[1], ptr[2], ptr[3]}` in registers.
///
/// # Safety / preconditions
///
/// - `ptr` must point to at least four readable `f32` in global memory.
/// - `ptr` must be 16-byte aligned (element index a multiple of 4).
///
/// Both are the caller's responsibility; violating either is undefined
/// behaviour on the device. The function is recognised by name in the GPU
/// codegen path and has no host implementation.
#[inline(never)]
pub fn ld_global_v4_f32(ptr: *const f32) -> CuSimd<f32, 4> {
    let _ = ptr;
    unreachable!("ld_global_v4_f32 called outside CUDA kernel context")
}
