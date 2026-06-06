/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Exporter state and kernel bookkeeping.

use pliron::{basic_block::BasicBlock, context::Ptr, value::Value};
use std::collections::HashMap;

/// Map from block to its predecessors with the values passed to each predecessor.
/// Used for PHI node generation when exporting to LLVM IR.
pub(super) type PredecessorMap = HashMap<Ptr<BasicBlock>, Vec<(Ptr<BasicBlock>, Vec<Value>)>>;

/// Cluster dimensions for a kernel (from `#[cluster(x,y,z)]` attribute).
pub(super) struct KernelClusterConfig {
    pub(super) name: String,
    pub(super) dim_x: u32,
    pub(super) dim_y: u32,
    pub(super) dim_z: u32,
}

/// Launch bounds for a kernel (from `#[launch_bounds(max, min)]` attribute).
pub(super) struct KernelLaunchBounds {
    pub(super) name: String,
    pub(super) max_threads: u32,
    pub(super) min_blocks: Option<u32>, // None if not specified (0 in attribute)
}

/// Basic kernel info (for backends that need annotations for all kernels).
pub(super) struct KernelInfo {
    pub(super) name: String,
}

pub(super) struct ModuleExportState<'a> {
    pub(super) ctx: &'a pliron::context::Context,
    /// Track if any convergent operations were used (for emitting attributes section)
    pub(super) convergent_used: bool,
    /// Track kernels with cluster configurations for nvvm.annotations metadata
    pub(super) cluster_kernels: Vec<KernelClusterConfig>,
    /// Track kernels with launch bounds for nvvm.annotations metadata
    pub(super) launch_bounds_kernels: Vec<KernelLaunchBounds>,
    /// Track ALL kernels (for backends that require annotations for every kernel)
    pub(super) all_kernels: Vec<KernelInfo>,
    /// Whether to track all kernels (set by backend config)
    pub(super) track_all_kernels: bool,
    /// Whether to print `ptx_kernel` on kernel definitions.
    pub(super) emit_ptx_kernel_keyword: bool,
    /// Track device function names for @llvm.used (standalone device fn compilation)
    pub(super) device_functions: Vec<String>,
    /// Render typed pointers (`i8 addrspace(N)*`) for pre-Blackwell libNVVM.
    pub(super) typed_pointers: bool,
    /// Monotonic counter for synthesized pointer-bitcast temporaries.
    pub(super) ptr_cast_counter: usize,
    /// Map from exported function name to its pointer-type string (for example
    /// `void (i8*, i64)*`). Populated during function export and used by typed
    /// mode to reference functions in `@llvm.used` and `nvvm.annotations`, where
    /// a function symbol needs its real pointer type rather than a bare `i8*`.
    pub(super) fn_ptr_types: HashMap<String, String>,
    /// Map from global symbol name to its (value-type string, address space),
    /// for example `("[256 x float]", 3)`. Populated before function export and
    /// used by typed mode to reference a global through a constant bitcast to the
    /// uniform `i8*`, since `@g` carries the global's real pointer type.
    pub(super) global_value_types: HashMap<String, (String, u32)>,
}

impl<'a> ModuleExportState<'a> {
    pub(super) fn new(
        ctx: &'a pliron::context::Context,
        track_all_kernels: bool,
        emit_ptx_kernel_keyword: bool,
        typed_pointers: bool,
    ) -> Self {
        Self {
            ctx,
            convergent_used: false,
            cluster_kernels: Vec::new(),
            launch_bounds_kernels: Vec::new(),
            all_kernels: Vec::new(),
            track_all_kernels,
            emit_ptx_kernel_keyword,
            device_functions: Vec::new(),
            typed_pointers,
            ptr_cast_counter: 0,
            fn_ptr_types: HashMap::new(),
            global_value_types: HashMap::new(),
        }
    }

    /// Allocate a fresh SSA name for a synthesized pointer bitcast. The prefix
    /// keeps it distinct from the `%vN` names assigned by the value pre-pass.
    pub(super) fn fresh_ptr_cast_name(&mut self) -> String {
        let n = self.ptr_cast_counter;
        self.ptr_cast_counter += 1;
        format!("%__ptrcast.{n}")
    }

    /// Check if a function name is a known convergent intrinsic.
    ///
    /// These intrinsics require warp-synchronous execution semantics and must
    /// be marked convergent to prevent LLVM from applying optimizations that
    /// would break GPU synchronization (like duplicating them into divergent branches).
    pub(super) fn is_convergent_intrinsic(name: &str) -> bool {
        // Block-level barriers
        name == "llvm.nvvm.barrier0"
            || name.starts_with("llvm.nvvm.barrier")
            // mbarrier operations
            || name.starts_with("llvm.nvvm.mbarrier")
            // Warp shuffles (though LLVM usually handles these)
            || name.starts_with("llvm.nvvm.shfl")
            // Warp votes
            || name.starts_with("llvm.nvvm.vote")
            // Async bulk operations (TMA)
            || name.starts_with("llvm.nvvm.cp.async.bulk")
    }
}
