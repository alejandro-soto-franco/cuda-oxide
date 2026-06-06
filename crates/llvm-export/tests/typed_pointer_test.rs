/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for typed-pointer NVVM export (pre-Blackwell, issue #98).

use llvm_export::export::{ExportBackendConfig, NvvmExportConfig, PtxExportConfig};

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
