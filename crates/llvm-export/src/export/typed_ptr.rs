/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed-pointer lowering helpers (Approach A: uniform i8* with local bitcasts).
//!
//! In typed mode every pointer renders as `i8 addrspace(N)*`. A precise pointer
//! type is materialized only transiently, at the memory op that needs it, via a
//! synthesized `bitcast`. These helpers emit those casts and compute the one
//! derived quantity the casts need: a GEP result's element type.

use std::collections::HashMap;
use std::fmt::Write;

use pliron::{context::Ptr, r#type::TypeObj, value::Value};

use crate::attributes::GepIndexAttr;
use crate::types::{ArrayType, StructType, VectorType};

use super::state::ModuleExportState;

impl<'a> ModuleExportState<'a> {
    /// Emit `%cast = bitcast i8 addrspace(N)* <ptr_val> to <pointee> addrspace(N)*`
    /// and return the fresh cast name, for use as a typed pointer operand.
    pub(super) fn emit_ptr_cast_to(
        &mut self,
        ptr_val: Value,
        pointee: Ptr<TypeObj>,
        addrspace: u32,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<String, String> {
        let name = self.fresh_ptr_cast_name();
        write!(output, "  {name} = bitcast ").unwrap();
        if addrspace != 0 {
            write!(output, "i8 addrspace({addrspace})* ").unwrap();
        } else {
            write!(output, "i8* ").unwrap();
        }
        self.export_value(ptr_val, value_names, output)?;
        write!(output, " to ").unwrap();
        self.write_typed_ptr(pointee, addrspace, output)?;
        writeln!(output).unwrap();
        Ok(name)
    }

    /// Emit `<result_name> = bitcast <pointee> addrspace(N)* <src_name> to i8 addrspace(N)*`,
    /// restoring the uniform-i8 invariant for a typed pointer result (GEP/alloca).
    pub(super) fn emit_ptr_cast_to_i8(
        &mut self,
        src_name: &str,
        pointee: Ptr<TypeObj>,
        addrspace: u32,
        result_name: &str,
        output: &mut String,
    ) -> Result<(), String> {
        write!(output, "  {result_name} = bitcast ").unwrap();
        self.write_typed_ptr(pointee, addrspace, output)?;
        write!(output, " {src_name} to ").unwrap();
        if addrspace != 0 {
            write!(output, "i8 addrspace({addrspace})*").unwrap();
        } else {
            write!(output, "i8*").unwrap();
        }
        writeln!(output).unwrap();
        Ok(())
    }

    /// Compute the element type a GEP yields, so the result can be bitcast back
    /// to i8*. The first index steps over the pointer (result so far = src elem);
    /// each later index descends into an aggregate.
    pub(super) fn gep_result_elem_type(
        &self,
        src_elem: Ptr<TypeObj>,
        indices: &[GepIndexAttr],
    ) -> Result<Ptr<TypeObj>, String> {
        let mut cur = src_elem;
        for idx in indices.iter().skip(1) {
            let cur_ref = cur.deref(self.ctx);
            if let Some(arr) = cur_ref.downcast_ref::<ArrayType>() {
                cur = arr.elem_type();
            } else if let Some(vec) = cur_ref.downcast_ref::<VectorType>() {
                cur = vec.elem_type();
            } else if let Some(st) = cur_ref.downcast_ref::<StructType>() {
                let field = match idx {
                    GepIndexAttr::Constant(c) => *c as usize,
                    GepIndexAttr::OperandIdx(_) => {
                        return Err("typed-pointer GEP into a struct needs a constant index".into());
                    }
                };
                cur = st
                    .fields()
                    .nth(field)
                    .ok_or("typed-pointer GEP struct index out of range")?;
            } else {
                return Err("typed-pointer GEP descends into a non-aggregate type".into());
            }
        }
        Ok(cur)
    }
}
