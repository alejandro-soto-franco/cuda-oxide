/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Memory operation conversion: `dialect-mir` → `dialect-llvm`.
//!
//! Converts `dialect-mir` memory operations to their `dialect-llvm` equivalents.
//!
//! # Operations
//!
//! | MIR Operation        | LLVM Operation(s)                 | Description                  |
//! |----------------------|-----------------------------------|------------------------------|
//! | `mir.load`           | `llvm.load`                       | Load from pointer            |
//! | `mir.store`          | `llvm.store`                      | Store to pointer             |
//! | `mir.ref`            | `llvm.alloca` + `llvm.store`      | Materialize aggregate in mem |
//! | `mir.ptr_offset`     | `llvm.getelementptr`              | Pointer arithmetic           |
//! | `mir.shared_alloc`   | `llvm.global` + `llvm.addressof`  | Static shared memory         |
//! | `mir.extern_shared`  | `llvm.global` + `llvm.addressof`  | Dynamic shared memory        |
//!
//! # Shared Memory
//!
//! ## Static Shared Memory (`SharedArray<T, N, ALIGN>`)
//!
//! Each static shared memory allocation gets a unique global symbol (`__shared_mem_N`).
//! Multiple allocations in the same or different kernels each have their own symbol
//! with their own size and alignment.
//!
//! ## Dynamic Shared Memory (`DynamicSharedArray<T, ALIGN>`)
//!
//! Dynamic shared memory uses a per-kernel symbol (`__dynamic_smem_{kernel_name}`).
//! Key characteristics:
//!
//! - **Per-kernel symbols**: Each kernel gets its own extern shared symbol
//! - **Pre-computed alignment**: A pre-pass scans all `DynamicSharedArray` calls in a kernel
//!   to determine the maximum alignment before creating the global
//! - **Single pool per kernel**: All `DynamicSharedArray` calls within a kernel share the
//!   same runtime pool (sized by `shared_mem_bytes` at launch)
//!
//! ### PTX Output Example
//!
//! ```ptx
//! ; Kernel with 128-byte aligned dynamic shared memory
//! .extern .shared .align 128 .b8 __dynamic_smem_my_kernel[];
//!
//! ; Another kernel with 16-byte aligned (default)
//! .extern .shared .align 16 .b8 __dynamic_smem_other_kernel[];
//! ```

use crate::context::{DeviceGlobalsMap, DynamicSmemAlignmentMap, SharedGlobalsMap};
use crate::convert::types::convert_type;
use crate::helpers;
use dialect_llvm::ops as llvm;
use dialect_llvm::types::ArrayType;
use dialect_mir::types::MirPtrType;
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeObj, Typed};

fn anyhow_to_pliron(e: anyhow::Error) -> pliron::result::Error {
    pliron::create_error!(
        pliron::location::Location::Unknown,
        pliron::result::ErrorKind::VerificationFailed,
        pliron::result::StringError(e.to_string())
    )
}

/// Convert `mir.store` to `llvm.store`.
///
/// Operand order: `[ptr, value]` - stores `value` to address `ptr`.
/// No result is produced (store is a side effect).
pub(crate) fn convert_store(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let (ptr, val) = match operands.as_slice() {
        [ptr, val] => (*ptr, *val),
        _ => {
            return pliron::input_err_noloc!("Store operation requires exactly 2 operands");
        }
    };

    let llvm_store = llvm::StoreOp::new(ctx, val, ptr);
    rewriter.insert_operation(ctx, llvm_store.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `mir.load` to `llvm.load`.
///
/// Takes a single pointer operand and returns the loaded value.
/// The result type is derived from the MIR operation's result type.
pub(crate) fn convert_load(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let ptr = op.deref(ctx).get_operand(0);
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let llvm_ty = convert_type(ctx, result_ty).map_err(anyhow_to_pliron)?;

    let llvm_load = llvm::LoadOp::new(ctx, ptr, llvm_ty);
    rewriter.insert_operation(ctx, llvm_load.get_operation());
    rewriter.replace_operation(ctx, op, llvm_load.get_operation());

    Ok(())
}

/// Convert `mir.alloca` to `llvm.alloca`.
///
/// `mir.alloca` carries its element type on the result pointer's pointee, and
/// emits a single-element stack slot of that type. We therefore convert the
/// pointee to an LLVM type and emit `llvm.alloca <pointee_ty>, i32 1`.
///
/// No value is stored into the slot; that is the caller's job via subsequent
/// `mir.store` / `llvm.store` ops. This matches the mem2reg-ready translator
/// model where every local is backed by one alloca in the entry block and
/// defs/uses go through `store`/`load` rather than SSA values.
pub(crate) fn convert_alloca(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let mir_pointee = {
        let ty_ref = result_ty.deref(ctx);
        let mir_ptr = ty_ref.downcast_ref::<MirPtrType>().ok_or_else(|| {
            anyhow_to_pliron(anyhow::anyhow!(
                "MirAllocaOp result must be MirPtrType (enforced by verifier)"
            ))
        })?;
        mir_ptr.pointee
    };
    let llvm_pointee = convert_type(ctx, mir_pointee).map_err(anyhow_to_pliron)?;

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let one_apint =
        pliron::utils::apint::APInt::from_i64(1, std::num::NonZeroUsize::new(32).unwrap());
    let one_attr = pliron::builtin::attributes::IntegerAttr::new(i32_ty, one_apint);
    let one_const = llvm::ConstantOp::new(ctx, one_attr.into());
    rewriter.insert_operation(ctx, one_const.get_operation());
    let one_val = one_const.get_operation().deref(ctx).get_result(0);

    let alloca = llvm::AllocaOp::new(ctx, llvm_pointee, one_val);
    rewriter.insert_operation(ctx, alloca.get_operation());
    rewriter.replace_operation(ctx, op, alloca.get_operation());

    Ok(())
}

/// Convert `mir.ref` — materialize the operand in stack memory via alloca+store.
///
/// `mir.ref` creates a pointer to an SSA value. In SSA form, values don't have
/// addresses, so we must place the value in memory to obtain a pointer.
/// This applies to all types: scalars (e.g. `&factor` where factor is `u32`),
/// aggregates (e.g. `&closure_env`), and pointers (e.g. `&&T`).
pub(crate) fn convert_ref(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operand = op.deref(ctx).get_operand(0);
    let operand_ty = operand.get_type(ctx);

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let one_apint =
        pliron::utils::apint::APInt::from_i64(1, std::num::NonZeroUsize::new(32).unwrap());
    let one_attr = pliron::builtin::attributes::IntegerAttr::new(i32_ty, one_apint);
    let one_const = llvm::ConstantOp::new(ctx, one_attr.into());
    rewriter.insert_operation(ctx, one_const.get_operation());
    let one_val = one_const.get_operation().deref(ctx).get_result(0);

    let alloca = llvm::AllocaOp::new(ctx, operand_ty, one_val);
    rewriter.insert_operation(ctx, alloca.get_operation());
    let alloca_ptr = alloca.get_operation().deref(ctx).get_result(0);

    let store = llvm::StoreOp::new(ctx, operand, alloca_ptr);
    rewriter.insert_operation(ctx, store.get_operation());

    rewriter.replace_operation_with_values(ctx, op, vec![alloca_ptr]);

    Ok(())
}

/// Convert `mir.ptr_offset` to `llvm.getelementptr`.
///
/// Operands: `[ptr, offset]` where offset is an integer index.
/// Uses the pointee type from the MIR pointer type for element sizing.
/// Falls back to i8 element type if pointee type cannot be determined.
pub(crate) fn convert_ptr_offset(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let (ptr, offset) = match operands.as_slice() {
        [ptr, offset] => (*ptr, *offset),
        _ => return pliron::input_err_noloc!("PtrOffset requires exactly 2 operands"),
    };

    let pointee_ty_opt = operands_info
        .lookup_most_recent_of_type::<MirPtrType>(ctx, ptr)
        .map(|mir_ptr| mir_ptr.pointee);

    let elem_ty = if let Some(pointee) = pointee_ty_opt {
        convert_type(ctx, pointee).map_err(anyhow_to_pliron)?
    } else {
        IntegerType::get(ctx, 8, Signedness::Signless).into()
    };

    let llvm_gep = llvm::GetElementPtrOp::new(
        ctx,
        ptr,
        vec![dialect_llvm::ops::GepIndex::Value(offset)],
        elem_ty,
    )?;
    rewriter.insert_operation(ctx, llvm_gep.get_operation());
    rewriter.replace_operation(ctx, op, llvm_gep.get_operation());

    Ok(())
}

/// Convert `mir.shared_alloc` to LLVM global variable in shared address space.
///
/// GPU shared memory is represented as a global variable with address space 3.
/// Uses `shared_globals` to deduplicate: multiple allocations with the same
/// `alloc_key` share the same global.
///
/// Called directly from `MirToLlvmConversionDriver::rewrite` (not through
/// op_cast dispatch) because it needs the cross-function `SharedGlobalsMap`.
pub fn convert_shared_alloc_dc(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    shared_globals: &mut SharedGlobalsMap,
) -> Result<()> {
    use pliron::builtin::attributes::{IntegerAttr, TypeAttr};

    let (alloc_key, mir_elem_type, size, alignment) = {
        let shared_alloc_op = dialect_mir::ops::MirSharedAllocOp::new(op);
        let op_ref = op.deref(ctx);

        let alloc_key: Option<String> = shared_alloc_op
            .get_attr_alloc_key(ctx)
            .map(|s| String::from((*s).clone()));

        let elem_type_attr = op_ref
            .attributes
            .0
            .get(&"elem_type".try_into().unwrap())
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "MirSharedAllocOp missing elem_type attribute"
                ))
            })?;
        let elem_type_attr = elem_type_attr
            .downcast_ref::<TypeAttr>()
            .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("elem_type is not a TypeAttr")))?;
        let mir_elem_type = elem_type_attr.get_type(ctx);

        let size_attr = op_ref
            .attributes
            .0
            .get(&"size".try_into().unwrap())
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!("MirSharedAllocOp missing size attribute"))
            })?;
        let size_attr = size_attr
            .downcast_ref::<IntegerAttr>()
            .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("size is not an IntegerAttr")))?;
        let size = size_attr.value().to_u64();

        let alignment = shared_alloc_op.get_alignment_value(ctx).unwrap_or(0);

        (alloc_key, mir_elem_type, size, alignment)
    };

    // Cache hit only when the op carries a key AND that key is already in
    // `shared_globals`. `as_ref()` borrows for the if-let scope so the else
    // branch can still move `alloc_key` into `create_shared_global` (which
    // takes ownership and inserts it into the cache).
    let global_name = if let Some(key) = alloc_key.as_ref()
        && let Some(existing_name) = shared_globals.get(key)
    {
        existing_name.clone()
    } else {
        create_shared_global(
            ctx,
            op,
            shared_globals,
            mir_elem_type,
            size,
            alignment,
            alloc_key,
        )?
    };

    let address_of_op = llvm::AddressOfOp::new(ctx, global_name, 3);
    rewriter.insert_operation(ctx, address_of_op.get_operation());
    rewriter.replace_operation(ctx, op, address_of_op.get_operation());

    Ok(())
}

/// Create a shared memory global variable in the module.
///
/// Creates an LLVM global variable with:
/// - Array type: `[size x element_type]`
/// - Address space 3 (shared memory)
/// - Optional alignment
/// - Unique generated name (`__shared_mem_N`)
///
/// The global is inserted at the front of the module block. When
/// `alloc_key` is `Some`, the key is moved into `shared_globals` so that
/// later allocations with the same key reuse this global (caller is
/// expected to have already checked the cache for a hit).
fn create_shared_global(
    ctx: &mut Context,
    op: Ptr<Operation>,
    shared_globals: &mut SharedGlobalsMap,
    mir_elem_type: Ptr<TypeObj>,
    size: u64,
    alignment: u64,
    alloc_key: Option<String>,
) -> Result<pliron::identifier::Identifier> {
    let llvm_elem_type = convert_type(ctx, mir_elem_type).map_err(anyhow_to_pliron)?;
    let array_type = ArrayType::get(ctx, llvm_elem_type, size);

    static SHARED_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let counter = SHARED_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let name: pliron::identifier::Identifier =
        format!("__shared_mem_{counter}").try_into().unwrap();

    let global_op = if alignment > 0 {
        llvm::GlobalOp::new_with_alignment(ctx, name.clone(), array_type.into(), alignment)
    } else {
        llvm::GlobalOp::new(ctx, name.clone(), array_type.into())
    };
    global_op.set_address_space(ctx, dialect_llvm::types::address_space::SHARED);

    let parent_block = op
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Op has no parent block")))?;
    let module_op = helpers::get_module_from_block(ctx, parent_block).map_err(anyhow_to_pliron)?;
    let region = module_op.deref(ctx).get_region(0);
    let module_block = region
        .deref(ctx)
        .iter(ctx)
        .next()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Module is empty")))?;

    global_op.get_operation().insert_at_front(module_block, ctx);

    if let Some(key) = alloc_key {
        shared_globals.insert(key, name.clone());
    }

    Ok(name)
}

/// Convert `mir.global_alloc` to an LLVM global in CUDA global memory.
///
/// Ordinary Rust `static` / `static mut` values have grid scope and
/// application lifetime, so they live in address space 1 rather than the
/// per-block shared-memory address space.
pub fn convert_global_alloc_dc(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    device_globals: &mut DeviceGlobalsMap,
) -> Result<()> {
    use pliron::builtin::attributes::{StringAttr, TypeAttr};

    let (global_key, mir_global_type, alignment, addr_space) = {
        let global_op = dialect_mir::ops::MirGlobalAllocOp::new(op);
        let op_ref = op.deref(ctx);

        let global_key_attr = op_ref
            .attributes
            .0
            .get(&"global_key".try_into().unwrap())
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "MirGlobalAllocOp missing global_key attribute"
                ))
            })?;
        let global_key_attr = global_key_attr
            .downcast_ref::<StringAttr>()
            .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("global_key is not a StringAttr")))?;
        let global_key = String::from((*global_key_attr).clone());

        let global_type_attr = op_ref
            .attributes
            .0
            .get(&"global_type".try_into().unwrap())
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "MirGlobalAllocOp missing global_type attribute"
                ))
            })?;
        let global_type_attr = global_type_attr
            .downcast_ref::<TypeAttr>()
            .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("global_type is not a TypeAttr")))?;
        let mir_global_type = global_type_attr.get_type(ctx);

        let alignment = global_op.get_alignment_value(ctx).unwrap_or(0);

        // Read the address space the op's result already carries — set by
        // mir-importer based on the static's type (`ConstantMemory<T>` → 4,
        // ordinary → 1). The dialect verifier accepts both.
        let res_ty = op_ref.get_result(0).get_type(ctx);
        let addr_space = res_ty
            .deref(ctx)
            .downcast_ref::<dialect_mir::types::MirPtrType>()
            .map(|p| {
                if p.address_space == dialect_mir::types::address_space::CONSTANT {
                    dialect_llvm::types::address_space::CONSTANT
                } else {
                    dialect_llvm::types::address_space::GLOBAL
                }
            })
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "MirGlobalAllocOp result is not a MirPtrType"
                ))
            })?;

        (global_key, mir_global_type, alignment, addr_space)
    };

    let global_name = if let Some(existing_name) = device_globals.get(&global_key) {
        existing_name.clone()
    } else {
        create_device_global(
            ctx,
            op,
            device_globals,
            &global_key,
            mir_global_type,
            alignment,
            addr_space,
        )?
    };

    let address_of_op = llvm::AddressOfOp::new(ctx, global_name, addr_space);
    rewriter.insert_operation(ctx, address_of_op.get_operation());
    rewriter.replace_operation(ctx, op, address_of_op.get_operation());

    Ok(())
}

fn create_device_global(
    ctx: &mut Context,
    op: Ptr<Operation>,
    device_globals: &mut DeviceGlobalsMap,
    global_key: &str,
    mir_global_type: Ptr<TypeObj>,
    alignment: u64,
    addr_space: u32,
) -> Result<pliron::identifier::Identifier> {
    let llvm_global_type = convert_type(ctx, mir_global_type).map_err(anyhow_to_pliron)?;

    // Constant-memory globals reuse the Rust-side mangled name so host code can
    // resolve them by name via `cuModuleGetGlobal`. Ordinary device globals
    // are private to the kernel and get a counter-based unique name.
    let name: pliron::identifier::Identifier =
        if addr_space == dialect_llvm::types::address_space::CONSTANT {
            global_key.try_into().map_err(|e| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "constant global_key {global_key:?} is not a valid identifier: {e:?}"
                ))
            })?
        } else {
            static DEVICE_GLOBAL_COUNTER: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            let counter = DEVICE_GLOBAL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            format!("__device_global_{counter}").try_into().unwrap()
        };

    let global_op = if alignment > 0 {
        llvm::GlobalOp::new_with_alignment(ctx, name.clone(), llvm_global_type, alignment)
    } else {
        llvm::GlobalOp::new(ctx, name.clone(), llvm_global_type)
    };
    global_op.set_address_space(ctx, addr_space);

    let parent_block = op
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Op has no parent block")))?;
    let module_op = helpers::get_module_from_block(ctx, parent_block).map_err(anyhow_to_pliron)?;
    let region = module_op.deref(ctx).get_region(0);
    let module_block = region
        .deref(ctx)
        .iter(ctx)
        .next()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Module is empty")))?;

    global_op.get_operation().insert_at_front(module_block, ctx);
    device_globals.insert(global_key.to_string(), name.clone());

    Ok(name)
}

/// Convert `mir.extern_shared` to LLVM extern global variable in shared address space.
///
/// Dynamic (extern) shared memory is represented as an external global variable
/// with address space 3 and zero-length array type `[0 x i8]`. The actual size
/// is determined at kernel launch via `LaunchConfig::shared_mem_bytes`.
///
/// # Per-Kernel Symbols
///
/// Each kernel gets its own dynamic shared memory symbol (`__dynamic_smem_{kernel_name}`).
/// This ensures explicit separation in the generated PTX.
///
/// # Alignment
///
/// The alignment is pre-computed during the lowering pre-pass. All
/// `DynamicSharedArray<T, ALIGN>` calls in a kernel share the same global, which
/// uses the maximum alignment requested by any call.
///
/// # Byte Offset
///
/// - `DynamicSharedArray::get()` / `get_raw()`: offset = 0, returns base pointer
/// - `DynamicSharedArray::offset(N)`: offset = N bytes, returns base + N
///
/// Called directly from `MirToLlvmConversionDriver::rewrite` (not through
/// op_cast dispatch) because it needs cross-function state maps.
pub fn convert_extern_shared_dc(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    shared_globals: &mut SharedGlobalsMap,
    dynamic_smem_alignments: &mut DynamicSmemAlignmentMap,
) -> Result<()> {
    let (byte_offset, alignment) = {
        let extern_shared_op = dialect_mir::ops::MirExternSharedOp::new(op);
        let byte_offset = extern_shared_op.get_byte_offset_value(ctx);
        let alignment = extern_shared_op.get_alignment_value(ctx);
        (byte_offset, alignment)
    };

    let func_name: String = {
        let parent_block = op
            .deref(ctx)
            .get_parent_block()
            .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Op has no parent block")))?;
        let func_op_ptr = helpers::get_parent_func(ctx, parent_block).map_err(anyhow_to_pliron)?;
        let llvm_func = Operation::get_op::<llvm::FuncOp>(func_op_ptr, ctx)
            .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Parent op is not an llvm.func")))?;
        llvm_func.get_symbol_name(ctx).to_string()
    };

    let global_name = get_or_create_extern_shared_global(
        ctx,
        op,
        &func_name,
        shared_globals,
        dynamic_smem_alignments,
        alignment,
    )?;

    let address_of_op = llvm::AddressOfOp::new(ctx, global_name, 3);
    rewriter.insert_operation(ctx, address_of_op.get_operation());

    let base_ptr = address_of_op.get_operation().deref(ctx).get_result(0);

    if byte_offset > 0 {
        let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
        let offset_attr = pliron::builtin::attributes::IntegerAttr::new(
            i64_ty,
            pliron::utils::apint::APInt::from_u64(
                byte_offset,
                std::num::NonZeroUsize::new(64).unwrap(),
            ),
        );
        let offset_const = llvm::ConstantOp::new(ctx, offset_attr.into());
        rewriter.insert_operation(ctx, offset_const.get_operation());
        let offset_value = offset_const.get_operation().deref(ctx).get_result(0);

        let i8_ty = IntegerType::get(ctx, 8, Signedness::Signless);
        let gep_op = llvm::GetElementPtrOp::new(
            ctx,
            base_ptr,
            vec![dialect_llvm::ops::GepIndex::Value(offset_value)],
            i8_ty.into(),
        )?;
        rewriter.insert_operation(ctx, gep_op.get_operation());
        rewriter.replace_operation(ctx, op, gep_op.get_operation());
    } else {
        rewriter.replace_operation(ctx, op, address_of_op.get_operation());
    }

    Ok(())
}

/// Get or create the extern shared memory global for a kernel.
///
/// Creates an LLVM global variable with:
/// - Zero-length array type: `[0 x i8]`
/// - External linkage (no initializer)
/// - Address space 3 (shared memory)
/// - Pre-computed maximum alignment from all DynamicSharedArray calls in the kernel
///
/// Each kernel gets its own dynamic shared memory symbol
/// (`__dynamic_smem_kernel_name`). Uses `shared_globals` for deduplication
/// (only one global per kernel).
fn get_or_create_extern_shared_global(
    ctx: &mut Context,
    op: Ptr<Operation>,
    func_name: &str,
    shared_globals: &mut SharedGlobalsMap,
    dynamic_smem_alignments: &mut DynamicSmemAlignmentMap,
    _requested_alignment: u64,
) -> Result<pliron::identifier::Identifier> {
    let (symbol_name, max_alignment) = dynamic_smem_alignments.get(func_name).cloned().ok_or_else(
        || {
            anyhow_to_pliron(anyhow::anyhow!(
                "Internal error: dynamic shared memory alignment not pre-computed for kernel '{}'. \
                 This should have been done in compute_max_dynamic_smem_alignment.",
                func_name
            ))
        },
    )?;

    let global_created_key = format!("__dynamic_smem_global_created_{}", func_name);
    if shared_globals.contains_key(&global_created_key) {
        return Ok(symbol_name);
    }

    let i8_ty = IntegerType::get(ctx, 8, Signedness::Signless);
    let array_type = ArrayType::get(ctx, i8_ty.into(), 0);

    let global_op = llvm::GlobalOp::new_with_alignment(
        ctx,
        symbol_name.clone(),
        array_type.into(),
        max_alignment,
    );
    global_op.set_address_space(ctx, dialect_llvm::types::address_space::SHARED);

    {
        use dialect_llvm::attributes::LinkageAttr;
        global_op.set_attr_llvm_global_linkage(ctx, LinkageAttr::ExternalLinkage);
    }

    let parent_block = op
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Op has no parent block")))?;
    let module_op = helpers::get_module_from_block(ctx, parent_block).map_err(anyhow_to_pliron)?;
    let region = module_op.deref(ctx).get_region(0);
    let module_block = region
        .deref(ctx)
        .iter(ctx)
        .next()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Module is empty")))?;

    global_op.get_operation().insert_at_front(module_block, ctx);

    shared_globals.insert(global_created_key, symbol_name.clone());

    Ok(symbol_name)
}

#[cfg(test)]
mod tests {
    //! End-to-end lowering tests for `dialect-mir` memory ops.
    //!
    //! The `convert_*` functions in this file take a live
    //! `DialectConversionRewriter`, which is owned by pliron's conversion
    //! driver and not constructible standalone. So each test builds a small
    //! MIR module, runs the full `lower_mir_to_llvm` pass on it, and asserts
    //! the lowered module contains the expected `dialect-llvm` shape — same
    //! pattern as `tests/lowering_test.rs`.

    use super::*;
    use dialect_llvm::op_interfaces::PointerTypeResult;
    use dialect_llvm::ops as llvm;
    use dialect_llvm::types::{PointerType, address_space as llvm_addr};
    use dialect_mir::ops as mir;
    use dialect_mir::types::MirPtrType;
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::{StringAttr, TypeAttr};
    use pliron::builtin::op_interfaces::SymbolOpInterface;
    use pliron::builtin::ops::ModuleOp;
    use pliron::builtin::types::{FunctionType, IntegerType, Signedness};
    use pliron::context::Context;
    use pliron::linked_list::ContainsLinkedList;
    use pliron::op::Op;
    use pliron::operation::Operation;

    fn make_ctx() -> Context {
        let mut ctx = Context::new();
        dialect_llvm::register(&mut ctx);
        dialect_mir::register(&mut ctx);
        dialect_nvvm::register(&mut ctx);
        crate::register(&mut ctx);
        ctx
    }

    /// Build a module with one MirFuncOp `kernel_func` that takes the given
    /// argument types and returns `()`. The function has a single empty
    /// entry block; the caller appends ops (and a `mir.return`) before
    /// running the lowering pass.
    fn build_kernel(
        ctx: &mut Context,
        arg_tys: Vec<Ptr<TypeObj>>,
    ) -> (Ptr<Operation>, Ptr<BasicBlock>) {
        let module = ModuleOp::new(ctx, "test_module".try_into().unwrap());
        let module_ptr = module.get_operation();

        let func_ty = FunctionType::get(ctx, arg_tys.clone(), vec![]);
        let func_op_ptr = Operation::new(
            ctx,
            mir::MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let func = mir::MirFuncOp::new(ctx, func_op_ptr, TypeAttr::new(func_ty.into()));
        func.set_symbol_name(ctx, "kernel_func".try_into().unwrap());

        let region = func.get_operation().deref(ctx).get_region(0);
        let block = BasicBlock::new(ctx, None, arg_tys);
        block.insert_at_back(region, ctx);

        let module_region = module_ptr.deref(ctx).get_region(0);
        let module_block = module_region.deref(ctx).iter(ctx).next().unwrap();
        func.get_operation().insert_at_back(module_block, ctx);

        (module_ptr, block)
    }

    fn append_return(ctx: &mut Context, block: Ptr<BasicBlock>) {
        let op = Operation::new(
            ctx,
            mir::MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        op.insert_at_back(block, ctx);
    }

    /// All blocks of `kernel_func` in the lowered module.
    ///
    /// `convert_func` builds a fresh `llvm.func` with a prologue entry block
    /// (which reconstructs aggregate args and branches to the inlined MIR
    /// region), so the lowered function generally has at least two blocks
    /// and the converted memory ops live in the second one. Tests iterate
    /// across all of them rather than picking a specific block.
    fn kernel_blocks(ctx: &Context, module_ptr: Ptr<Operation>) -> Vec<Ptr<BasicBlock>> {
        let module_op = module_ptr.deref(ctx);
        let region = module_op.get_region(0);
        let module_block = region.deref(ctx).iter(ctx).next().unwrap();
        for op in module_block.deref(ctx).iter(ctx) {
            let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, ctx) else {
                continue;
            };
            if func_op.get_symbol_name(ctx).to_string() != "kernel_func" {
                continue;
            }
            let func_region = func_op.get_operation().deref(ctx).get_region(0);
            return func_region.deref(ctx).iter(ctx).collect();
        }
        panic!("kernel_func not found in lowered module");
    }

    fn module_top_block(ctx: &Context, module_ptr: Ptr<Operation>) -> Ptr<BasicBlock> {
        module_ptr
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .iter(ctx)
            .next()
            .unwrap()
    }

    fn count_ops<T: Op>(ctx: &Context, blocks: &[Ptr<BasicBlock>]) -> usize {
        blocks
            .iter()
            .flat_map(|b| b.deref(ctx).iter(ctx).collect::<Vec<_>>())
            .filter(|op| Operation::get_op::<T>(*op, ctx).is_some())
            .count()
    }

    fn find_first<T: Op>(ctx: &Context, blocks: &[Ptr<BasicBlock>]) -> Option<T> {
        blocks
            .iter()
            .flat_map(|b| b.deref(ctx).iter(ctx).collect::<Vec<_>>())
            .find_map(|op| Operation::get_op::<T>(op, ctx))
    }

    fn ptr_addrspace(ctx: &Context, ty: Ptr<TypeObj>) -> u32 {
        ty.deref(ctx)
            .downcast_ref::<PointerType>()
            .expect("expected llvm.PointerType")
            .address_space
    }

    #[test]
    fn convert_alloca_lowers_to_llvm_alloca() {
        let mut ctx = make_ctx();
        let i32_ty: Ptr<TypeObj> = IntegerType::get(&mut ctx, 32, Signedness::Signless).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![]);

        let alloca_op = Operation::new(
            &mut ctx,
            mir::MirAllocaOp::get_concrete_op_info(),
            vec![mir_ptr_ty.into()],
            vec![],
            vec![],
            0,
        );
        alloca_op.insert_at_back(block, &ctx);
        append_return(&mut ctx, block);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(
            count_ops::<llvm::AllocaOp>(&ctx, &body),
            1,
            "expected exactly one llvm.alloca"
        );
        let alloca = find_first::<llvm::AllocaOp>(&ctx, &body).unwrap();
        // Element type should round-trip through convert_type as i32.
        let elem_ty = alloca.result_pointee_type(&ctx);
        assert!(elem_ty.deref(&ctx).is::<IntegerType>());
    }

    #[test]
    fn convert_store_lowers_to_llvm_store() {
        let mut ctx = make_ctx();
        let i32_ty: Ptr<TypeObj> = IntegerType::get(&mut ctx, 32, Signedness::Signless).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);

        // Kernel takes (ptr, val) so we can store one into the other.
        let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into(), i32_ty]);
        let ptr_val = block.deref(&ctx).get_argument(0);
        let val = block.deref(&ctx).get_argument(1);

        let store_op = Operation::new(
            &mut ctx,
            mir::MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![ptr_val, val],
            vec![],
            0,
        );
        store_op.insert_at_back(block, &ctx);
        append_return(&mut ctx, block);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(
            count_ops::<llvm::StoreOp>(&ctx, &body),
            1,
            "expected one llvm.store"
        );
        // The original mir.store must be gone.
        assert_eq!(count_ops::<mir::MirStoreOp>(&ctx, &body), 0);

        // convert_store swaps operand order: mir.store is [ptr, value] but
        // llvm.store takes (value, ptr). Verify that mapping survived.
        let store = find_first::<llvm::StoreOp>(&ctx, &body).unwrap();
        let addr_ty = store.address_opd(&ctx).get_type(&ctx);
        assert!(addr_ty.deref(&ctx).is::<PointerType>(), "operand 1 is ptr");
    }

    #[test]
    fn convert_load_lowers_to_llvm_load() {
        let mut ctx = make_ctx();
        let i32_ty: Ptr<TypeObj> = IntegerType::get(&mut ctx, 32, Signedness::Signless).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, false);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into()]);
        let ptr_val = block.deref(&ctx).get_argument(0);

        let load_op = Operation::new(
            &mut ctx,
            mir::MirLoadOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![ptr_val],
            vec![],
            0,
        );
        load_op.insert_at_back(block, &ctx);
        append_return(&mut ctx, block);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(count_ops::<llvm::LoadOp>(&ctx, &body), 1);
        assert_eq!(count_ops::<mir::MirLoadOp>(&ctx, &body), 0);
    }

    #[test]
    fn convert_ref_lowers_to_alloca_then_store() {
        let mut ctx = make_ctx();
        let i32_ty: Ptr<TypeObj> = IntegerType::get(&mut ctx, 32, Signedness::Signless).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, false);

        // Take a u32 by value, build `&x`.
        let (module_ptr, block) = build_kernel(&mut ctx, vec![i32_ty]);
        let arg = block.deref(&ctx).get_argument(0);

        let ref_op_ptr = Operation::new(
            &mut ctx,
            mir::MirRefOp::get_concrete_op_info(),
            vec![mir_ptr_ty.into()],
            vec![arg],
            vec![],
            0,
        );
        mir::MirRefOp::new(ref_op_ptr).set_mutable(&mut ctx, false);
        ref_op_ptr.insert_at_back(block, &ctx);
        append_return(&mut ctx, block);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(
            count_ops::<llvm::AllocaOp>(&ctx, &body),
            1,
            "ref must materialize via alloca"
        );
        assert_eq!(
            count_ops::<llvm::StoreOp>(&ctx, &body),
            1,
            "ref must store the value into the alloca"
        );
        assert_eq!(count_ops::<mir::MirRefOp>(&ctx, &body), 0);
    }

    #[test]
    fn convert_ptr_offset_lowers_to_gep_with_pointee_elem_type() {
        let mut ctx = make_ctx();
        let i32_ty: Ptr<TypeObj> = IntegerType::get(&mut ctx, 32, Signedness::Signless).into();
        let i64_ty: Ptr<TypeObj> = IntegerType::get(&mut ctx, 64, Signedness::Signless).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into(), i64_ty]);
        let ptr_val = block.deref(&ctx).get_argument(0);
        let off_val = block.deref(&ctx).get_argument(1);

        let off_op = Operation::new(
            &mut ctx,
            mir::MirPtrOffsetOp::get_concrete_op_info(),
            vec![mir_ptr_ty.into()],
            vec![ptr_val, off_val],
            vec![],
            0,
        );
        off_op.insert_at_back(block, &ctx);
        append_return(&mut ctx, block);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let gep = find_first::<llvm::GetElementPtrOp>(&ctx, &body).expect("expected one llvm.gep");
        // Element type must come from the MirPtrType pointee (i32), not the
        // i8 fallback used when no operand-type info is available.
        let elem_ty = gep.src_elem_type(&ctx);
        let elem_ty_ref = elem_ty.deref(&ctx);
        let int_ty = elem_ty_ref
            .downcast_ref::<IntegerType>()
            .expect("gep src_elem_type should be IntegerType");
        assert_eq!(int_ty.width(), 32, "gep elem type must be i32 (pointee)");
    }

    /// Build a `mir.shared_alloc` returning `MirPtrType<i32, addrspace=3>` of
    /// length `size`, with the given alloc_key, and append it to `block`.
    fn append_shared_alloc(ctx: &mut Context, block: Ptr<BasicBlock>, alloc_key: &str, size: u64) {
        use pliron::builtin::attributes::IntegerAttr;
        use pliron::utils::apint::APInt;

        let i32_ty: Ptr<TypeObj> = IntegerType::get(ctx, 32, Signedness::Signless).into();
        let result_ty = MirPtrType::get_shared(ctx, i32_ty, true);
        let op = Operation::new(
            ctx,
            mir::MirSharedAllocOp::get_concrete_op_info(),
            vec![result_ty.into()],
            vec![],
            vec![],
            0,
        );
        let alloc = mir::MirSharedAllocOp::new(op);
        alloc.set_attr_elem_type(ctx, TypeAttr::new(i32_ty));
        let size_attr = IntegerAttr::new(
            IntegerType::get(ctx, 64, Signedness::Signless),
            APInt::from_u64(size, std::num::NonZeroUsize::new(64).unwrap()),
        );
        alloc.set_attr_size(ctx, size_attr);
        alloc.set_attr_alloc_key(ctx, StringAttr::new(alloc_key.to_string()));
        op.insert_at_back(block, ctx);
    }

    #[test]
    fn convert_shared_alloc_creates_global_in_addrspace_3() {
        let mut ctx = make_ctx();
        let (module_ptr, block) = build_kernel(&mut ctx, vec![]);
        append_shared_alloc(&mut ctx, block, "k1", 64);
        append_return(&mut ctx, block);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        // Global lives at module level; addressof lives in the function body.
        let top = module_top_block(&ctx, module_ptr);
        let global = top
            .deref(&ctx)
            .iter(&ctx)
            .find_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
            .expect("expected an llvm.global for the shared allocation");
        assert_eq!(
            global.get_address_space(&ctx),
            llvm_addr::SHARED,
            "shared_alloc global must live in addrspace 3"
        );
        assert!(
            global
                .get_symbol_name(&ctx)
                .to_string()
                .starts_with("__shared_mem_"),
            "shared global should have __shared_mem_ prefix"
        );

        let body = kernel_blocks(&ctx, module_ptr);
        let addrof =
            find_first::<llvm::AddressOfOp>(&ctx, &body).expect("expected an llvm.addressof");
        assert_eq!(
            ptr_addrspace(
                &ctx,
                addrof
                    .get_operation()
                    .deref(&ctx)
                    .get_result(0)
                    .get_type(&ctx)
            ),
            llvm_addr::SHARED,
        );
    }

    #[test]
    fn convert_shared_alloc_deduplicates_by_alloc_key() {
        let mut ctx = make_ctx();
        let (module_ptr, block) = build_kernel(&mut ctx, vec![]);
        // Two allocations sharing the same alloc_key — they must collapse to
        // a single underlying global (this is what enables a single `static`
        // referenced from multiple sites to land in one shared region).
        append_shared_alloc(&mut ctx, block, "same-key", 64);
        append_shared_alloc(&mut ctx, block, "same-key", 64);
        // A third with a different key must NOT dedupe with them.
        append_shared_alloc(&mut ctx, block, "other-key", 32);
        append_return(&mut ctx, block);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let top = module_top_block(&ctx, module_ptr);
        let shared_globals = top
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
            .filter(|g| g.get_address_space(&ctx) == llvm_addr::SHARED)
            .count();
        assert_eq!(
            shared_globals, 2,
            "two distinct alloc_keys must produce two globals"
        );

        // Each of the three mir.shared_alloc ops becomes one addressof.
        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(count_ops::<llvm::AddressOfOp>(&ctx, &body), 3);
    }

    fn append_global_alloc(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        global_key: &str,
        constant: bool,
    ) {
        let i32_ty: Ptr<TypeObj> = IntegerType::get(ctx, 32, Signedness::Signless).into();
        let result_ty = if constant {
            MirPtrType::get_constant(ctx, i32_ty, false)
        } else {
            MirPtrType::get_global(ctx, i32_ty, true)
        };
        let op = Operation::new(
            ctx,
            mir::MirGlobalAllocOp::get_concrete_op_info(),
            vec![result_ty.into()],
            vec![],
            vec![],
            0,
        );
        let alloc = mir::MirGlobalAllocOp::new(op);
        alloc.set_attr_global_type(ctx, TypeAttr::new(i32_ty));
        alloc.set_attr_global_key(ctx, StringAttr::new(global_key.to_string()));
        op.insert_at_back(block, ctx);
    }

    #[test]
    fn convert_global_alloc_places_in_global_or_constant_addrspace() {
        let mut ctx = make_ctx();
        let (module_ptr, block) = build_kernel(&mut ctx, vec![]);
        append_global_alloc(&mut ctx, block, "ordinary_static", false);
        append_global_alloc(&mut ctx, block, "_ZN7my_mod3KEYE", true);
        append_return(&mut ctx, block);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let top = module_top_block(&ctx, module_ptr);
        let globals: Vec<_> = top
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
            .collect();
        let global_addr_global = globals
            .iter()
            .find(|g| g.get_address_space(&ctx) == llvm_addr::GLOBAL)
            .expect("expected one global in addrspace(1)");
        let global_addr_const = globals
            .iter()
            .find(|g| g.get_address_space(&ctx) == llvm_addr::CONSTANT)
            .expect("expected one global in addrspace(4)");

        // Constant-memory globals reuse the Rust mangled name so host code can
        // resolve them by name via `cuModuleGetGlobal`; ordinary globals get
        // a counter-suffixed `__device_global_N`.
        assert_eq!(
            global_addr_const.get_symbol_name(&ctx).to_string(),
            "_ZN7my_mod3KEYE",
            "constant globals must keep the mangled global_key as symbol name"
        );
        assert!(
            global_addr_global
                .get_symbol_name(&ctx)
                .to_string()
                .starts_with("__device_global_"),
            "ordinary device globals get the __device_global_ prefix"
        );
    }
}
