# Complex Multiply/Add (issue #35)

Regression test for [#35](https://github.com/NVlabs/cuda-oxide/issues/35):
device codegen rejected `z = z*z + c` on a complex-number type with

```text
Unsupported construct: Alias type not yet supported: Mul::Output
```

## Root cause

The MIR type translator resolved arithmetic-trait associated outputs
(`<T as Mul>::Output`, `<T as Add>::Output`, ...) to the self type *only when
`T` was a primitive*. For an aggregate such as `num_complex::Complex32` (or any
user struct implementing `Mul`), `<Complex32 as Mul>::Output` is an ADT, so it
fell through to the unsupported-type arm and aborted PTX generation. The second
report on the issue confirms a plain user struct implementing `Mul` hits the
identical error.

## Fix

`crates/mir-importer/src/translator/types.rs` now also resolves the output when
the operand is `RigidTy::Adt(..)`, translating the self type directly (recursion
then lays out the aggregate). This example uses a self-contained `Complex32`
rather than the `num_complex` dependency so it exercises exactly the ADT
`Output = Self` path.

## Running

```bash
cargo oxide run complex_mul
```

Expected output: `complex_square_add: PASS (result = 0.75)`.
