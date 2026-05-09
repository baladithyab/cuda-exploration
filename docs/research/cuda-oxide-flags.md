# cuda-oxide flags / fast-math investigation

Source under investigation:
`/home/codeseys/.cargo/git/checkouts/cuda-oxide-6d394bb007f5e114/6de0509/` (branch `master`, commit checked out by cargo).

## TL;DR

**No.** cuda-oxide (at this revision) has **zero** user-facing knobs for fast-math, FMA contraction, `-ffast-math`, `fp-contract`, or a read-only-cache / `__ldg` hint. The plumbing for LLVM fast-math flags *exists* end-to-end (dialect-llvm defines `FastmathFlags { NNAN, NINF, NSZ, ARCP, CONTRACT, AFN, REASSOC, FAST }`, MIR-lower calls `add_fastmath_flags` on every `fadd` / `fsub` / `fmul` / `fdiv` / `frem` / `fneg`), but every callsite passes `FastmathFlagsAttr::default()` which is `FastmathFlags::empty()` — i.e. the attribute is attached with **no bits set**. There is no CLI flag, no `#[kernel(...)]` parameter, no env var, and no RUSTFLAGS path that changes this. `core::intrinsics::fmuladdf32` is lowered to a **libdevice call `__nv_fmaf`** rather than `llvm.fmuladd`, so the one route you'd expect to give you a hardware FMA from user code goes through libdevice instead of letting ptxas contract it. This is almost certainly why the generated PTX has `0 fma.rn.f32` while `nvcc -O3` emits 5.

## CLI surface

Full enumeration of `crates/cargo-oxide/src/main.rs` (lines 48–138). Subcommands and their flags:

- `Run <example?>`: `--dlto`, `--emit-nvvm-ir`, `--arch <sm_XX>`, `--features <csv>`, `--bin <name>`, `-v/--verbose`
- `Build <example?>`: `--dlto`, `--emit-nvvm-ir`, `--arch`, `--features`, `-v/--verbose`
- `Pipeline <example>`: `--dlto`, `--emit-nvvm-ir`, `--arch`
- `Debug <example>`: `--cgdb`, `--tui`
- `Fmt`: `--check`
- `New <name>`: `--async`
- `Doctor`, `Setup`: no flags

`rg -n "fast[-_]?math|ffast|no-bounds|fp-contract|-O3|RUSTFLAGS" crates/cargo-oxide/src/` → **zero matches** for any FP or fast-math flag in the CLI module. The only RUSTFLAGS are hard-coded in `commands.rs:803`:

```
803|fn build_rustflags(backend_so: &Path, debug: bool) -> String {
804|    let mut flags = format!(
805|        "-Z codegen-backend={} -C opt-level=3 -C debug-assertions=off \
             -Z mir-enable-passes=-JumpThreading -Csymbol-mangling-version=v0",
806|        backend_so.display()
807|    );
808|    if debug { flags.push_str(" -C debuginfo=2"); }
811|    flags
812|}
```

Relevant env vars read anywhere in the tree (`rg "CUDA_OXIDE_"`): `CUDA_OXIDE_BACKEND`, `CUDA_OXIDE_VERBOSE`, `CUDA_OXIDE_SHOW_RUSTC_MIR`, `CUDA_OXIDE_DUMP_MIR`, `CUDA_OXIDE_DUMP_LLVM`, `CUDA_OXIDE_EMIT_LTOIR`, `CUDA_OXIDE_EMIT_NVVM_IR`, `CUDA_OXIDE_ARCH`, `CUDA_OXIDE_TARGET`. **None** are fast-math related.

## `#[kernel]` attribute parameters

From `crates/cuda-macros/src/lib.rs:95-115`:

```
 96|/// Attribute arguments for #[kernel(...)]
 97|/// Supports: #[kernel] or #[kernel(Type1, Type2, Type3)]
 98|struct KernelArgs {
 99|    /// Types to instantiate generic kernels for
100|    instantiate_types: Vec<Type>,
101|}
103|impl Parse for KernelArgs {
104|    fn parse(input: ParseStream) -> syn::Result<Self> {
105|        if input.is_empty() { return Ok(KernelArgs { instantiate_types: vec![] }); }
111|        let types: Punctuated<Type, Token![,]> = Punctuated::parse_terminated(input)?;
```

The **only** thing `#[kernel(...)]` accepts is a comma-separated list of types to monomorphize a generic kernel against. There is no `fast_math`, no `restrict`, no `contract`, no `opt`. Sibling proc-macros in the same crate: `#[launch_bounds]`, `#[cluster_launch]`, `#[device]`, `#[convergent]`, `#[pure]`, `#[readonly]`. The last three apply to *device-extern FFI declarations* and map to LLVM `convergent` / `readnone` / `readonly` function attributes (collector.rs:708, device_codegen.rs:302), **not** to load instructions.

## LLVM IR lowering: are `fast` flags set?

No. Every floating-point op in `crates/mir-lower/src/convert/ops/arithmetic.rs` gets the *empty* fast-math flags attribute:

```
 97|/// Add fastmath flags attribute to a floating-point operation.
 98|fn add_fastmath_flags(ctx: &mut Context, op: Ptr<Operation>) {
 99|    let flags = FastmathFlagsAttr::default();
100|    let key: pliron::identifier::Identifier = "llvm_fast_math_flags".try_into().unwrap();
101|    op.deref_mut(ctx).attributes.0.insert(key, flags.into());
102|}
```

And the default is empty:

```
// crates/dialect-llvm/src/attributes.rs
117|#[pliron_attr(name = "llvm.fast_math_flags", verifier = "succ")]
119|pub struct FastmathFlagsAttr(pub FastmathFlags);
121|impl Default for FastmathFlagsAttr {
122|    fn default() -> Self {
123|        FastmathFlagsAttr(FastmathFlags::empty())
124|    }
125|}
```

The bit set *is* defined (`attributes.rs:89-100`):

```
 89|bitflags::bitflags! {
 90|    pub struct FastmathFlags: u32 {
 91|        const NNAN     = 1 << 1;
 92|        const NINF     = 1 << 2;
 93|        const NSZ      = 1 << 3;
 94|        const ARCP     = 1 << 4;
 95|        const CONTRACT = 1 << 5;
 96|        const AFN      = 1 << 6;
 97|        const REASSOC  = 1 << 7;
 98|        const FAST     = NNAN|NINF|NSZ|ARCP|CONTRACT|AFN|REASSOC;
 99|    }
100|}
```

Every fadd / fsub / fmul / fdiv / frem / fneg callsite in mir-lower calls `add_fastmath_flags` with this empty default (arithmetic.rs:122, 148, 174, 200, 227, 542; cast.rs:453). Grep confirmation:

```
$ rg -n "FastmathFlagsAttr::" crates/
crates/mir-lower/src/convert/ops/arithmetic.rs:99:  let flags = FastmathFlagsAttr::default();
crates/mir-lower/src/convert/ops/arithmetic.rs:542: llvm::FNegOp::new_with_fast_math_flags(ctx, operand, FastmathFlagsAttr::default())
crates/mir-lower/src/convert/ops/cast.rs:453:       let flags = dialect_llvm::attributes::FastmathFlagsAttr::default();
crates/dialect-llvm/src/ops/comparison.rs:209:      op.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
```

No constructor in the entire tree produces a non-default (i.e. non-empty) `FastmathFlagsAttr`. **`contract` is never emitted**, so ptxas / NVVM sees `fmul` + `fadd` without permission to contract and won't form `fma.rn.f32`.

### What about explicit `fmul_add` / `f32::mul_add` / `core::intrinsics::fmuladdf32`?

These do *not* lower to `llvm.fmuladd.f32` (which NVPTX can lower to hardware FMA). Instead they lower to a **libdevice call**:

```
// crates/mir-lower/src/convert/ops/call.rs
175|    FmuladdF32,
176|    FmuladdF64,
...
269|    Self::FmaF32 | Self::FmuladdF32 => Ok("__nv_fmaf"),
270|    Self::FmaF64 | Self::FmuladdF64 => Ok("__nv_fma"),
```

So even the explicit escape hatch produces a libdevice function call rather than a contractable IR construct.

## Read-only cache / `__ldg` equivalent

No `__ldg`, no `nvvm.ldg`, no `!invariant.load` metadata, no texture-fetch path.

```
$ rg -n -i "ldg|__ldg|nvvm\.ldg|invariant\.load|tex1Dfetch" crates/
(no matches)
```

`#[readonly]` *exists* but is unrelated — it's a marker on `extern` device-function declarations that sets LLVM's function-level `readonly` attribute for NVVM attribute-propagation (`crates/cuda-macros/src/lib.rs:1249`, `crates/rustc-codegen-cuda/src/collector.rs:733`, `crates/dialect-llvm/src/export.rs:94-95`):

```
1249|pub fn readonly(_attr: TokenStream, item: TokenStream) -> TokenStream {
1250|    // Marker attribute - collector will read and apply LLVM readonly attribute
```

`cuda_host::ReadOnly<'a, T>` (in `crates/cuda-host/src/launch.rs:120`) is a host-side *argument wrapper* for launch bindings; it does not influence IR generation.

## Book + README

```
$ rg -ni "fast.?math|fma\b|ffast|fp.?contract|contraction|-O3" cuda-oxide-book/ README.md CONTRIBUTING.md
(no matches)
```

The only `fma` hit in the whole checkout is the `FmaF32/FmaF64/FmuladdF32/FmuladdF64` enum in `mir-lower` and `mir-importer`, plus a passing `fast_math_helper` example name in a `#[device]` docstring (`cuda-macros/src/lib.rs:929`) — **not** a feature, just an example of an extern FFI decl.

## What we'd need to add upstream

Concrete minimal change set (listed roughly in effort order):

1. **Read a context flag in `add_fastmath_flags`**. In `crates/mir-lower/src/convert/ops/arithmetic.rs:98`, replace `FastmathFlagsAttr::default()` with a value looked up from a lowering config (thread a `LoweringOptions { fast_math: FastmathFlags }` through `Context`, or read it from an env var / attribute on the parent function). Same call in `cast.rs:453` and the fneg site at `arithmetic.rs:542`.
2. **Expose a CLI flag**. Add `--fast-math` / `--fp-contract=on|fast|off` to `Run`/`Build`/`Pipeline` in `crates/cargo-oxide/src/main.rs` and plumb it to a new env var (e.g. `CUDA_OXIDE_FAST_MATH`) forwarded in `commands.rs` alongside the existing `CUDA_OXIDE_*` vars. A minimum-viable version could set only `CONTRACT` (= `-ffp-contract=fast`) without the full `FAST` bundle, which keeps IEEE semantics for everything except FMA fusion.
3. **Per-kernel opt-in**. Extend `KernelArgs` in `crates/cuda-macros/src/lib.rs:98` with an optional `fast_math` (or `contract`) keyword, e.g. `#[kernel(fast_math)]` or `#[kernel(contract)]`. The collector already parses per-function attributes (see `crates/rustc-codegen-cuda/src/collector.rs:733`), so propagating a per-kernel FMF through the device-codegen context is feasible.
4. **Lower `fmuladdf32` to `llvm.fmuladd`** (as an alternative or complement to `__nv_fmaf`). Change `call.rs:269-270` so `FmuladdF32/F64` become the `llvm.fmuladd.f32/f64` intrinsic rather than a libdevice call. NVPTX already lowers `llvm.fmuladd` to `fma.rn.*` regardless of `-ffp-contract`, so this alone would let users opt in via `a.mul_add(b, c)` without any global flag.
5. **Optional: `__ldg` hint**. Introduce either a `ReadOnlyPtr<T>` wrapper in `cuda-device` that emits an `llvm.nvvm.ldg.global.*` call, or a `#[ldg]` load intrinsic. Currently none of this exists.

A PR that does (1) + (4) is probably the smallest patch that closes the measured FMA gap vs `nvcc -O3`.

## Search commands run

```bash
rg -n --no-heading -i "fast[-_]?math|ffast-math|unsafe-fp|fp-contract|fmuladd|contract|reassoc|nnan|ninf|arcp|fast_flags|set_fast|LLVMSetFastMath|FMF" crates/
rg -n --no-heading "clap|Parser|Subcommand|#\[arg|long\s*=" crates/cargo-oxide/src/
rg -n --no-heading -i "fast[-_]?math|fp[-_]?contract|optlevel|OptLevel|opt_level" crates/rustc-codegen-cuda/ crates/cargo-oxide/ crates/cuda-macros/
rg -n --no-heading "kernel\(|#\[kernel|parse_kernel|KernelAttr|kernel_attrs|restrict|no_bounds" crates/cuda-macros/src/lib.rs
rg -n --no-heading -i "ldg|__ldg|readonly|read_only|nv_weak|tex1Dfetch|nvvm\.ldg|invariant\.load" crates/
rg -n --no-heading -i "fast[-_ ]?math|fma\b|ffast|fp[-_ ]?contract|contraction|opt[- ]?level|-O3|fmuladd" cuda-oxide-book/ README.md CONTRIBUTING.md
rg -n --no-heading "FastmathFlagsAttr::" crates/
rg -n --no-heading "CUDA_OXIDE_" crates/cargo-oxide/src/ crates/rustc-codegen-cuda/src/ crates/mir-lower/src/
rg -n --no-heading "fmuladd|FmuladdF|fmaf|__nv_fma" crates/mir-lower/src/convert/ops/call.rs
```

All commands executed from
`/home/codeseys/.cargo/git/checkouts/cuda-oxide-6d394bb007f5e114/6de0509/`.
