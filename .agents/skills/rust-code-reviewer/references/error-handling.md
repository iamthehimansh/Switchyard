# Error handling review

**Open when the diff has**: `Result`, `Option`, `?`, `unwrap`, `expect`, `panic!`, `anyhow`, `thiserror`, custom error enums, or error conversions.

**Companion skill**: `/m06-error-handling`.

## The design question

**Is this failure expected, absent, a bug, or unrecoverable?** Each maps to a different tool. Mixing them is how you end up with `.unwrap()` on user input and `Result<…, String>` deep in a library.

| Failure kind | Tool |
|---|---|
| Expected (I/O, parse, network) | `Result<T, E>` with typed `E` |
| Absence (lookup, optional) | `Option<T>` |
| Programmer bug (broken invariant) | `panic!` / `assert!` |
| Unrecoverable (OOM, corrupted state) | `panic!` or process exit |

Library vs application:
- **Library code**: typed errors with `thiserror`, so callers can match.
- **Application/binary code**: `anyhow::Error` with `.context()` for the call chain.

## Red flags

### 1. `.unwrap()` / `.expect()` in production code
Switchyard project rule: forbidden, including in Rust tests. Propagate with `?`, return typed errors, or `match` explicitly. If you see `.unwrap()` and the author argues "it can't fail," ask them to encode that in the type system (`if let`, exhaustive `match`, type-state) so the compiler proves it.

Common exceptions worth challenging:
- `.unwrap()` after a `is_some()` check — use `if let Some(x)` or `match` instead.
- `.unwrap()` on regex compilation of a literal at startup — replace with `OnceLock` + return-`Result` constructor, or use `regex_static`.

### 2. `panic!` for recoverable failures
Network timeout, bad JSON, missing config — none of these are programmer bugs. They're expected runtime conditions and belong in `Result`. `panic!` is for "this should never happen and if it did the program is in a state we can't reason about."

### 3. `Result<T, String>` or `Result<T, Box<dyn Error>>` in a library
Strings are unmatchable; `Box<dyn Error>` discards type information. For a library, define a `thiserror` enum so downstream code can match on variants.

### 4. Swallowed errors
```rust
// Smell
let _ = some_fallible_op();
some_op().ok();
match res { Ok(v) => v, Err(_) => default }  // why default? what error?
```
Ignoring errors silently is rarely correct. At minimum log them with `tracing::warn!` or `tracing::error!` with structured fields. Better: handle them, or propagate.

### 5. `?` followed by `.unwrap()` immediately
Inconsistent: you decided one was recoverable, then the next isn't. Re-examine.

### 6. Lossy error conversion
`.map_err(|_| MyError::Generic)?` discards the original. Use `#[from]` (thiserror) or wrap with context: `.map_err(|e| MyError::Backend { source: e })?` or `.with_context(|| "backend call")?`.

### 7. `anyhow::Error` in a public library API
`anyhow` is the application-side ergonomic helper. Exposing it from a library forces all downstream code to depend on `anyhow` and unable to programmatically match. Use `thiserror` for the lib's error type; let the binary wrap in `anyhow` if it wants.

### 8. `.context()` overload
Every `?` having `.context("…")` adds noise. Add context at *boundaries* (function entry/exit, foreign-system calls), not on every internal line.

## Acceptable cases

- `unreachable!()` with a comment explaining the invariant.
- `assert!`/`debug_assert!` for programmer-bug guards at function entry.
- `panic!` in a `From` impl that's only used to convert a `Result` into a known-Ok path — but flag and ask if `expect_err`/`unwrap_or_else` is cleaner.

## Comment templates

- "`.unwrap()` here — what's the invariant that says this can't fail? Encode it in the types or propagate the error."
- "`Result<T, String>` from a public function — callers can't match. Use a `thiserror` enum."
- "Swallowed error on line N. Either handle it or `tracing::warn!(error = %e, …)`."
- "`anyhow::Error` in a public lib API — switch to `thiserror`."

## Trace up

Repeated `.unwrap()`/`.expect()`/`panic!` in a module is rarely "a few bad lines." It's a sign the author didn't decide what failure means in this layer. That's a domain-error question — see [`domain-error-resilience.md`](domain-error-resilience.md) and `/m13-domain-error`.
