# Anti-pattern catalog

**Open when**: you sense something is off and want a quick smell-to-refactor lookup.

**Companion skill**: `/m15-anti-pattern`.

This is the dictionary view. Each row is a smell → likely cause → suggested fix → deeper reference.

## The top offenders

### 1. `.clone()` everywhere
- **Cause**: fighting the borrow checker, ownership model never decided.
- **Fix**: pass references, redesign the function signature, use `Arc` for shared ownership, use `Cow` for sometimes-borrow-sometimes-own.
- **See**: [`ownership-and-borrowing.md`](ownership-and-borrowing.md).

### 2. `.unwrap()` / `.expect()` in production
- **Cause**: author punted on the failure case.
- **Fix**: propagate with `?`, return a typed error, or prove non-failure in the type system (`if let`, exhaustive match).
- **See**: [`error-handling.md`](error-handling.md).

### 3. `Arc<Mutex<…>>` reflex
- **Cause**: "we might share this" defensive wrapping at construction time.
- **Fix**: let callers wrap when they actually share. Or redesign to message passing.
- **See**: [`smart-pointers.md`](smart-pointers.md), [`async-and-concurrency.md`](async-and-concurrency.md).

### 4. Lock held across `.await`
- **Cause**: forgot the executor is single-threaded per task.
- **Fix**: drop the guard in a scope before the await.
- **See**: [`async-and-concurrency.md`](async-and-concurrency.md).

### 5. Stringly-typed everything
- **Cause**: `String` is easy, types are work.
- **Fix**: newtype with a `parse` boundary so the rest of the code can't pass garbage.
- **See**: [`type-driven-design.md`](type-driven-design.md).

### 6. Booleans where an enum belongs
- **Cause**: started with two states, never refactored when a third appeared.
- **Fix**: enum.
- **See**: [`type-driven-design.md`](type-driven-design.md).

### 7. Unbounded channel
- **Cause**: easier than picking a number.
- **Fix**: pick a number. The number is wrong, you'll learn the right one. Unbounded is OOM-shaped.
- **See**: [`async-and-concurrency.md`](async-and-concurrency.md).

### 8. Reflexive `tokio::spawn`
- **Cause**: "async means spawn."
- **Fix**: `.await` inline unless you need concurrency. Track join handles if you do spawn.
- **See**: [`async-and-concurrency.md`](async-and-concurrency.md).

### 9. Returning `Box<dyn Trait>` for a single implementor
- **Cause**: future-proofing for an unrealized abstraction.
- **Fix**: return the concrete type or `impl Trait` until a second implementor appears.
- **See**: [`generics-and-dispatch.md`](generics-and-dispatch.md).

### 10. Hand-rolled cleanup
- **Cause**: didn't know `Drop` existed or didn't want to bother.
- **Fix**: `impl Drop`. RAII saves the error paths you forgot about.
- **See**: [`resource-lifecycle.md`](resource-lifecycle.md).

### 11. Retry on every error
- **Cause**: didn't distinguish transient from permanent.
- **Fix**: split the error type so retry can match on category.
- **See**: [`domain-error-resilience.md`](domain-error-resilience.md).

### 12. `lazy_static!` in new code
- **Cause**: copy-pasted from old code.
- **Fix**: `OnceLock` (std) or `Lazy::new` (once_cell).
- **See**: [`resource-lifecycle.md`](resource-lifecycle.md).

### 13. `LinkedList`
- **Cause**: muscle memory from another language.
- **Fix**: `Vec` or `VecDeque`.
- **See**: [`performance.md`](performance.md).

### 14. `&mut self` that only reads
- **Cause**: copy-pasted method signature.
- **Fix**: `&self`. Stops forcing exclusive borrows on callers.
- **See**: [`mutability.md`](mutability.md).

### 15. Re-implementing standard combinators
- **Cause**: didn't know the method existed.
- **Fix**: `map`, `and_then`, `unwrap_or_else`, `ok_or_else`, `if let Some`, `let-else`. Read `Option`/`Result` docs.

### 16. Long lists of similar tests (especially AI-added)
- **Cause**: AI loves enumeration.
- **Fix**: keep the 3 that cover distinct behaviors. Delete the rest. Coverage is about behaviors, not inputs.

### 17. Comments restating the function name
- **Cause**: AI tax or formality-by-default.
- **Fix**: delete. If the name doesn't say it, fix the name, don't paper over with a comment.
- **See**: [`naming-and-comments.md`](naming-and-comments.md).

### 18. "Just use `String` for now"
- **Cause**: deferred modeling.
- **Fix**: model now or write down explicitly that the type is intentionally loose and why.

### 19. `match e.to_string().contains("…")`
- **Cause**: error type is opaque, downstream needs to decide.
- **Fix**: expose the variant. String-matching errors is fragile.
- **See**: [`error-handling.md`](error-handling.md), [`domain-error-resilience.md`](domain-error-resilience.md).

### 20. Big god struct with `Arc<Mutex<…>>` on every field
- **Cause**: no decomposition.
- **Fix**: split into smaller types with clear ownership. The locks usually disappear with the split.
- **See**: [`smart-pointers.md`](smart-pointers.md), [`mutability.md`](mutability.md).

## How to use this list

Treat it as a fast scan during your second pass. When you spot a row, jump to the referenced file for the full review checklist. Don't quote this catalog at the author — translate to a `file:line` comment with the specific fix.
