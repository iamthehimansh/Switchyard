# Generics & dispatch review

**Open when the diff has**: `impl Trait`, `dyn Trait`, `Box<dyn …>`, generic functions, `where` clauses, trait bounds, or trait object collections.

**Companion skill**: `/m04-zero-cost`.

## The design question

**Compile-time or runtime polymorphism?** Generics → monomorphization → one instantiation per concrete type → fast at runtime, slow to compile, bigger binary. Trait objects → vtable lookup → small binary, slower hot path, supports heterogeneous collections. Pick on need, not on reflex.

| Need | Use |
|---|---|
| One concrete type at each call site, hot path | Generics (`fn foo<T: Trait>`) |
| Heterogeneous collection (`Vec<Box<dyn Trait>>`) | Trait object |
| Public API that wants to hide the type | `impl Trait` return |
| Trait that must be object-safe | Avoid `Self`-receiver methods, generic methods |

## Red flags

### 1. `Box<dyn Trait>` for a single concrete type
```rust
// Smell — only one implementor in the whole project
fn build() -> Box<dyn Backend> { Box::new(OpenAiBackend::new()) }
```
If there's only one implementor and no testing/mocking reason, return the concrete type or `impl Backend`.

### 2. Generics with complex `where` clauses where `dyn` would suffice
```rust
// Smell — every call site monomorphizes 4 KB of code for no perf win
fn log_each<T: Display + Send + Sync + 'static>(items: Vec<T>) {
    for x in items { tracing::info!(%x); }
}
```
If the function isn't hot and isn't pinned by type, `&[&dyn Display]` is fine.

### 3. Object-unsafe traits forced into `dyn`
A trait with a generic method or `Self` by value isn't object-safe and `dyn Trait` won't compile. If the author started adding `where Self: Sized` workarounds, it's a signal the trait design is wrong — split the trait or stop using `dyn`.

### 4. Trait bound proliferation
```rust
fn f<T: Clone + Send + Sync + 'static + Debug + Default + Eq>(t: T) -> T { … }
```
Each bound is a constraint on callers. Drop the ones the body doesn't actually need.

### 5. `impl Trait` in a position that hides too much
Returning `impl Iterator<Item = u64>` is fine for a private helper. Returning `impl Trait` from a public API where callers might want to *name* the type (store it in a struct, return it from another fn) makes them unable to. Be deliberate.

### 6. `Vec<Box<dyn Trait>>` where an enum would do
If you have a fixed, known-at-compile-time set of variants, an enum is simpler, faster, and easier to exhaustively match. `Vec<Box<dyn Trait>>` is for the open set case.

## Acceptable cases

- `Box<dyn Error + Send + Sync>` — standard error trait object.
- `Arc<dyn Backend>` for runtime-chosen backends — that's exactly what `dyn` is for.
- Generic functions with one or two bounds where the body genuinely uses each.
- `impl Trait` in argument position (`fn f(items: impl IntoIterator<Item = u64>)`) — usually a readability win.

## Comment templates

- "Only one implementor — `Box<dyn Backend>` is buying us nothing. Return `OpenAiBackend` or `impl Backend`."
- "These bounds are a wishlist. The body doesn't use `Default` or `Eq` — drop them."
- "Returning `impl Iterator` here means downstream can't store it. Was that intentional?"
- "Closed set of variants — this should be an enum, not `Vec<Box<dyn …>>`."

## Trace up

Generics-vs-trait-objects is a Layer-1 decision in service of a Layer-2 question: "what is the abstraction here, and at what boundary does it close?" If the trait keeps gaining methods or callers keep adding bounds, the abstraction is leaky — that's a design problem (`/m09-domain`), not a dispatch problem.
