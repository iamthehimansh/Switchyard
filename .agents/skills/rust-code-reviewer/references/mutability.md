# Mutability review

**Open when the diff has**: `RefCell`, `Cell`, `Mutex`, `RwLock`, `&mut self`, or interior-mutability patterns.

**Companion skill**: `/m03-mutability`.

## The design question

**Does this really need to mutate, or did the author reach for `mut` because the data model is wrong?** Interior mutability bypasses the borrow checker's compile-time guarantees and replaces them with runtime checks (`RefCell` panics) or runtime cost (`Mutex` contention). Use it when needed, never reflexively.

## Red flags

### 1. `RefCell` because the author "got a borrow error"
The borrow checker complained, and the fastest fix was `RefCell`. Now you have a runtime panic landmine. Read the original code — usually the right fix was restructuring (split the struct, return a value instead of mutating, use `&mut` on a smaller scope).

### 2. `Mutex` to mutate a field that's set once
```rust
// Smell
struct Server { config: Mutex<Config> }
```
If `config` is set at construction and read forever after, it doesn't need a `Mutex`. Pass it as `Arc<Config>` (immutable share) or use `OnceCell<Config>`.

### 3. `&mut self` that doesn't actually mutate
If a method takes `&mut self` but only reads fields, change it to `&self`. The `&mut` forces exclusive access on callers for no reason.

### 4. Lock held across `.await`
This is the headline async-mutability bug. `std::sync::Mutex` held across `.await` is a soundness issue (`MutexGuard: !Send`); `parking_lot::Mutex` is the same; `tokio::sync::Mutex` is the only correct choice for that pattern. Even then, **prefer dropping the guard before awaiting**:
```rust
// Smell
let mut state = mutex.lock().await;
state.update();
some_async_call().await;  // lock held the whole time

// Better
{
    let mut state = mutex.lock().await;
    state.update();
}  // guard dropped
some_async_call().await;
```
See [`async-and-concurrency.md`](async-and-concurrency.md) for the full pattern.

### 5. `RwLock` where reads dominate but writes are unrelated
If readers and writers touch different fields, you're forcing them to serialize for no reason. Split the struct so each lock guards a coherent unit.

### 6. `Cell<T>` for non-`Copy` types
`Cell` is for `Copy` types (`Cell<u64>`). For non-`Copy`, you need `RefCell`. If you see `Cell<String>` it doesn't compile — but if you see contortions to make `Cell` work, the design is fighting the tool.

## Acceptable cases

- `RefCell` inside a `pub`-API type where the interior mutation is genuinely an implementation detail (e.g. lazy caching).
- `Mutex` on a field that's read and written from multiple tasks for legitimate shared state.
- `Atomic*` for counters / flags — these are zero-overhead and `Send + Sync` for free.

## Comment templates

- "`&mut self` here but I only see reads — `&self` works."
- "`RefCell` looks reactive to a borrow error. What was the original code trying to do? There's probably a structural fix."
- "Lock held across `.await` on line N — drop the guard in a scope first."
- "`Mutex<Config>` but it's set once. `Arc<Config>` or `OnceCell` instead."

## Trace up

Pervasive interior mutability usually points at a "god struct" that's trying to be everything. The fix is decomposition — split the struct so each piece has clear single-purpose ownership. See `/m09-domain`.
