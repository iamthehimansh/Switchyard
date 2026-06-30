# Async & concurrency review

**Open when the diff has**: `async fn`, `.await`, `tokio::`, `spawn`, channels (`mpsc`, `oneshot`, `broadcast`, `watch`), `Mutex`/`RwLock`, `Send`/`Sync` bounds, or shared state across tasks.

**Companion skill**: `/m07-concurrency`.

## The design question

**CPU-bound or I/O-bound, and what's the sharing model?** Async (`tokio`) is for I/O concurrency. Threads (`std::thread`, `rayon`) are for CPU parallelism. Mixing them up — running `rayon` work on a `tokio` runtime, calling blocking I/O from an async fn — is the source of most "the server hangs" bugs.

## Red flags

### 1. Lock held across `.await`
```rust
// Smell
let guard = state.lock().await;
let resp = http.get(url).send().await?;  // guard held — blocks all other tasks
guard.update(resp);
```
The whole point of async is concurrent I/O. A lock across `.await` serializes everyone. Fix by dropping the guard before awaiting:
```rust
let snapshot = {
    let guard = state.lock().await;
    guard.snapshot()
};
let resp = http.get(url).send().await?;
state.lock().await.update(resp);
```
For `std::sync::Mutex` / `parking_lot::Mutex`, the guard is `!Send` and the code won't even compile in most cases — but `tokio::sync::Mutex` lets it compile and bite you at runtime.

### 2. Blocking work on the executor thread
- `std::thread::sleep` inside an `async fn` — should be `tokio::time::sleep`.
- `std::fs::read_to_string` in async code — use `tokio::fs` or `tokio::task::spawn_blocking`.
- CPU-heavy work without `spawn_blocking` — starves the runtime.

Convention from the main SKILL.md: write `tokio::time::sleep` fully qualified, and stdlib `sleep` with `use std::thread::sleep;` to make the distinction obvious at a glance.

### 3. `tokio::spawn` for the sake of it
Not every task belongs in a separate task. Spawning has overhead (scheduling, allocation, potential for orphaned work) and complicates error propagation. Reflexive spawning is a smell. Ask:
- Does this work need to run concurrently with the caller? If not, just `.await` it.
- Is the spawned task awaited or `join_handle.abort()`-on-drop'd? Orphaned spawns leak.
- Does it propagate errors back? If the spawn returns `Result` and nothing reads the `JoinHandle`, errors are silently dropped.

### 4. Unbounded channels
```rust
// Smell
let (tx, rx) = mpsc::unbounded_channel();
```
Unbounded channels can OOM the process. Producers don't block, so the queue grows until the box dies. **Bounded channels are defense-in-depth** — `mpsc::channel(N)` with a reasonable `N` for your domain. If the bound is hit, you learn something is wrong. Unbounded only with a justification (explicit comment) that the producer side is provably finite.

### 5. `Arc<Mutex<…>>` as a substitute for message passing
The "share state with a mutex" pattern often loses to "give one task the state and send it messages." When the lock-contended state is on the hot path, an actor-style design (`mpsc::Sender<Command>`) is faster and easier to reason about.

### 6. `tokio::select!` arms that aren't cancel-safe
`select!` cancels the losing futures. If one arm has side effects (e.g. partially read from a stream), it may corrupt state on cancellation. Use cancellation-safe APIs (`tokio::io::AsyncReadExt::read_buf` is, `read_exact` is not — read the tokio docs for each). Comment when non-trivial.

### 7. `Send` + `Sync` bound proliferation
If every function signature has `T: Send + Sync + 'static`, you've effectively made the whole codebase global. Sometimes the right answer is to keep the data on one task and avoid the share entirely.

### 8. Spawned tasks with no shutdown story
Long-running tasks (poll loops, watchdogs) spawned via `tokio::spawn` need a cancellation token, a shutdown channel, or to be tied to a `JoinSet` that the owner drops. Otherwise they outlive the system that created them.

### 9. `block_on` inside an async context
Calling `Runtime::block_on` from inside a `tokio::spawn`'d task or any async fn deadlocks or panics. Always a bug.

### 10. `parking_lot::RwLock` vs `tokio::sync::RwLock`
Default to `parking_lot::RwLock` for short critical sections (no `.await` inside) — it's faster and fairer. Reach for `tokio::sync::RwLock` only when you genuinely need to hold the lock across `.await`.

## Acceptable cases

- A `tokio::sync::Mutex` held across `.await` on a strict serialization boundary (e.g. one HTTP client per upstream) where serializing is the point.
- `Arc<Mutex<HashMap<…>>>` for a small read/write cache where the contention is bounded and measured.
- `tokio::spawn` for long-running independent work *with* a documented shutdown path.

## Comment templates

- "`MutexGuard` held across `.await` on line N. Drop it in a scope first."
- "`std::thread::sleep` inside an async fn — `tokio::time::sleep`."
- "Unbounded channel — what bounds the producer? If nothing, this is an OOM waiting to happen."
- "Spawning here but nothing awaits the `JoinHandle` — errors will be lost."
- "`tokio::sync::RwLock` for a short read with no await — `parking_lot::RwLock` is faster and fairer."
- "`select!` arm is not cancel-safe — `read_exact` will lose bytes on cancellation."

## Trace up

If the concurrency model is unclear ("who owns this state? who runs this code?"), the fix isn't a different lock — it's a clearer task structure. Pair this review with `/m09-domain` for the modeling angle and `/m12-lifecycle` for the cleanup angle.
