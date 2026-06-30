# Resource lifecycle review

**Open when the diff has**: `impl Drop`, `OnceCell`/`OnceLock`/`Lazy`, `lazy_static!`, connection pools, file/socket handles, or any "set up at start, clean up at end" pattern.

**Companion skill**: `/m12-lifecycle`.

## The design question

**Whose lifetime owns this resource, and when does it end?** Rust's RAII is the default cleanup mechanism — if you find yourself writing manual cleanup paths or worrying about "did I close it?", the type system can usually do it for you via `Drop`.

## Red flags

### 1. Manual cleanup paths instead of `Drop`
```rust
// Smell
fn process() -> Result<()> {
    let conn = open_conn()?;
    let result = do_work(&conn);
    conn.close()?;          // forgot on the early-return path below
    if let Err(e) = result { /* … */ }
    result
}

// Better
struct Conn { /* … */ }
impl Drop for Conn { fn drop(&mut self) { /* close */ } }
fn process() -> Result<()> {
    let conn = Conn::open()?;   // closed automatically on any exit
    do_work(&conn)
}
```
Every `close`/`release`/`shutdown` call by hand is a place to forget on the error path.

### 2. `Drop` impl that panics
`Drop` runs during unwinding. Panicking in `Drop` during another panic → process abort. If cleanup can fail, log via `tracing::error!` and swallow; don't `unwrap()` inside `drop()`.

### 3. `Drop` impl that does heavy / blocking work
`drop` runs synchronously, in whatever context the value goes out of scope — including hot paths, including inside `async fn` (where it blocks the executor). Heavy cleanup should be moved to an explicit `async fn close(self)` and the `Drop` impl should just emit a `tracing::warn!` if the explicit close was skipped.

### 4. `Drop` impl that needs `async`
There's no `AsyncDrop` in stable Rust. If the resource genuinely needs async cleanup, the common pattern is "explicit async `close(self)` + best-effort `Drop` + warn if dropped without close." Flag if the code pretends a sync `Drop` is doing async work.

### 5. `lazy_static!` in new code
`lazy_static!` is the old crate. Modern Rust: `std::sync::OnceLock` (thread-safe, no init macro magic) or `once_cell::sync::Lazy`. Suggest the rename in new code; don't churn working `lazy_static!` for taste.

### 6. `OnceCell::new()` then `set()` later vs `Lazy::new(|| init())`
If init is deterministic and infallible, `Lazy::new(|| …)` is cleaner. `OnceCell` + manual `set` makes sense when init needs runtime data not available at module load.

### 7. Connection pool without an idle/cap policy
A pool that grows unboundedly is a leak. Flag pools without `max_size` / `min_idle` / `max_age` config or sensible defaults.

### 8. Scope guards reinvented
If the code does "do A, do thing, undo A" by hand, suggest a guard type. The `scopeguard` crate or a small custom struct with `Drop` is cleaner and survives `?` returns.

### 9. `tokio::spawn`'d task with no `JoinHandle` tracking
Tied to the lifecycle question: who owns this task? When the parent goes away, does the task? Use `JoinSet` or `tokio_util::task::TaskTracker` for graceful shutdown. See [`async-and-concurrency.md`](async-and-concurrency.md).

### 10. Resource handle leaked across an error path
`let f = File::open(p)?; let _ = parse(&f); f.close()?;` — what if `parse` panics? Rust's RAII saves you here (`File` closes on drop), but custom types with manual close don't. Audit each error path.

## Acceptable cases

- `Drop` impl that flushes a buffer and logs on error — that's idiomatic.
- `OnceLock` for a global config / regex / lookup table set once at startup.
- Manual `close(self)` taking ownership when the close itself can fail and the caller needs the error.

## Comment templates

- "`conn.close()` on the happy path only — what about the `?` above it? Make `Conn` `Drop`."
- "`Drop` calls `.unwrap()` — if this fires during unwinding, the process aborts. Log and swallow."
- "Heavy work in `Drop` blocks the executor when this is dropped from async. Move to explicit `close()`."
- "`lazy_static!` — for new code use `OnceLock` or `Lazy::new`."
- "Pool has no max size — what happens at 10k connections?"

## Trace up

Lifecycle problems usually surface as "leaks in production" or "tests are flaky on shutdown." That's a sign no single owner has been assigned to the resource. Pair this with `/m09-domain` for the ownership-design angle.
