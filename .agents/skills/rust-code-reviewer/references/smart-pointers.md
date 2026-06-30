# Smart pointers review

**Open when the diff has**: `Box<…>`, `Rc<…>`, `Arc<…>`, `Weak<…>`, `RefCell<…>`, or nested forms like `Arc<Mutex<…>>` / `Rc<RefCell<…>>`.

**Companion skill**: `/m02-resource`.

## The design question

**What ownership pattern does this actually need?** A wrapper is a tax — atomic ops (`Arc`), heap allocation (`Box`), runtime borrow checks (`RefCell`). Pay the tax only when the design demands it.

| Need | Use |
|---|---|
| Single owner, value too large for stack or trait object | `Box<T>` |
| Shared ownership, single thread | `Rc<T>` |
| Shared ownership, multiple threads | `Arc<T>` |
| Break a cycle | `Weak<T>` |
| Interior mutability, single thread | `RefCell<T>` |
| Interior mutability, multiple threads, sync code | `Mutex<T>` / `RwLock<T>` |
| Interior mutability, async, held across `.await` | `tokio::sync::Mutex<T>` |

## Red flags

### 1. `Arc` reflexively on construction
```rust
// Smell — Switchyard's pre-wrap-in-constructor anti-pattern
fn new(config: Config) -> Self {
    Self { config: Arc::new(Mutex::new(config)) }
}
```
Owners decide their own synchronization. Don't `Arc<Mutex<…>>` in a constructor unless every caller will need it. Let them wrap when they share.

### 2. `Arc<Box<T>>`
Both are pointers. You only need one indirection. If you see this, ask for justification or unwrap one.
Exception: `Arc<dyn Trait>` where the inner already had to be `Box<dyn Trait>` somewhere — but usually `Arc<dyn Trait>` directly is what you want.

### 3. `Arc` around an already-cheap-to-clone handle
```rust
// Smell
let sender = Arc::new(tx);  // tx: tokio::sync::mpsc::Sender — already Arc internally
```
Channels, `Bytes`, `reqwest::Client`, `tokio::sync::mpsc::Sender`, `Arc<…>` itself — all cheap to clone. Wrapping them in another `Arc` adds an indirection for no win.

### 4. `Rc<RefCell<T>>` in code that might go multithreaded
Read the surrounding module. Is it touched from any async context, any background thread? If yes, `Rc<RefCell<…>>` will break the moment someone moves it across a thread. Suggest `Arc<Mutex<…>>` — or, better, redesign so no sharing is needed.

### 5. `Arc<Mutex<HashMap<K, V>>>` as the central state
This pattern works but is rarely the right answer. Ask:
- Is contention actually a problem? `parking_lot::RwLock` or `dashmap` may be better.
- Could the work be modeled as message passing (`mpsc`) so each task owns its own state?
- Is the lock held across `.await`? If yes, [`async-and-concurrency.md`](async-and-concurrency.md) applies.

### 6. `Box<T>` for small types
`Box<u64>` is a heap allocation for 8 bytes. Almost always a mistake. Same for `Box<MyEnum>` where the enum is small. The author may have copied from C++ instinct.

### 7. Missing `Weak` for parent pointers
Parent ↔ child with `Rc<…>` both ways = memory leak. The child's pointer to parent should be `Weak`. Same for `Arc` cycles.

## Acceptable cases

- `Arc<dyn Trait>` for shared trait objects across tasks — idiomatic.
- `Box<dyn Error + Send + Sync>` in error types — standard.
- `Box<T>` when erasing a generic into a `dyn` or when the value is huge.
- `Arc<RwLock<T>>` for read-heavy shared config — but check it's not held across `.await` ([`async-and-concurrency.md`](async-and-concurrency.md)).

## Comment templates

- "Why `Arc<Mutex<…>>` here? Caller decides synchronization. Move this out of the constructor."
- "`Arc<Box<…>>` — drop the `Box`, `Arc<…>` already heap-allocates."
- "`tx` is already a cheap handle; the extra `Arc` is noise."
- "`Rc<RefCell<…>>` in a module that's used from `tokio::spawn` will explode. Use `Arc<Mutex<…>>` or redesign."

## Trace up

If a module is drowning in `Arc<Mutex<…>>`, the symptom is "we share state everywhere." The fix isn't a different wrapper — it's deciding which task *owns* the state and routing access through it (message passing). See `/m09-domain` for that framing, `/m07-concurrency` for the async-specific angle.
