# Dispatch: diff signal → which lens

Use this table on the first pass. `grep` the diff (`git diff`) for the listed signals, then read the matching reference. If the signal fires for code you don't fully grasp at the design level, also invoke the companion `/m*-*` skill — it gives the cognitive framing for the same area.

The references are review checklists (concrete smells, before/after). The `m*-*` skills are design-craft pedagogy. Read the reference first; invoke the skill when you need the "why" to write a useful comment.

## Quick-scan greps

Run these against the diff first. Each hit suggests one or more lenses.

| `git diff` pattern | Likely lens(es) |
|---|---|
| `\.clone()` | `ownership-and-borrowing` → `performance` |
| `\.unwrap()`, `\.expect(` | `error-handling` |
| `panic!`, `todo!`, `unimplemented!` | `error-handling` |
| `Arc<`, `Rc<`, `Box<`, `Weak<` | `smart-pointers` → `async-and-concurrency` if `Arc<Mutex<…>>` |
| `RefCell`, `Cell`, `\.borrow_mut\(` | `mutability` |
| `Mutex`, `RwLock` | `async-and-concurrency` → `mutability` |
| `async fn`, `\.await`, `tokio::`, `#\[tokio::` | `async-and-concurrency` |
| `tokio::spawn`, `task::spawn` | `async-and-concurrency` |
| `unbounded_channel`, `mpsc::unbounded`, `crossbeam_channel::unbounded` | `async-and-concurrency` |
| `dyn `, `impl Trait`, `Box<dyn ` | `generics-and-dispatch` |
| `PhantomData`, `marker::` | `type-driven-design` |
| `lazy_static!`, `OnceCell`, `OnceLock`, `Lazy::new` | `resource-lifecycle` |
| `impl Drop for`, `drop(` | `resource-lifecycle` |
| `retry`, `backoff`, `circuit`, `with_retry`, `tokio::time::sleep` in a loop | `domain-error-resilience` |
| `Vec::new()` near `\.push(` in a loop, `String::new()` near `+= ` | `performance` |
| `for .* in 0\.\.` over a collection (vs `.iter()`) | `performance` |
| `pub fn (get_|set_)` | `naming-and-comments` |
| Triple-slash docs on private items, doc paragraphs that restate the fn name | `naming-and-comments` |
| `mod.rs` added | `naming-and-comments` |
| `unsafe`, `transmute`, `*mut`, `*const`, `extern "C"` | invoke `/unsafe-checker` skill |
| `#[pyfunction]`, `#[pymethods]`, `Python<'_>`, `PyResult` | switchyard PyO3 section in main `SKILL.md` |
| Files under `crates/switchyard-translation/` | switchyard translation section in main `SKILL.md` |

## Decision flow

1. **Run the greps above** against `git diff`. Each hit adds a lens to your pass list.
2. **Read each lens's reference** before doing its pass — they're short and signal-specific.
3. **For design-level concerns**, invoke the matching `/m*-*` skill. Use these especially when:
   - You can't articulate *why* a pattern is bad, just that it feels off.
   - The author may push back — the m-skill gives you the design argument, not just the rule.
   - The smell points up a layer: a `Arc<Mutex<…>>` everywhere isn't a synchronization bug, it's a domain-modeling bug (`/m09-domain`); `.clone()` everywhere isn't a perf bug, it's an ownership-design bug (`/m01-ownership`).

## Anti-patterns in dispatch itself

- **Don't apply every lens to every diff.** A 5-line bug fix doesn't need an ownership review.
- **Don't skip dispatch on "small" diffs.** A 5-line diff that adds `Arc<Mutex<HashMap<…>>>` is exactly when ownership/concurrency review matters most.
- **Don't quote the m-skill at the author.** Translate it to a `file:line` comment in their PR. The skills are for you, the reviewer.
