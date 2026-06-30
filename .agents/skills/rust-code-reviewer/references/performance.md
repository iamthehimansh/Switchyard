# Performance review

**Open when the diff has**: hot paths, benchmarks, allocation patterns, large data structures, or anything labelled "optimization."

**Companion skill**: `/m10-performance`.

## The design question

**Did the author measure?** If a PR says "optimization" without a baseline, ask for one. Otherwise the change is taste-based and you can't tell if it helped.

Typical impact hierarchy (don't optimize down a level until the one above is done):
1. **Algorithm** (10–1000×)
2. **Data structure** (2–10×)
3. **Allocation reduction** (2–5×)
4. **Cache layout / SIMD / micro-tuning** (1.5–3×)

## Red flags

### 1. "Optimization" without measurement
No `criterion` numbers, no flamegraph, no before/after — just intuition. Reject or downgrade to "consider for later." Especially flag if the change makes the code less readable.

### 2. Benchmarks in debug mode
`cargo bench` runs release by default, but ad-hoc timing with `Instant::now()` in `cargo run` (debug) tells you nothing. Always `--release`.

### 3. `.clone()` in a hot loop
A clone per request × N requests/sec = throughput tax. See [`ownership-and-borrowing.md`](ownership-and-borrowing.md). For strings consider `Cow<'_, str>` if it sometimes-borrows-sometimes-owns.

### 4. `Vec::new()` then loop of `.push()` with known size
```rust
// Smell
let mut out = Vec::new();
for x in input { out.push(transform(x)); }

// Better
let mut out = Vec::with_capacity(input.len());
// or
let out: Vec<_> = input.iter().map(transform).collect();
```
`with_capacity` avoids `2 + 4 + 8 + 16 + …` re-allocations during growth.

### 5. `String` concatenation in a loop with `+`/`format!`
O(n²). Use `String::with_capacity(…)` + `push_str`, or `itertools::join`, or write into a single `String` once.

### 6. `HashMap` for tiny / fixed key sets
`HashMap` hashing isn't free. For < ~16 entries, a `Vec<(K, V)>` with linear search is often faster (cache-friendly). For known compile-time keys, an enum-keyed array or a `match` is faster still.

### 7. `LinkedList`
Almost always wrong in Rust. No cache locality, pointer-chasing, allocation per node. Use `Vec` or `VecDeque`.

### 8. Hidden allocations in hot paths
- `.collect::<Vec<_>>()` when iteration would suffice.
- `format!()` to build a one-off log message you never structured.
- Returning `String` where `&str` from the input would work.
- `to_string()` on something that's already `Display`.

### 9. `Box<[T]>` vs `Vec<T>`
If the collection is built once and never resized, `Box<[T]>` is smaller (no capacity field) and clearer in intent.

### 10. Synchronous I/O blocking the runtime
A file read or DNS lookup on the executor thread tanks p99 latency for everything. See [`async-and-concurrency.md`](async-and-concurrency.md).

## Acceptable cases

- A clone on a small string for readability in a non-hot path.
- `Vec::new()` (no capacity) when the size is unpredictable and tiny.
- `HashMap` for any non-tiny dynamic key set — its O(1) amortized beats linear scan.

## Comment templates

- "This PR is labelled optimization — got a before/after `criterion` run?"
- "Hot-path `.clone()` on every request. `&` or `Arc::clone` if shared."
- "`Vec::new()` then 1000 `push`es — `Vec::with_capacity(1000)`."
- "`LinkedList` in Rust is almost never the answer — `VecDeque`?"
- "`format!()` to build a log message that becomes a structured field anyway — drop the `format!`."

## Trace up

Real perf wins usually come from changing what you compute, not how. If a function is slow, look at the call site: can it skip work? Can it batch? Can it cache? That's an architectural question (`/m09-domain`), not a tight-loop question.
