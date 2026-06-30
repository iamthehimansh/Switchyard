# Ownership & borrowing review

**Open when the diff has**: `.clone()`, lifetimes (`'a`, `'static`), moves, `&mut` patterns, or borrow-checker workarounds.

**Companion skill**: `/m01-ownership` (design framing — "who should own this data?").

## The design question

Before flagging a `.clone()` or suggesting a lifetime, ask: **who should own this, and for how long?** An over-cloned codebase isn't a perf problem first — it's a sign nobody decided the ownership model.

## Red flags

### 1. Reflexive `.clone()`
```rust
// Smell
fn handle(req: ChatRequest, headers: HashMap<String, String>) {
    let h = headers.clone();  // why?
    forward(req.clone(), h);
}
```

Ask:
- Can the caller pass by reference? Does `forward` actually need to *own* it?
- Is the type `Copy` already? Then `.clone()` is pure noise.
- Is the type a cheap handle (`Arc`, `Sender`, `tokio::sync::mpsc::Sender`)? `Arc::clone(&x)` is intentional; `x.clone()` for the same effect is a style call but write the intent.

### 2. `.clone()` to satisfy the borrow checker
A `.clone()` introduced because "the borrow checker complained" almost always means the function's interface is wrong — it asks for ownership when it only needs a borrow, or vice versa. Fix the signature, not the call site.

### 3. `'static` everywhere
`'static` on a function parameter or struct field is a strong constraint. It's usually a sign the author wanted to silence a lifetime error rather than think about the actual scope. Flag and ask: does this *really* need to live for the program's lifetime?

### 4. Owned `String` where `&str` would do
```rust
// Smell — forces caller to allocate
fn parse_model(model: String) -> Result<Model> { … }

// Better
fn parse_model(model: &str) -> Result<Model> { … }
```
Return owned, accept borrowed — unless you genuinely need to keep it around.

### 5. `Vec<T>` taken by value when iterated once
```rust
// Smell
fn sum(values: Vec<u64>) -> u64 { values.iter().sum() }

// Better
fn sum(values: &[u64]) -> u64 { values.iter().sum() }
```

### 6. Preserved `_var` that's actually used
`_text` → `text` once it's read. The underscore was a "TODO: maybe later" that got committed.

## Acceptable cases

- `.clone()` on `Arc`/`Sender`/`Bytes` — these are cheap handles. Just keep them out of hot paths if cloned per-request.
- `.clone()` of a small `Copy`-adjacent value (a `String` ID used twice) where readability beats one extra alloc — but say so in the comment if non-obvious.
- `'static` on background tasks spawned with `tokio::spawn` — that bound is real, not avoidable.

## Comment templates

- "This `.clone()` looks reflexive — `&headers` should work since `forward` only reads."
- "Why owned `String` here? Caller has to allocate. `&str` parses fine."
- "`'static` here is doing a lot of work — what's the actual scope? Looks like the borrow can be tightened to `'a`."

## Trace up

If you find ownership confusion all over a module, that's not a `.clone()` problem — it's a domain-modeling problem. The author hasn't decided which type owns the data. Comment at the design level, not on each clone. See `/m09-domain` and `/m01-ownership`.
