# Domain error & resilience review

**Open when the diff has**: retry loops, backoff, circuit breakers, fallbacks, "graceful degradation," or domain-error hierarchies.

**Companion skill**: `/m13-domain-error`.

## The design question

**Is this error transient (retry helps), permanent (retry doesn't), or a contract violation (caller's bug)?** Different categories need different responses. Treating all errors the same — retry-everything, or fail-on-anything — is the most common architectural mistake in network code.

| Category | Example | Response |
|---|---|---|
| Transient | timeout, 503, connection reset | retry with backoff |
| Permanent | 400, parse error, auth | surface to caller |
| Contract | invariant violated | panic / fail loud |
| Degraded | one upstream of N is down | fall back, log, continue |

## Red flags

### 1. Retry on every error
```rust
// Smell
loop {
    match call().await {
        Ok(v) => return Ok(v),
        Err(_) => sleep(Duration::from_secs(1)).await,
    }
}
```
This retries `BadRequest` forever. Match on the error variant and only retry the transient ones (timeouts, 5xx, connection drops).

### 2. Fixed-interval retry
Sleeping `1s` between retries hammers an upstream that's already struggling. Use exponential backoff with jitter (`tokio::time::sleep` over `2^n + rand`). The `backon` or `tokio-retry` crates do this; rolling your own is fine if simple.

### 3. Retry with no max attempts / no max duration
Infinite retry hides bugs. Cap at N attempts *and* a total deadline. The deadline matters more — N attempts at exponential backoff can sleep for hours.

### 4. No circuit breaker on a high-volume dependency
If you're calling a flaky upstream at 1k/s, retry amplifies its problem (retry storm). A circuit breaker (`tower::limit`, `failsafe`, hand-rolled) opens after a failure threshold and short-circuits subsequent calls — protects both you and the upstream.

### 5. Fallback that silently degrades
```rust
let result = primary().await.unwrap_or_else(|_| secondary());
```
No log, no metric, no signal. Operators have no idea the primary is failing. Always log/meter the fallback path with `tracing::warn!` and a counter.

### 6. Retry inside a request handler with no deadline
Web handler that retries 5× × 1s each = 5s p99. Set a request-wide timeout (`tokio::time::timeout`) above the retry loop, so the user gets a 504 instead of a connection that hangs.

### 7. Error type that doesn't distinguish categories
```rust
// Smell — caller can't decide whether to retry
enum BackendError { Failed(String) }

// Better
enum BackendError {
    Transient { source: anyhow::Error },   // worth retrying
    Permanent { status: u16, body: String }, // do not retry
    InvalidRequest(String),                  // caller bug
}
```
The error type is the API contract for failure. If it can't drive the retry decision, the retry decision will be wrong somewhere.

### 8. `match` on string error messages
`if e.to_string().contains("timeout") { … }` is fragile and breaks the moment the upstream changes wording. Match on the typed variant or status code.

### 9. Catching too broadly
`Result<T, Box<dyn Error>>` in a retry decision means you can't see what failed. The retry layer should see a typed error.

## Acceptable cases

- A single retry on a known-transient operation with a small fixed delay — readable inline code beats pulling in a retry crate.
- Best-effort writes (analytics, telemetry) where loss is acceptable — but log and meter the drop.
- Panic on a contract violation discovered at runtime (programmer bug). That's `panic!`'s job.

## Comment templates

- "Retrying every error variant — this retries `BadRequest` forever. Match on the kind first."
- "Fixed `sleep(1s)` between retries — add exponential backoff + jitter."
- "No max attempts → infinite retry on a permanent failure. Cap it and add a deadline."
- "Fallback path is silent — log and meter, otherwise we won't notice degradation."
- "`BackendError::Failed(String)` can't drive a retry decision. Split into `Transient`/`Permanent`/`InvalidRequest`."

## Trace up

Resilience patterns belong in the domain, not in a generic util. Where you put the retry/fallback/circuit decides what gets protected and what doesn't. That's an architectural question (`/m09-domain`). Don't accept a "wrap everything in a retry helper" PR without asking what the failure boundaries actually are.
