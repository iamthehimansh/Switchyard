# Type-driven design review

**Open when the diff has**: `PhantomData`, marker traits, newtypes, builder patterns, type-state transitions, or types that "encode an invariant."

**Companion skill**: `/m05-type-driven`.

## The design question

**Can the type system make this invalid state unrepresentable?** Most "validate at runtime" code is "we didn't model it at the type level." Rust's expressive types pay back the up-front cost in deleted runtime checks.

## Red flags

### 1. Boolean fields that should be enums
```rust
// Smell
struct Request { is_streaming: bool, is_anthropic: bool, is_responses: bool }
// What does (true, true, true) mean? Nothing. But it compiles.

// Better
enum Format { OpenAiChat, OpenAiResponses, Anthropic }
struct Request { format: Format, streaming: bool }
```
If two booleans can't both be true, that's an enum. If three can't all coexist, definitely an enum.

### 2. `String` for things with structure
`model: String` where the legal values are `"openai/gpt-4"`, `"anthropic/claude-3"`, etc. Wrap it in a newtype that parses on construction: `Model::parse(&s)` returns `Result<Model, ParseModelError>`. Every other function in the codebase can then take `&Model` knowing it's valid.

### 3. Validation that runs on every method
```rust
impl Connection {
    fn send(&self, msg: Msg) -> Result<()> {
        if !self.is_open { return Err(NotOpen); }
        …
    }
    fn close(&self) -> Result<()> {
        if !self.is_open { return Err(NotOpen); }
        …
    }
}
```
The `is_open` check is everywhere because the type doesn't encode the state. Typestate would: `Connection<Open>` has `send`/`close`; `Connection<Closed>` has neither. The check becomes a compile error instead of a runtime one.

### 4. Builder with `Option<T>` for every required field
```rust
// Smell — caller can call build() with nothing set
struct ServerBuilder { host: Option<String>, port: Option<u16> }
impl ServerBuilder {
    fn build(self) -> Result<Server> {
        Ok(Server {
            host: self.host.ok_or("host required")?,
            port: self.port.ok_or("port required")?,
        })
    }
}
```
A type-state builder uses different types for "configured" vs "missing" so `.build()` only compiles when everything is set. Worth it for builders with many required fields and confusing failure modes.

### 5. `PhantomData` without explanation
`PhantomData<T>` is meaningful — it tells the compiler the type "owns" a `T` for variance/dropck. If you see it without a comment, ask why. Often it's leftover from a refactor.

### 6. Newtype with all the inner type's methods re-exposed
If you write `pub struct Email(pub String)` and then `impl Email { pub fn as_str(&self) -> &str { &self.0 } pub fn into_string(self) -> String { self.0 } }` and `pub` the inner field — you've added zero safety, just noise. The point of a newtype is to *restrict* what callers can do.

## Acceptable cases

- `PhantomData<&'a T>` for explicit variance on lifetime-parameterized types — write a comment.
- A boolean field when the two states are genuinely orthogonal to everything else.
- A builder with `Option<T>` for *optional* fields (with sensible defaults at `build()`).

## Comment templates

- "Three booleans that can't coexist — should be an enum."
- "`String` model name — could be a `Model` newtype that parses at the boundary, so the rest of the code doesn't re-validate."
- "Every method checks `is_open` — typestate would make this a compile-time guarantee."
- "`PhantomData` without a comment — what's it doing?"

## Trace up

Type-driven design is the bridge from Layer 1 (mechanics) to Layer 2 (design). When you see a lot of runtime validation, the issue isn't "missing checks" — it's that the domain model is shaped wrong (`/m09-domain`).
