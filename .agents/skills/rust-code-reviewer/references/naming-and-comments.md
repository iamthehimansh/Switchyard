# Naming & comments review

**Open when the diff has**: new identifiers, doc comments, module reshuffles, or anything renamed.

**Companion skill**: `/coding-guidelines`.

## The design question

**Does the name carry the right amount of meaning — no more, no less?** A name that implies less than it does fails its reader. A name that implies more sets a false expectation. Both are bugs.

## Naming

### 1. Don't promise what you don't do
- `serve` implies a long-running server loop. If your function does one request, call it `handle` or `process`.
- `connect` implies a network connection — don't use it for "create a struct that might dial later."
- `Manager` / `Handler` / `Helper` / `Util` — too generic to mean anything. Pick something that names the responsibility.

### 2. Boolean prefixes
Use `is_`, `has_`, `needs_`, `can_`. The truthy meaning must be obvious without reading the body.
- `fn is_streaming(req: &Req) -> bool` ✅
- `fn streaming(req: &Req) -> bool` ❌
- `let cancelled = …` ✅ (`cancelled` reads as a state)
- `let cancel = bool::…` ❌ (verb where you want adjective)

### 3. Conversion conventions
- `as_X` — cheap reference conversion (`as_str`, `as_bytes`). No allocation.
- `to_X` — expensive owned conversion (`to_string`, `to_owned`). Allocates.
- `into_X` — by-value conversion that consumes self.

Mis-using these misleads readers about cost.

### 4. Getters
Rust convention: omit `get_`. `fn name(&self) -> &str` not `fn get_name(&self) -> &str`. Setters keep `set_` only if you're matching an external API.

### 5. Module file naming
- `mod.rs` is the older convention. Prefer `parent/foo.rs` + `parent/foo/` over `parent/foo/mod.rs` — easier to find, no naming collisions in tabs.
- File name = `snake_case` of its primary class/struct/trait. Rename on touch.

### 6. Don't preserve `_` on used variables
`_text` was "TODO maybe later." Once read, drop the `_`. Otherwise lints don't help and readers wonder why it's there.

### 7. Lifetimes that mean something
`'a` is fine for one. `'src`, `'req` are better when multiple lifetimes coexist. Don't randomly name them `'b`, `'c` — pick names that say what's tied together.

### 8. Type names
- `CamelCase` for types, traits, enums.
- `SCREAMING_SNAKE_CASE` for consts and statics.
- `snake_case` for fns, vars, modules.
- One-word type names beat two-word. `Server` beats `ServerObject`.

## Comments

### 1. Delete restating comments
```rust
// Smell — says what the line says
// increment counter
counter += 1;

// Smell — restates the fn name
/// Parses the model name
fn parse_model_name(s: &str) -> Result<Model> { … }
```
If the code reads cleanly, the comment is noise. If you need a comment to explain *what*, fix the name.

### 2. Comments earn their place by saying *why*
Good comments answer: "Why this, not the obvious alternative?" "What invariant is this protecting?" "What surprised me when I wrote this?" Not: "this calls X then Y."

### 3. Don't put history in comments
"Previously used a HashMap, switched to Vec for cache locality" — that belongs in the commit message. Comments are read by people who don't care what it used to be.

### 4. AI-generated comment smell
LLMs love restating. If every line has an above-the-line comment, ask the author to delete the obvious ones and rephrase the rest. Pattern to look for:
- `// Check if X` above `if x { … }`
- `// Loop through Y` above `for y in … { … }`
- A multi-line docstring on a 3-line function.

### 5. `///` vs `//`
- `///` is doc — appears in `cargo doc`. For public items, write it. For private items, only when there's something non-obvious to say.
- `//` is internal.
- Don't mix `///` on private fields of a private struct — it ends up half-published.

### 6. Copyright headers
Switchyard convention: only the two SPDX lines. Anything beyond is noise — trim.
```rust
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
```

### 7. `TODO`/`FIXME` without a name or ticket
A bare `// TODO: handle this` rots. Either link to a ticket (`// TODO(PROJ-123): …`) or fix it now. Reviewer-suggested: ask for one or the other before merge.

### 8. Test comments
Test names are documentation. `test_returns_404_when_model_missing` beats `test_missing_model` + a comment. Use the name slot.

## Comment templates

- "`serve` reads like a long-running loop — this is a one-shot handler. `handle_request`?"
- "`Manager` doesn't tell me what it's responsible for. Pick something specific."
- "This `//` restates the line — delete it."
- "`get_name` — drop `get_`, just `name(&self)`."
- "`'a`, `'b`, `'c` — name them by what they tie to (`'req`, `'src`)."
- "`_text` is read on line N — drop the underscore."
- "Bare `// TODO` — link a ticket or fix it now."
- "Doc comment restates the function name. Either say why, or delete."

## Trace up

Persistent naming confusion in a module signals that the concepts aren't clean. The fix isn't a rename — it's deciding what each concept *is*. That's domain modeling (`/m09-domain`).
