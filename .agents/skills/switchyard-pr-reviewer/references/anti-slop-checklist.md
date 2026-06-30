# Anti-AI-slop checklist

Run this in Phase 3 (verification) and Phase 6 (slop filter). It encodes the real false
alarms and over-flagging seen across past reviews. The bar: a comment must teach the author
something true that they'd act on.

## A. Refute-before-report (Phase 3)
A candidate dies unless you re-read the code and it still holds. Refute by execution or grep,
never by vibes.

| Refute by checking… | Real false alarms it would have killed |
|---|---|
| Read the actual path end-to-end | "streaming bypasses detection" — it doesn't; "ContextWindowExceeded → 500" — it's caught/translated |
| Confirm the grep tree is the PR branch | "zero consumers" false negative from grepping a sibling branch lacking this PR's Python |
| Check the error hierarchy | "catch-order shadows ConfigError" — siblings, not parent/child; order is harmless |
| Check lifecycle/statefulness | "lazy translator singleton is a leak" — stateless, no startup/shutdown |
| Read both files before "duplication" | context_error vs upstream_error do different jobs — not reuse |
| Diagnose the environment | local test failing on rustc 1.92-vs-1.94 is an env issue, NOT a PR defect |
| Confirm wiring is live | call something dead only after grepping all consumers (incl. FFI re-exports + tests) |

Self-correction rules:
- **Verify a post by CONTENT, not the API response** (`thread:null` can be misleading; a
  comment that appeared may be the user's UI comment, not yours).
- **Re-verify on the moving head** before every post; diff the new commits to confirm your
  anchor line is unchanged.
- **Verify fixes against code, not commit messages** ("commit messages aren't proof"); catch
  PARTIAL fixes (e.g. `.expect()` fixed in the anchored file but still present in the sibling
  files your comment named) and leave that thread open.
- If you reasoned instead of running, **say so** ("this is reasoning, not a test").

## B. Don't-post / fold-into-summary (Phase 6)
Drop or fold, don't thread:
- Restates what the code obviously says; obvious nit matching the file's existing conventions.
- AI over-enumeration: near-identical tests / repeated assertions → recommend collapsing to ~3, name it once.
- Pre-existing / out-of-diff / "not introduced here" → mention once, no thread.
- Retired by real-world context ("legacy path is going away").
- Framework-only / tautological tests as if they were bugs → categorize, don't inflate.
- Style nits not worth their own thread → fold into the summary.
- Demanding Rust unit tests for one-line PyO3 delegations.

## C. Body hygiene
- SHORT and direct (most comments 1–3 lines). No thinking-out-loud, no "wait/actually".
- No project-management refs in the comment (no step/plan IDs); a plain ticket cite in a
  proposed `// TODO(SWITCH-NNN):` is fine.
- Uncertain intent → ask the author ("Intentional?"), don't assert a bug.
- Flag the author's own thinking-out-loud comments left in source.
- Severity honesty: judgment calls are Minor/Style or questions, not Blocking; correctness
  outranks cleanup.

## D. Process hygiene
- Draft first; post only the approved subset; never auto-post; never post the full agent dump.
- Never probe-spam throwaway comments to find a commentable line — re-anchor on a consuming line.
- Never retry a failing mutation into the user's live pending review; never touch/submit the
  user's comments. Resolve only your own threads.
- Comments-only means comments-only: do not fix/commit when only comments were requested.
- One chunk at a time when walking the user through; don't dump a wall of text.
