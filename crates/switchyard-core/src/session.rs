// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session-affinity primitives: a stable per-conversation key derived from a
//! request body and a bounded, access-ordered LRU cache keyed by that string.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;

use lru::LruCache;
use serde_json::Value;

/// Derive a stable per-conversation key from a request body.
///
/// Hashes only the prefix a harness never rewrites — system prompt + first user
/// message — so every turn of a conversation shares a key while distinct
/// conversations differ. Returns a 16-char lowercase hex string.
pub fn session_key_from_body(body: &Value) -> String {
    let mut hasher = DefaultHasher::new();

    // Anthropic carries the system prompt at the top level.
    flatten_text(body.get("system")).hash(&mut hasher);

    // OpenAI uses "messages"; the Responses API uses "input". A messages list
    // with no user message falls through to "input".
    let mut anchored = false;
    for seq_key in ["messages", "input"] {
        if let Some(Value::Array(items)) = body.get(seq_key) {
            for item in items {
                let Some(role) = item.get("role").and_then(Value::as_str) else {
                    continue;
                };
                match role {
                    "system" | "developer" => {
                        flatten_text(item.get("content")).hash(&mut hasher);
                    }
                    "user" => {
                        flatten_text(item.get("content")).hash(&mut hasher);
                        anchored = true;
                        break;
                    }
                    _ => {}
                }
            }
        }
        if anchored {
            break;
        }
    }

    format!("{:016x}", hasher.finish())
}

/// Flatten a message-content value into a single string for hashing: strings
/// pass through, content-block arrays concatenate their first non-empty
/// `text`/`content` field (or the raw block for non-objects), null/absent yield
/// empty, and other scalars stringify. Text-only by design (block metadata such
/// as `cache_control` is excluded so it can't perturb the key).
fn flatten_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => {
            let mut out = String::new();
            for block in blocks {
                if let Value::Object(map) = block {
                    let text = map
                        .get("text")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .or_else(|| {
                            map.get("content")
                                .and_then(Value::as_str)
                                .filter(|s| !s.is_empty())
                        });
                    if let Some(text) = text {
                        out.push_str(text);
                    }
                } else {
                    out.push_str(&block.to_string());
                }
            }
            out
        }
        None | Some(Value::Null) => String::new(),
        Some(other) => other.to_string(),
    }
}

/// A bounded, access-ordered LRU cache keyed by `String`.
///
/// Recency refreshes on both [`SessionCache::get`] and [`SessionCache::put`];
/// the least-recently-used entry is evicted when capacity is exceeded. A
/// capacity of 0 retains nothing.
///
/// NOT thread-safe — intended for single-event-loop use.
pub struct SessionCache<V> {
    cache: Option<LruCache<String, V>>,
}

impl<V> SessionCache<V> {
    /// Create a cache holding at most `max_sessions` entries (0 retains nothing).
    pub fn new(max_sessions: usize) -> Self {
        Self {
            cache: NonZeroUsize::new(max_sessions).map(LruCache::new),
        }
    }

    /// Look up `key`, refreshing its recency to most-recently-used.
    pub fn get(&mut self, key: &str) -> Option<&V> {
        self.cache.as_mut().and_then(|c| c.get(key))
    }

    /// Insert `value` as most-recently-used, evicting the LRU entry if over
    /// capacity. No-op when capacity is 0.
    pub fn put(&mut self, key: String, value: V) {
        if let Some(c) = self.cache.as_mut() {
            c.put(key, value);
        }
    }

    /// Number of entries currently retained.
    pub fn len(&self) -> usize {
        self.cache.as_ref().map_or(0, LruCache::len)
    }

    /// Whether the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The configured maximum number of sessions.
    pub fn max_sessions(&self) -> usize {
        self.cache.as_ref().map_or(0, |c| c.cap().get())
    }

    /// Iterate over the retained values (order unspecified).
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.cache.iter().flat_map(|c| c.iter().map(|(_, v)| v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn get_refreshes_recency() {
        let mut cache: SessionCache<i32> = SessionCache::new(2);
        cache.put("a".to_string(), 1);
        cache.put("b".to_string(), 2);
        // Touch "a" so "b" becomes the LRU entry.
        assert!(cache.get("a").is_some());
        cache.put("c".to_string(), 3);
        assert!(cache.get("a").is_some());
        assert!(cache.get("b").is_none());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn eviction_bounds_len() {
        let mut cache: SessionCache<i32> = SessionCache::new(2);
        cache.put("a".to_string(), 1);
        cache.put("b".to_string(), 2);
        cache.put("c".to_string(), 3);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn zero_capacity_retains_nothing() {
        let mut cache: SessionCache<&str> = SessionCache::new(0);
        cache.put("a".to_string(), "x");
        assert!(cache.get("a").is_none());
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.max_sessions(), 0);
    }

    #[test]
    fn session_key_stable_across_appended_turns() {
        let base = json!({
            "system": "you are helpful",
            "messages": [
                {"role": "user", "content": "first question"},
            ],
        });
        let extended = json!({
            "system": "you are helpful",
            "messages": [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "an answer"},
                {"role": "user", "content": "a follow up"},
            ],
        });
        assert_eq!(
            session_key_from_body(&base),
            session_key_from_body(&extended)
        );
    }

    #[test]
    fn session_key_distinct_on_system_or_first_user() {
        let base = json!({
            "system": "you are helpful",
            "messages": [{"role": "user", "content": "first question"}],
        });
        let diff_system = json!({
            "system": "you are terse",
            "messages": [{"role": "user", "content": "first question"}],
        });
        let diff_user = json!({
            "system": "you are helpful",
            "messages": [{"role": "user", "content": "another question"}],
        });
        assert_ne!(
            session_key_from_body(&base),
            session_key_from_body(&diff_system)
        );
        assert_ne!(
            session_key_from_body(&base),
            session_key_from_body(&diff_user)
        );
    }

    #[test]
    fn session_key_blocks_equal_plain() {
        let blocks = json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        });
        let plain = json!({
            "messages": [{"role": "user", "content": "hi"}],
        });
        assert_eq!(
            session_key_from_body(&blocks),
            session_key_from_body(&plain)
        );
    }

    #[test]
    fn key_is_16_char_hex() {
        let key = session_key_from_body(&json!({"messages": [{"role": "user", "content": "x"}]}));
        assert_eq!(key.len(), 16);
        assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn session_key_uses_input_when_messages_has_no_user() {
        // A messages list with no user message falls through to `input`; the
        // first user there anchors the key (proven by it affecting the hash).
        let alpha = json!({
            "messages": [{"role": "system", "content": "sys"}],
            "input": [{"role": "user", "content": "alpha"}],
        });
        let beta = json!({
            "messages": [{"role": "system", "content": "sys"}],
            "input": [{"role": "user", "content": "beta"}],
        });
        assert_ne!(session_key_from_body(&alpha), session_key_from_body(&beta));
    }

    #[test]
    fn session_key_supports_responses_input_only() {
        // Responses API bodies carry turns under `input` with no `messages`.
        let a = json!({"input": [{"role": "user", "content": "x"}]});
        let b = json!({"input": [{"role": "user", "content": "y"}]});
        assert_ne!(session_key_from_body(&a), session_key_from_body(&b));
    }

    #[test]
    fn session_key_includes_system_message_in_messages() {
        // A system/developer message inside `messages` (OpenAI shape) contributes.
        let a = json!({
            "messages": [
                {"role": "system", "content": "sys A"},
                {"role": "user", "content": "q"},
            ],
        });
        let b = json!({
            "messages": [
                {"role": "system", "content": "sys B"},
                {"role": "user", "content": "q"},
            ],
        });
        assert_ne!(session_key_from_body(&a), session_key_from_body(&b));
    }

    #[test]
    fn get_missing_returns_none() {
        let mut cache: SessionCache<i32> = SessionCache::new(2);
        assert!(cache.get("nope").is_none());
        cache.put("a".to_string(), 1);
        assert_eq!(cache.get("a"), Some(&1));
        assert!(cache.get("b").is_none());
    }

    #[test]
    fn put_overwrites_existing_value() {
        // Re-pinning a session updates its value (mirrors pin-on-success).
        let mut cache: SessionCache<i32> = SessionCache::new(2);
        cache.put("a".to_string(), 1);
        cache.put("a".to_string(), 2);
        assert_eq!(cache.get("a"), Some(&2));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn max_sessions_reports_capacity() {
        let cache: SessionCache<i32> = SessionCache::new(5);
        assert_eq!(cache.max_sessions(), 5);
    }

    #[test]
    fn values_yields_all_retained() {
        let mut cache: SessionCache<i32> = SessionCache::new(3);
        cache.put("a".to_string(), 1);
        cache.put("b".to_string(), 2);
        let mut vals: Vec<i32> = cache.values().copied().collect();
        vals.sort();
        assert_eq!(vals, vec![1, 2]);
    }
}
