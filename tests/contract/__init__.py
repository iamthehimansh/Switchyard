# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the API surface that downstream consumers (e.g. the
NeMo Platform `nemo-switchyard` middleware plugin) depend on.

These tests intentionally restate import paths and class shapes verbatim so a
PR that renames, deletes, or refactors a symbol fails *here* with a clear
message instead of breaking downstream at integration time.

If a test in this suite fails because you intentionally changed the contract,
that's the signal to coordinate the migration with downstream consumers
*before* merging — don't silently update the test to match the new shape.
"""
