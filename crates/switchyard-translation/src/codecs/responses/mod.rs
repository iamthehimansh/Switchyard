// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI Responses buffered and streaming codecs.

mod buffered;
mod stream;

pub use buffered::OpenAiResponsesCodec;
pub use stream::OpenAiResponsesStreamCodec;
