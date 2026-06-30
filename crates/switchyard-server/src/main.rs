// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Binary entrypoint for `switchyard-server`.

use std::process::ExitCode;

mod cli;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> ExitCode {
    match cli::run(cli::ServerArgs::parse_args()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
