// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![no_std]
#![no_main]
#![feature(alloc_error_handler)]

extern crate alloc;

use alloc::boxed::Box;
use ledger_service::actor::LedgerActor;
use oak_restricted_kernel_sdk::{
    attestation::InstanceAttester,
    channel::{start_blocking_server, FileDescriptorChannel},
    crypto::InstanceSigner,
    entrypoint,
    utils::samplestore::StaticSampleStore,
};
use tcp_proto::runtime::endpoint::EndpointServiceServer;
use tcp_runtime::service::ApplicationService;

#[entrypoint]
fn run_server() -> ! {
    let mut invocation_stats = StaticSampleStore::<1000>::new().unwrap();
    let actor = LedgerActor::create(
        Box::new(InstanceAttester::create().unwrap()),
        Box::new(InstanceSigner::create().unwrap()),
    )
    .expect("LedgerActor failed to create");
    let service: ApplicationService<LedgerActor> = ApplicationService::new(actor);
    let server = EndpointServiceServer::new(service);
    start_blocking_server(Box::<FileDescriptorChannel>::default(), server, &mut invocation_stats)
        .expect("server encountered an unrecoverable error");
}
