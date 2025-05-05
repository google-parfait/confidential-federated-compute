// Copyright 2025 Google LLC.
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

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{bail, Context};
use key_management_service::{get_init_request, KeyManagementService};
use kms_proto::fcp::confidentialcompute::key_management_service_server::KeyManagementServiceServer;
use oak_attestation_verification_types::util::Clock;
use oak_proto_rust::oak::attestation::v1::{Evidence, ReferenceValues, TeePlatform};
use oak_sdk_common::{StaticAttester, StaticEndorser};
use oak_sdk_containers::{InstanceSigner, OrchestratorClient};
use prost::Message;
use session_v1_service_proto::oak::services::oak_session_v1_service_client::OakSessionV1ServiceClient;
use slog::Drain;
use storage_actor::StorageActor;
use storage_client::GrpcStorageClient;
use tcp_proto::runtime::endpoint::endpoint_service_server::EndpointServiceServer;
use tcp_runtime::service::TonicApplicationService;

fn get_reference_values(evidence: &Evidence) -> anyhow::Result<ReferenceValues> {
    // TODO: b/400476265 - Add ReferenceValues for SEV-SNP.
    match evidence.root_layer.as_ref().map(|rl| rl.platform.try_into()) {
        Some(Ok(TeePlatform::None)) => {
            // When running in insecure mode, simply skip all reference values.
            // This is only used for tests.
            ReferenceValues::decode(include_bytes!(env!("INSECURE_REFERENCE_VALUES")).as_slice())
                .context("failed to decode insecure ReferenceValues")
        }
        platform => bail!("platform {:?} is not supported", platform),
    }
}

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default()
            // TODO: b/398874186 - Review whether this should be increased to Warn.
            .default_filter_or("info"),
    )
    .write_style(env_logger::WriteStyle::Never)
    .init();
    log::info!("KMS starting...");

    let channel = oak_sdk_containers::default_orchestrator_channel()
        .await
        .expect("failed to create orchestrator channel");
    let mut orchestrator_client = OrchestratorClient::create(&channel);
    let endorsed_evidence =
        orchestrator_client.get_endorsed_evidence().await.expect("failed to get endorsed evidence");
    let evidence = endorsed_evidence.evidence.expect("EndorsedEvidence.evidence not set");
    let endorsements =
        endorsed_evidence.endorsements.expect("EndorsedEvidence.endorsements not set");

    let attester = Arc::new(StaticAttester::new(evidence.clone()));
    let endorser = Arc::new(StaticEndorser::new(endorsements));
    let signer = InstanceSigner::create(&channel);
    let reference_values = get_reference_values(&evidence).expect("failed to get reference values");
    let clock = Arc::new(SystemClock {});

    // Create the KeyManagementService. The host matches Oak's `launcher_addr`
    // (https://github.com/search?q=repo%3Aproject-oak%2Foak+launcher_addr&type=code),
    // and the port matches the forwarding rules set by the host.
    let session_service_client = OakSessionV1ServiceClient::connect("http://10.0.2.100:8008")
        .await
        .expect("failed to create OakSessionV1ServiceClient");
    let key_management_service = KeyManagementService::new(
        GrpcStorageClient::new(
            session_service_client,
            get_init_request,
            attester.clone(),
            endorser.clone(),
            signer.clone(),
            reference_values.clone(),
            clock.clone(),
        ),
        signer.clone(),
    );

    // Create the TCP EndpointService, following the TCP recommendation of only
    // logging WARNING and above.
    let logger = slog::Logger::root(
        slog_stdlog::StdLog.filter_level(slog::Level::Warning).fuse(),
        slog::o!(),
    );
    let endpoint_service =
        TonicApplicationService::new(channel, evidence, Some(logger), move || {
            StorageActor::new(attester, endorser, signer, reference_values, clock)
        });

    // Start the gRPC server.
    orchestrator_client.notify_app_ready().await.expect("failed to notify that app is ready");
    tonic::transport::Server::builder()
        .add_service(KeyManagementServiceServer::new(key_management_service))
        .add_service(EndpointServiceServer::new(endpoint_service))
        .serve("[::]:8080".parse().expect("failed to parse address"))
        .await
        .expect("failed to start server");
}

struct SystemClock {}
impl Clock for SystemClock {
    fn get_milliseconds_since_epoch(&self) -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime before Unix epoch")
            .as_millis()
            .try_into()
            .expect("SystemTime too large")
    }
}
