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

use anyhow::{anyhow, bail, Context};
use key_management_service::KeyManagementService;
use kms_proto::fcp::confidentialcompute::key_management_service_server::KeyManagementServiceServer;
use oak_attestation_types::{attester::Attester, endorser::Endorser};
use oak_attestation_verification_types::util::Clock;
use oak_proto_rust::oak::attestation::v1::{Endorsements, Evidence, ReferenceValues};
use oak_sdk_common::{StaticAttester, StaticEndorser};
use oak_sdk_containers::{InstanceSessionBinder, OrchestratorClient};
use prost::Message;
use session_v1_service_proto::oak::services::oak_session_v1_service_client::OakSessionV1ServiceClient;
use storage_actor::StorageActor;
use storage_client::StorageClient;
use tcp_proto::runtime::endpoint::endpoint_service_server::EndpointServiceServer;
use tcp_runtime::service::TonicApplicationService;

fn get_reference_values(evidence: &Evidence) -> anyhow::Result<ReferenceValues> {
    // TODO: b/400476265 - Add ReferenceValues for SEV-SNP.
    match evidence.root_layer.as_ref().map(|rl| rl.platform) {
        None => {
            // When running in insecure mode, simply skip all reference values.
            // This is only used for tests.
            ReferenceValues::decode(include_bytes!(env!("INSECURE_REFERENCE_VALUES")).as_slice())
                .context("failed to decode insecure ReferenceValues")
        }
        Some(platform) => bail!("platform {} is not supported", platform),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    log::info!("KMS starting...");
    log::set_max_level(log::LevelFilter::Warn);

    let channel = oak_sdk_containers::default_orchestrator_channel()
        .await
        .context("failed to create orchestrator channel")?;
    let mut orchestrator_client = OrchestratorClient::create(&channel);
    let endorsed_evidence = orchestrator_client
        .get_endorsed_evidence()
        .await
        .context("failed to get endorsed evidence")?;
    let evidence =
        endorsed_evidence.evidence.ok_or_else(|| anyhow!("EndorsedEvidence.evidence not set"))?;
    let endorsements = endorsed_evidence
        .endorsements
        .ok_or_else(|| anyhow!("EndorsedEvidence.endorsements not set"))?;

    let attester = CloneableAttester { inner: Arc::new(StaticAttester::new(evidence.clone())) };
    let endorser = CloneableEndorser { inner: Arc::new(StaticEndorser::new(endorsements)) };
    let session_binder = InstanceSessionBinder::create(&channel);
    let reference_values = get_reference_values(&evidence)?;
    let clock = Arc::new(SystemClock {});

    // Create the KeyManagementService.
    let session_service_client = OakSessionV1ServiceClient::connect("http://[::1]:8008")
        .await
        .context("failed to create OakSessionV1ServiceClient")?;
    let key_management_service = KeyManagementService::new(StorageClient::new(
        session_service_client,
        KeyManagementService::get_init_request,
        attester.clone(),
        endorser.clone(),
        session_binder.clone(),
        reference_values.clone(),
        clock.clone(),
    ));

    // Create the TCP EndpointService.
    let endpoint_service = TonicApplicationService::new(channel, evidence, move || {
        StorageActor::new(attester, endorser, session_binder, reference_values, clock)
    });

    // Start the gRPC server.
    orchestrator_client.notify_app_ready().await.context("failed to notify that app is ready")?;
    tonic::transport::Server::builder()
        .add_service(KeyManagementServiceServer::new(key_management_service))
        .add_service(EndpointServiceServer::new(endpoint_service))
        .serve("[::]:8080".parse()?)
        .await?;
    Ok(())
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

#[derive(Clone)]
struct CloneableAttester {
    inner: Arc<StaticAttester>,
}
impl Attester for CloneableAttester {
    fn extend(&mut self, _encoded_event: &[u8]) -> anyhow::Result<()> {
        anyhow::bail!("This attester type is finalized and can not be extended.")
    }

    fn quote(&self) -> anyhow::Result<Evidence> {
        self.inner.quote()
    }
}

#[derive(Clone)]
struct CloneableEndorser {
    inner: Arc<StaticEndorser>,
}
impl Endorser for CloneableEndorser {
    fn endorse(&self, evidence: Option<&Evidence>) -> anyhow::Result<Endorsements> {
        self.inner.endorse(evidence)
    }
}
