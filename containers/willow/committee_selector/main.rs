// Copyright 2025 The Trusted Computations Platform Authors.
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

use anyhow::{anyhow, bail, Context};
use oak_proto_rust::oak::attestation::v1::{Evidence, ReferenceValues, TeePlatform};
use oak_sdk_containers::OrchestratorClient;
use prost::Message;
use tcp_proto::runtime::endpoint::endpoint_service_server::EndpointServiceServer;
use tcp_runtime::service::TonicApplicationService;
use willow_committee_selector_service::actor::CommitteeSelectorActor;

fn get_reference_values(evidence: &Evidence) -> anyhow::Result<ReferenceValues> {
    match evidence.root_layer.as_ref().map(|rl| rl.platform.try_into()) {
        Some(Ok(TeePlatform::AmdSevSnp)) => {
            ReferenceValues::decode(include_bytes!(env!("REFERENCE_VALUES")).as_slice())
                .context("failed to decode ReferenceValues")
        }
        Some(Ok(TeePlatform::None)) => {
            ReferenceValues::decode(include_bytes!(env!("INSECURE_REFERENCE_VALUES")).as_slice())
                .context("failed to decode insecure ReferenceValues")
        }
        platform => bail!("platform {:?} is not supported", platform),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Only log warnings and errors to reduce the risk of accidentally leaking
    // execution information through debug logs.
    log::set_max_level(log::LevelFilter::Warn);

    let channel = oak_sdk_containers::default_orchestrator_channel()
        .await
        .context("failed to create orchestrator channel")?;
    let mut orchestrator_client = OrchestratorClient::create(&channel);
    let evidence = orchestrator_client
        .get_endorsed_evidence()
        .await
        .context("failed to get endorsed evidence")?
        .evidence
        .ok_or_else(|| anyhow!("EndorsedEvidence.evidence not set"))?;
    let reference_values =
        get_reference_values(&evidence).context("failed to get reference values")?;
    let max_committees = match std::env::var("MAX_NUMBER_OF_COMMITTEES") {
        Ok(val) => match val.parse::<usize>() {
            Ok(n) => n,
            Err(_) => {
                log::warn!(
                    "Invalid MAX_NUMBER_OF_COMMITTEES value: '{}', falling back to 128",
                    val
                );
                128
            }
        },
        Err(_) => 128,
    };
    let service =
        TonicApplicationService::new(channel, evidence, /* logger= */ None, move || {
            CommitteeSelectorActor::new_with_reference_values(
                reference_values.clone(),
                max_committees,
            )
        });

    orchestrator_client.notify_app_ready().await.context("failed to notify that app is ready")?;
    tonic::transport::Server::builder()
        .add_service(EndpointServiceServer::new(service))
        .serve("[::]:8080".parse()?)
        .await?;
    Ok(())
}
