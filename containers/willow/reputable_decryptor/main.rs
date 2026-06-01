// Copyright 2026 The Trusted Computations Platform Authors.
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

use anyhow::{anyhow, Context};
use oak_proto_rust::oak::attestation::v1::{
    binary_reference_value, kernel_binary_reference_value, reference_values, text_reference_value,
    BinaryReferenceValue, ContainerLayerReferenceValues, InsecureReferenceValues,
    KernelBinaryReferenceValue, KernelLayerReferenceValues, OakContainersReferenceValues,
    ReferenceValues, RootLayerReferenceValues, SkipVerification, SystemLayerReferenceValues,
    TextReferenceValue,
};
use oak_sdk_containers::OrchestratorClient;
use tcp_proto::runtime::endpoint::endpoint_service_server::EndpointServiceServer;
use tcp_runtime::service::TonicApplicationService;
use willow_reputable_decryptor_service::actor::ReputableDecryptorActor;

fn get_reference_values() -> ReferenceValues {
    let skip = BinaryReferenceValue {
        r#type: Some(binary_reference_value::Type::Skip(SkipVerification::default())),
    };
    ReferenceValues {
        r#type: Some(reference_values::Type::OakContainers(OakContainersReferenceValues {
            root_layer: Some(RootLayerReferenceValues {
                insecure: Some(InsecureReferenceValues::default()),
                ..Default::default()
            }),
            kernel_layer: Some(KernelLayerReferenceValues {
                kernel: Some(KernelBinaryReferenceValue {
                    r#type: Some(kernel_binary_reference_value::Type::Skip(
                        SkipVerification::default(),
                    )),
                }),
                kernel_cmd_line_text: Some(TextReferenceValue {
                    r#type: Some(text_reference_value::Type::Skip(SkipVerification::default())),
                }),
                init_ram_fs: Some(skip.clone()),
                memory_map: Some(skip.clone()),
                acpi: Some(skip.clone()),
                ..Default::default()
            }),
            system_layer: Some(SystemLayerReferenceValues { system_image: Some(skip.clone()) }),
            container_layer: Some(ContainerLayerReferenceValues {
                binary: Some(skip.clone()),
                configuration: Some(skip.clone()),
            }),
        })),
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

    let max_decryptors = match std::env::var("MAX_NUMBER_OF_DECRYPTORS") {
        Ok(val) => match val.parse::<usize>() {
            Ok(n) => n,
            Err(_) => {
                log::warn!(
                    "Invalid MAX_NUMBER_OF_DECRYPTORS value: '{}', falling back to 128",
                    val
                );
                128
            }
        },
        Err(_) => 128,
    };

    let max_keys = match std::env::var("MAX_NUMBER_OF_KEYS") {
        Ok(val) => match val.parse::<usize>() {
            Ok(n) => n,
            Err(_) => {
                log::warn!("Invalid MAX_NUMBER_OF_KEYS value: '{}', falling back to 100", val);
                100
            }
        },
        Err(_) => 100,
    };

    let service =
        TonicApplicationService::new(channel, evidence, /* logger= */ None, move || {
            ReputableDecryptorActor::new_with_reference_values(
                get_reference_values(),
                max_decryptors,
                max_keys,
            )
        });

    orchestrator_client.notify_app_ready().await.context("failed to notify that app is ready")?;
    tonic::transport::Server::builder()
        .add_service(EndpointServiceServer::new(service))
        .serve("[::]:8080".parse()?)
        .await?;
    Ok(())
}
