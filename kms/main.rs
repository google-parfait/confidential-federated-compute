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

use std::sync::Arc;

use anyhow::{bail, Context};
use key_management_service::{get_init_request, KeyManagementService};
use kms_proto::fcp::confidentialcompute::key_management_service_server::KeyManagementServiceServer;
use oak_proto_rust::oak::attestation::v1::{Evidence, ReferenceValues, TeePlatform};
use oak_sdk_common::{StaticAttester, StaticEndorser};
use oak_sdk_containers::{
    init_metrics, InstanceSessionBinder, InstanceSigner, MetricsConfig, OrchestratorClient,
};
use oak_time_std::clock::SystemTimeClock;
use opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::logs::LoggerProvider;
use prost::Message;
use session_v1_service_proto::oak::services::oak_session_v1_service_client::OakSessionV1ServiceClient;
use slog::Drain;
use storage_actor::StorageActor;
use storage_client::GrpcStorageClient;
use tcp_proto::runtime::endpoint::endpoint_service_server::EndpointServiceServer;
use tcp_runtime::service::TonicApplicationService;
use tracing_subscriber::prelude::*;

/// The address for OpenTelemetry logging, which is the Oak Launcher address:
/// https://github.com/project-oak/oak/blob/ac3a692abe67331d7c62e75523c7385e86adf29b/oak_containers/orchestrator/src/lib.rs#L40.
const OPEN_TELEMETRY_ADDR: &str = "http://10.0.2.100:8080";

/// The OakSessionV1Service address. This uses the same host as the Oak Launcher
/// but the host forwarding rules specify a different port.
const OAK_SESSION_SERVICE_ADDR: &str = "http://10.0.2.100:8008";

fn get_reference_values(evidence: &Evidence) -> anyhow::Result<ReferenceValues> {
    match evidence.root_layer.as_ref().map(|rl| rl.platform.try_into()) {
        Some(Ok(TeePlatform::AmdSevSnp)) => {
            ReferenceValues::decode(include_bytes!(env!("REFERENCE_VALUES")).as_slice())
                .context("failed to decode ReferenceValues")
        }
        Some(Ok(TeePlatform::None)) => {
            // When running in insecure mode, simply skip all reference values.
            // This is only used for tests.
            ReferenceValues::decode(include_bytes!(env!("INSECURE_REFERENCE_VALUES")).as_slice())
                .context("failed to decode insecure ReferenceValues")
        }
        platform => bail!("platform {:?} is not supported", platform),
    }
}

fn initialize_logging() {
    opentelemetry::global::set_error_handler(|err| eprintln!("KMS: OTLP error: {}", err))
        .expect("failed to set OTLP error handler");
    let log_exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint(OPEN_TELEMETRY_ADDR)
        .build_log_exporter()
        .expect("failed to create LogExporter");
    let logger_provider = LoggerProvider::builder()
        .with_batch_exporter(log_exporter, opentelemetry_sdk::runtime::Tokio)
        .build();
    tracing_subscriber::registry()
        .with(OpenTelemetryTracingBridge::new(&logger_provider))
        .with(tracing_subscriber::filter::LevelFilter::INFO)
        .init();

    // Update the panic hook to flush logs before exiting.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let result = logger_provider.force_flush();
        if result.iter().any(|r| r.is_err()) {
            eprintln!("Failed to flush OTLP logs: {:?}", result);
        }
        prev_hook(info);
    }));
}

#[tokio::main]
async fn main() {
    initialize_logging();
    tracing::info!("KMS starting...");

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
    let session_binder = Arc::new(InstanceSessionBinder::create(&channel));
    let signer = InstanceSigner::create(&channel);
    let reference_values = get_reference_values(&evidence).expect("failed to get reference values");
    let clock = Arc::new(SystemTimeClock {});

    // Export basic metrics to OpenTelemetry (e.g. CPU usage and RPC latency).
    let oak_observer = init_metrics(MetricsConfig {
        launcher_addr: OPEN_TELEMETRY_ADDR.into(),
        scope: "kms",
        excluded_metrics: None,
    });

    // Create the KeyManagementService.
    let session_service_client = OakSessionV1ServiceClient::connect(OAK_SESSION_SERVICE_ADDR)
        .await
        .expect("failed to create OakSessionV1ServiceClient");
    let key_management_service = KeyManagementService::new(
        GrpcStorageClient::new(
            session_service_client,
            get_init_request,
            attester.clone(),
            endorser.clone(),
            session_binder.clone(),
            reference_values.clone(),
            clock.clone(),
        ),
        signer,
    );

    // Create the TCP EndpointService, following the TCP recommendation of only
    // logging WARNING and above.
    let logger = slog::Logger::root(
        tracing_slog::TracingSlogDrain.filter_level(slog::Level::Warning).fuse(),
        slog::o!(),
    );
    let endpoint_service =
        TonicApplicationService::new(channel, evidence, Some(logger), move || {
            StorageActor::new(attester, endorser, session_binder, reference_values, clock)
        });

    // Start the gRPC server.
    orchestrator_client.notify_app_ready().await.expect("failed to notify that app is ready");
    tonic::transport::Server::builder()
        .layer(oak_observer.create_monitoring_layer())
        .add_service(KeyManagementServiceServer::new(key_management_service))
        .add_service(EndpointServiceServer::new(endpoint_service))
        .serve("[::]:8080".parse().expect("failed to parse address"))
        .await
        .expect("failed to start server");
}
