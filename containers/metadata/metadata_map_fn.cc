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
#include "containers/metadata/metadata_map_fn.h"

#include "absl/container/flat_hash_map.h"
#include "absl/numeric/bits.h"
#include "absl/strings/str_cat.h"
#include "containers/big_endian.h"
#include "containers/fns/map_fn.h"
#include "fcp/base/digest.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/constants.h"
#include "fcp/confidentialcompute/time_window_utilities.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/tee_payload_metadata.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::metadata {
namespace {
using ::confidential_federated_compute::Session;
using ::confidential_federated_compute::fns::KeyValue;
using ::confidential_federated_compute::fns::WriteConfigurationMap;
using ::fcp::client::EventTimeRange;
using ::fcp::confidential_compute::kEventTimeColumnName;
using ::fcp::confidential_compute::kPrivacyIdColumnName;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::ConvertEventTimeToCivilSecond;
using ::fcp::confidentialcompute::EventTimeGranularity;
using ::fcp::confidentialcompute::MetadataConfig;
using ::fcp::confidentialcompute::MetadataContainerConfig;
using ::fcp::confidentialcompute::MetadataContainerInitializationConfig;
using ::fcp::confidentialcompute::PayloadMetadataSet;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::TeePayloadMetadata;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;

absl::StatusOr<std::string> GetPrivacyId(CheckpointParser& parser) {
  FCP_ASSIGN_OR_RETURN(Tensor privacy_id_tensor,
                       parser.GetTensor(kPrivacyIdColumnName));
  if (privacy_id_tensor.dtype() !=
      tensorflow_federated::aggregation::DataType::DT_STRING) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`%s` tensor must be a string tensor", kPrivacyIdColumnName));
  }
  if (!privacy_id_tensor.is_scalar()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("`%s` tensor must be a scalar", kPrivacyIdColumnName));
  }
  return std::string(privacy_id_tensor.AsScalar<absl::string_view>());
}

absl::StatusOr<uint64_t> GetUpper64HashedPrivacyId(CheckpointParser& parser) {
  FCP_ASSIGN_OR_RETURN(std::string privacy_id, GetPrivacyId(parser));
  return LoadBigEndian<uint64_t>(fcp::ComputeSHA256(privacy_id));
}

// Computes the partition key for the privacy ID held by the parser and
// configured by num_partitions.
absl::StatusOr<uint64_t> ComputePartitionKey(
    uint64_t upper_64_hashed_privacy_id,
    const MetadataConfig& metadata_config) {
  // Determine the number of bits to zero out based on the number of
  // partitions.
  uint64_t num_partitions = metadata_config.num_partitions();
  if (num_partitions == 0) {
    return absl::InvalidArgumentError("num_partitions cannot be 0.");
  }
  // Handle this case specially because shifting a uint64 by 64 is undefined
  // behavior.
  if (num_partitions == 1) {
    return 0;
  }
  // Calculate the number of bits required to represent all partition indices
  // (i.e., from 0 to num_partitions - 1). For example, with 10 partitions, we
  // need 4 bits to represent indices 0 through 9.
  uint64_t num_partition_bits = absl::bit_width(num_partitions - 1);
  uint64_t num_lower_zero_bits = 64 - num_partition_bits;

  // To partition the data, we use the most significant `num_partition_bits` of
  // the hashed privacy ID. This is achieved by zeroing out the lower `64 -
  // num_partition_bits`.
  uint64_t mask = ~0ULL << num_lower_zero_bits;
  return upper_64_hashed_privacy_id & mask;
}

google::type::DateTime ToDayGranularityDateTime(
    const absl::CivilSecond& civil_second) {
  google::type::DateTime date_time;
  date_time.set_year(civil_second.year());
  date_time.set_month(civil_second.month());
  date_time.set_day(civil_second.day());
  return date_time;
}

absl::StatusOr<EventTimeRange> GetCoarseEventTimeRange(
    absl::CivilSecond min_event_time, absl::CivilSecond max_event_time,
    EventTimeGranularity event_time_granularity) {
  switch (event_time_granularity) {
    case EventTimeGranularity::EVENT_TIME_GRANULARITY_DAY: {
      EventTimeRange event_time_range;
      *event_time_range.mutable_start_event_time() =
          ToDayGranularityDateTime(min_event_time);
      // EventTimeRange.end_event_time is exclusive, so add 1 day to make the
      // range inclusive of the max_event_time day.
      absl::CivilDay end_day = absl::CivilDay(max_event_time) + 1;
      *event_time_range.mutable_end_event_time() =
          ToDayGranularityDateTime(end_day);
      return event_time_range;
    }
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported event time granularity: %d", event_time_granularity));
  }
}

absl::StatusOr<Tensor> GetEventTime(CheckpointParser& parser,
                                    absl::string_view on_device_query_name) {
  // All checkpoints, including message-based ones, represent the event time as
  // a scalar string Tensor.
  FCP_ASSIGN_OR_RETURN(Tensor time_tensor,
                       parser.GetTensor(absl::StrCat(on_device_query_name, "/",
                                                     kEventTimeColumnName)));
  if (time_tensor.dtype() !=
      tensorflow_federated::aggregation::DataType::DT_STRING) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`%s` tensor must be a string tensor", kEventTimeColumnName));
  }
  if (time_tensor.shape().dim_sizes().size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`%s` tensor must have one dimension", kEventTimeColumnName));
  }
  return time_tensor;
}

struct MinMaxEventTimes {
  absl::CivilSecond min = absl::CivilSecond::max();
  absl::CivilSecond max = absl::CivilSecond::min();
};

absl::StatusOr<std::optional<MinMaxEventTimes>> GetMinMaxEventTimes(
    CheckpointParser& parser, absl::string_view on_device_query_name) {
  FCP_ASSIGN_OR_RETURN(Tensor time_tensor,
                       GetEventTime(parser, on_device_query_name));
  if (time_tensor.num_elements() == 0) {
    return std::nullopt;
  }
  MinMaxEventTimes min_max_event_times;
  // Iterate over the event times and find the min and max.
  for (auto event_time : time_tensor.AsSpan<absl::string_view>()) {
    FCP_ASSIGN_OR_RETURN(absl::CivilSecond civil_time,
                         ConvertEventTimeToCivilSecond(event_time));
    min_max_event_times.min = std::min(min_max_event_times.min, civil_time);
    min_max_event_times.max = std::max(min_max_event_times.max, civil_time);
  }
  return min_max_event_times;
}

// MapFn implementation that computes metadata for each input.
class MetadataMapFn final : public confidential_federated_compute::fns::MapFn {
 public:
  explicit MetadataMapFn(const MetadataContainerConfig& config,
                         std::string on_device_query_name)
      : config_(config),
        on_device_query_name_(std::move(on_device_query_name)) {}

  // Parses the unencrypted data, for each metadata config compute the
  // corresponding metadata.
  absl::StatusOr<KeyValue> Map(KeyValue input,
                               Session::Context& context) override {
    FederatedComputeCheckpointParserFactory parser_factory;
    absl::StatusOr<std::unique_ptr<CheckpointParser>> parser =
        parser_factory.Create(absl::Cord(std::move(input.value.data)));
    if (!parser.ok()) {
      return absl::Status(
          parser.status().code(),
          absl::StrCat("Failed to deserialize checkpoint: ", parser.status()));
    }

    FCP_ASSIGN_OR_RETURN(uint64_t upper_64_hashed_privacy_id,
                         GetUpper64HashedPrivacyId(**parser));
    FCP_ASSIGN_OR_RETURN(std::optional<MinMaxEventTimes> min_max_event_times,
                         GetMinMaxEventTimes(**parser, on_device_query_name_));

    PayloadMetadataSet payload_metadata_set;
    for (auto& [name, metadata_config] : config_.metadata_configs()) {
      FCP_ASSIGN_OR_RETURN(
          uint64_t partition_key,
          ComputePartitionKey(upper_64_hashed_privacy_id, metadata_config));
      TeePayloadMetadata tee_payload_metadata;
      tee_payload_metadata.set_partition_key(partition_key);

      if (min_max_event_times.has_value()) {
        FCP_ASSIGN_OR_RETURN(
            EventTimeRange event_time_range,
            GetCoarseEventTimeRange(
                min_max_event_times->min, min_max_event_times->max,
                metadata_config.event_time_range_granularity()));
        *tee_payload_metadata.mutable_event_time_range() =
            std::move(event_time_range);
      }

      payload_metadata_set.mutable_metadata()->insert(
          {name, std::move(tee_payload_metadata)});
    }

    KeyValue output;
    output.key.PackFrom(payload_metadata_set);
    return output;
  }

 private:
  const MetadataContainerConfig config_;
  const std::string on_device_query_name_;
};

// Factory for the MetadataMapFn. Thread-safe.
class MetadataMapFnFactory
    : public confidential_federated_compute::fns::FnFactory {
 public:
  explicit MetadataMapFnFactory(MetadataContainerConfig config,
                                std::string on_device_query_name)
      : config_(std::move(config)),
        on_device_query_name_(std::move(on_device_query_name)) {}

  absl::StatusOr<std::unique_ptr<fns::Fn>> CreateFn() const override {
    return std::make_unique<MetadataMapFn>(config_, on_device_query_name_);
  }

 private:
  const MetadataContainerConfig config_;
  const std::string on_device_query_name_;
};
}  // namespace

absl::StatusOr<std::unique_ptr<fns::FnFactory>> ProvideMetadataMapFnFactory(
    const google::protobuf::Any& configuration,
    const google::protobuf::Any& config_constraints,
    const WriteConfigurationMap& /*write_configuration_map*/) {
  // We expect to find the configuration that describes what metadata to
  // generate in the trusted config_constraints.
  MetadataContainerConfig config;
  if (!config_constraints.UnpackTo(&config)) {
    return absl::InvalidArgumentError(
        "Config constraints cannot be unpacked to MetadataContainerConfig.");
  }
  MetadataContainerInitializationConfig init_config;
  if (!configuration.UnpackTo(&init_config)) {
    return absl::InvalidArgumentError(
        "Configuration cannot be unpacked to "
        "MetadataContainerInitializationConfig.");
  }
  if (init_config.on_device_query_name().empty()) {
    return absl::InvalidArgumentError(
        "on_device_query_name must be set in the initialization config.");
  }
  return std::make_unique<MetadataMapFnFactory>(
      std::move(config), std::move(init_config.on_device_query_name()));
}

}  // namespace confidential_federated_compute::metadata