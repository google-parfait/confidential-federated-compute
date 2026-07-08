# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json

from google.protobuf import json_format
from google.protobuf import text_format

import attestation_policy_pb2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_file", required=True)
    parser.add_argument("--alts_enabled", required=True)
    parser.add_argument("--jwks_endpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Load and parse the textproto
    policy = attestation_policy_pb2.AttestationPolicy()
    with open(args.policy_file) as f:
        text_format.Parse(f.read(), policy)

    # Convert to JSON dictionary
    policy_dict = json_format.MessageToDict(
        policy,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True)

    metadata = {
        "attestation_policy": policy_dict,
        "alts_enabled": args.alts_enabled.lower() == 'true',
        "jwks_endpoint": args.jwks_endpoint
    }

    with open(args.out, "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
