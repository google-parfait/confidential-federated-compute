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

"""Traces a container digest to its source commit using GitHub Attestations.

This script queries the GitHub Attestations API for a given container digest,
uses the sigstore CLI to cryptographically verify the SLSA provenance bundle
(against Fulcio and Rekor), and extracts the source commit and GitHub Actions
run ID that produced it.

Setup:
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Usage:
    python3 trace_digest.py <sha256_digest>
"""

import argparse
import sys
import provenance_lib



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trace a container digest to its source commit via GitHub Attestations.",
        epilog='''example:\n  python3 trace_digest.py 2dce970207711ffeb036de533243295752d04862687210aabe77c0decdc57d56''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("digest", type=str, help="The hex SHA256 digest of the container (client or server).")

    args = parser.parse_args()

    subject_name, subject_digest, commits, custom_metadata, workflows = provenance_lib.fetch_and_verify(args.digest)
    provenance_lib.success(commits, custom_metadata, workflows)
