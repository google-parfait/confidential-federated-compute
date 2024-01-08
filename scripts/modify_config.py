# Copyright 2023 Google LLC.
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
#
#
# Python script for use in scripts/export_container_bundle.sh to modify a
# config.json object read from stdin to make it suitable for use with Oak
# Containers.
#
# Replaces the process.args field with a json object provided as a string to the
# --command argument.
#
# Also sets the root.readonly field to False which is necessary to allow python
# to write to temporary directories.
#
# Outputs the modified json object to stdout.
#
# Used to avoid an additional dependency on jq.
import argparse
import json
import sys

argparser = argparse.ArgumentParser()
# Add a string argument which should contain a JSON list which includes the
# portions of the command to run on the Docker container.
argparser.add_argument(
    'command',
    type=str,
    default='["sh"]',  # default if nothing is provided
)
args = argparser.parse_args()
# Parse both stdin and the command argument as JSON. Replace the process.args
# field from stdin with the JSON command.
command_list = json.loads(args.command)
json_obj = json.load(sys.stdin)
json_obj['process']['args'] = command_list
json_obj['root']['readonly'] = False
json.dump(json_obj, sys.stdout)
