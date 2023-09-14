# Python script for use in scripts/export_container_bundle.sh to replace
# the process.args field in a json object read from stdin with a json object
# provided as a string to the --command argument, and output the result to
# stdout.
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
json.dump(json_obj, sys.stdout)
