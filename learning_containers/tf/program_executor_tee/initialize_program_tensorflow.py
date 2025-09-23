from collections.abc import Callable
import functools

from fcp.confidentialcompute.python import compiler
from fcp.protos.confidentialcompute import data_read_write_pb2
import federated_language
from program_executor_tee.program_context import execution_context
from tensorflow_federated.proto.v0 import executor_pb2
from tf.program_executor_tee import compilers


TENSORFLOW_COMPUTATION_RUNNER_BINARY_PATH = (
    "tf/program_executor_tee/computation_runner_binary_tensorflow"
)


def get_program_initialize_fn(
    outgoing_server_address: str,
    worker_bns: list[str] = [],
    serialized_reference_values: bytes = b"",
    parse_read_response_fn: Callable[
        [data_read_write_pb2.ReadResponse, str, str], executor_pb2.Value
    ] = None,
):
  def initialize():
    if worker_bns:
      compiler_fn = functools.partial(
          compiler.to_composed_tee_form,
          num_client_workers=len(worker_bns) - 1,
      )
    else:
      compiler_fn = compilers.compile_tf_to_call_dominant
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            compiler_fn,
            TENSORFLOW_COMPUTATION_RUNNER_BINARY_PATH,
            outgoing_server_address,
            worker_bns,
            serialized_reference_values,
            parse_read_response_fn,
        )
    )

  return initialize
