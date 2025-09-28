from collections.abc import Callable

from fcp.protos.confidentialcompute import data_read_write_pb2
import federated_language
from program_executor_tee.program_context import execution_context
from tensorflow_federated.proto.v0 import executor_pb2
import tensorflow_federated as tff


XLA_COMPUTATION_RUNNER_BINARY_PATH = (
    "program_executor_tee/program_context/cc/computation_runner_binary_xla"
)

def compile_to_call_dominant(
      comp: federated_language.framework.ConcreteComputation,
  ) -> federated_language.framework.ConcreteComputation:
    """Compile a computation to run on the program executor TEE."""
    comp_bb = tff.framework.to_call_dominant(comp.to_building_block())
    return federated_language.framework.ConcreteComputation.from_building_block(
        comp_bb
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
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            compile_to_call_dominant,
            XLA_COMPUTATION_RUNNER_BINARY_PATH,
            outgoing_server_address,
            worker_bns,
            serialized_reference_values,
            parse_read_response_fn,
        )
    )

  return initialize
