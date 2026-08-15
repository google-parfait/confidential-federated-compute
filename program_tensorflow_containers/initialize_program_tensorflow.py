import compilers
import federated_language
from program_executor_tee.program_context import execution_context
import tensorflow_federated as tff

TENSORFLOW_COMPUTATION_RUNNER_BINARY_PATH = (
    "computation_runner_binary_tensorflow"
)


def get_program_initialize_fn(
    outgoing_server_address: str,
    worker_bns: list[str] = [],
    serialized_reference_values: bytes = b"",
    max_concurrent_computation_calls=-1,
    use_elastic_composing_executor: bool = False,
    use_mergeable_execution_context: bool = False,
):

  def initialize():
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            compilers.compile_tf_to_call_dominant,
            tff.backends.native.compile_to_mergeable_comp_form
            if use_mergeable_execution_context
            else None,
            TENSORFLOW_COMPUTATION_RUNNER_BINARY_PATH,
            outgoing_server_address,
            worker_bns,
            serialized_reference_values,
            max_concurrent_computation_calls,
            use_elastic_composing_executor,
        )
    )

  return initialize
