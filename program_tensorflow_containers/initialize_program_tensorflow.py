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
    mergeable_execution_subrounds_multiplier: int = 0,
):

  def initialize():
    num_subrounds = None
    if (
        mergeable_execution_subrounds_multiplier > 0
        and len(worker_bns) > 0
    ):
      num_subrounds = len(worker_bns) * mergeable_execution_subrounds_multiplier

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
            num_subrounds=num_subrounds,
        )
    )

  return initialize
