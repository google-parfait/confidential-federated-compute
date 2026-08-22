import federated_language
from program_executor_tee.program_context import execution_context
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
    max_concurrent_computation_calls=-1,
    use_elastic_composing_executor: bool = False,
    use_mergeable_execution_context: bool = False,
    mergeable_execution_subrounds_multiplier: int = 0,
):
  if use_mergeable_execution_context:
    raise ValueError(
        "use_mergeable_execution_context is not supported in the JAX/XLA"
        " stack yet."
    )

  def initialize():
    num_subrounds = None
    if (
        mergeable_execution_subrounds_multiplier > 0
        and len(worker_bns) > 0
    ):
      num_subrounds = len(worker_bns) * mergeable_execution_subrounds_multiplier

    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            compiler_fn=compile_to_call_dominant,
            mergeable_comp_compiler_fn=None,
            computation_runner_binary_path=XLA_COMPUTATION_RUNNER_BINARY_PATH,
            outgoing_server_address=outgoing_server_address,
            worker_bns=worker_bns,
            serialized_reference_values=serialized_reference_values,
            max_concurrent_computation_calls=max_concurrent_computation_calls,
            use_elastic_composing_executor=use_elastic_composing_executor,
            num_subrounds=num_subrounds,
        )
    )

  return initialize
