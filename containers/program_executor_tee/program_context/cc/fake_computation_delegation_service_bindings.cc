#include <grpcpp/grpcpp.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "containers/program_executor_tee/program_context/cc/fake_computation_delegation_service.h"
#include "fcp/protos/confidentialcompute/computation_delegation.grpc.pb.h"
#include "fcp/protos/confidentialcompute/computation_delegation.pb.h"
#include "grpcpp/server_context.h"
#include "pybind11_protobuf/native_proto_caster.h"

namespace confidential_federated_compute::program_executor_tee {

namespace py = pybind11;

struct GrpcInitializer {
  GrpcInitializer() {
    std::cout << "Initializing gRPC in fake service bindings..." << std::endl;
    grpc_init();
  }
  ~GrpcInitializer() {
    std::cout << "Shutting down gRPC in fake service bindings..." << std::endl;
    grpc_shutdown();
  }
};

class FakeServer {
 public:
  FakeServer(int port, std::vector<std::string> worker_bns)
      : server_address_("[::1]:" + std::to_string(port)) {
    service_ = std::make_unique<FakeComputationDelegationService>(
        std::move(worker_bns));
  }

  void Start() {
    if (server_) {
      throw std::runtime_error("Server is already running.");
    }
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address_,
                             grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    if (!server_) {
      throw std::runtime_error("Could not start server on " + server_address_);
    }
    std::cout << "Server listening on: " << server_address_ << std::endl;
  }

  void Stop() {
    if (!server_) {
      return;  // It's often better to make Stop idempotent.
    }
    server_->Shutdown();
    std::cout << "Server stopped" << std::endl;
  }

  const std::string& GetAddress() const { return server_address_; }

  // The destructor will ensure Stop() is called.
  ~FakeServer() {
    if (server_) {
      Stop();
    }
  }

 private:
  std::string server_address_;
  std::unique_ptr<FakeComputationDelegationService> service_;
  std::unique_ptr<grpc::Server> server_;
};

PYBIND11_MODULE(fake_computation_delegation_service_bindings, m) {
  // Enables automatic conversions between Python and C++ protobuf messages.
  pybind11_protobuf::ImportNativeProtoCasters();

  static GrpcInitializer grpc_initializer;

  m.doc() =
      "Python bindings for the FakeComputationDelegationService, primarily for "
      "testing.";

  py::class_<FakeComputationDelegationService>(
      m, "FakeComputationDelegationService")
      .def(py::init<std::vector<std::string>>())
      .def(
          "Execute",
          [](FakeComputationDelegationService& self,
             const fcp::confidentialcompute::outgoing::ComputationRequest&
                 request)
              -> fcp::confidentialcompute::outgoing::ComputationResponse {
            grpc::ServerContext context;
            fcp::confidentialcompute::outgoing::ComputationResponse response;
            grpc::Status status = self.Execute(&context, &request, &response);
            if (!status.ok()) {
              throw std::runtime_error("gRPC call failed in Execute: " +
                                       status.error_message());
            }
            return response;
          },
          // Add documentation for the Python method.
          "Simulates a call to the Execute gRPC method.");

  py::class_<FakeServer>(m, "FakeServer")
      .def(py::init<int, std::vector<std::string>>(), py::arg("port"),
           py::arg("worker_bns"))
      .def("start", &FakeServer::Start, "Starts the gRPC server.")
      .def("stop", &FakeServer::Stop, "Stops the gRPC server.")
      .def("get_address", &FakeServer::GetAddress, "Gets the server address.");
}

}  // namespace confidential_federated_compute::program_executor_tee