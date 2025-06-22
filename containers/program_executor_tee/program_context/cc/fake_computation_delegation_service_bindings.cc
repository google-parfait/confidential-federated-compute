#include <grpcpp/grpcpp.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "containers/program_executor_tee/program_context/cc/fake_computation_delegation_service.h"
#include "fcp/protos/confidentialcompute/computation_delegation.grpc.pb.h"
#include "fcp/protos/confidentialcompute/computation_delegation.pb.h"
#include "grpcpp/server_context.h"

namespace confidential_federated_compute::program_executor_tee {

namespace py = pybind11;

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
  }

  void Stop() {
    if (!server_) {
      return;  // It's often better to make Stop idempotent.
    }
    server_->Shutdown();
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
  m.doc() =
      "Python bindings for the FakeComputationDelegationService, primarily for "
      "testing.";

  py::class_<FakeServer>(m, "FakeServer")
      .def(py::init<int, std::vector<std::string>>(), py::arg("port"),
           py::arg("worker_bns"))
      .def("start", &FakeServer::Start, "Starts the gRPC server.")
      .def("stop", &FakeServer::Stop, "Stops the gRPC server.")
      .def("get_address", &FakeServer::GetAddress, "Gets the server address.");
}

}  // namespace confidential_federated_compute::program_executor_tee