// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
// Function to handle a single client connection
void handle_client(int client_socket) {
  char buffer[1024] = {0};
  read(client_socket, buffer, 1024);
  std::string request(buffer);
  std::string response_body;
  std::string status_line = "HTTP/1.1 200 OK\r\n";
  std::string content_type = "Content-Type: text/plain\r\n";
  // Rudimentary parsing of the HTTP request to get method and path
  std::string method = request.substr(0, request.find(" "));
  std::string path = request.substr(request.find(" ") + 1);
  path = path.substr(0, path.find(" "));
  if (method == "GET" && path == "/") {
    // Get the current time in UTC
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream time_ss;
    time_ss << std::put_time(std::gmtime(&in_time_t), "%Y-%m-%d %H:%M:%S UTC");
    response_body = "Hello from your tiny C++ server!\nThe current time is: " +
                    time_ss.str() + "\n";
  } else {
    status_line = "HTTP/1.1 404 Not Found\r\n";
    response_body = "404 Not Found\n";
  }
  std::stringstream response_ss;
  response_ss << status_line << content_type
              << "Content-Length: " << response_body.length() << "\r\n"
              << "\r\n"
              << response_body;
  std::string response = response_ss.str();
  send(client_socket, response.c_str(), response.length(), 0);
  close(client_socket);
}
int main() {
  int server_fd;
  struct sockaddr_in address;
  int opt = 1;
  int addrlen = sizeof(address);
  int port = 8000;
  std::cout << "Server starting: Listening on port " << port << "..."
            << std::endl;
  // Creating socket file descriptor
  if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
    perror("socket failed");
    exit(EXIT_FAILURE);
  }
  // Forcefully attaching socket to the port
  if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                 sizeof(opt))) {
    perror("setsockopt");
    exit(EXIT_FAILURE);
  }
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port);
  // Bind the socket to the network address and port
  if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
    perror("bind failed");
    exit(EXIT_FAILURE);
  }
  // Start listening for connections
  if (listen(server_fd, 3) < 0) {
    perror("listen");
    exit(EXIT_FAILURE);
  }
  // Main loop to accept and handle connections
  while (true) {
    int client_socket;
    if ((client_socket = accept(server_fd, (struct sockaddr*)&address,
                                (socklen_t*)&addrlen)) < 0) {
      perror("accept");
      continue;  // Continue to the next iteration on accept error
    }
    handle_client(client_socket);
  }
  return 0;
}
