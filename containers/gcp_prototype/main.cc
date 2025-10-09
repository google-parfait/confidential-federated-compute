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

#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <chrono> // Required for std::chrono
#include <arpa/inet.h> // Required for inet_addr

const int PORT = 8000;
const int BUFFER_SIZE = 1024;

// --- Helper Function to get Current Time ---
std::string get_current_time_utc() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream time_ss;
    time_ss << std::put_time(std::gmtime(&in_time_t), "%Y-%m-%d %H:%M:%S UTC");
    return time_ss.str();
}

int main() {
    // --- Create Socket ---
    // AF_INET for IPv4, SOCK_STREAM for TCP
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        std::cerr << "Failed to create socket" << std::endl;
        return 1;
    }

    // --- Prepare the sockaddr_in structure ---
    sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    // THIS IS THE CRITICAL CHANGE: Bind to 0.0.0.0 instead of the default.
    // This ensures the server accepts connections from outside the container.
    server_addr.sin_addr.s_addr = inet_addr("0.0.0.0");
    server_addr.sin_port = htons(PORT);

    // --- Bind the socket to the address and port ---
    if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Bind failed" << std::endl;
        close(server_fd);
        return 1;
    }

    // --- Listen for incoming connections ---
    if (listen(server_fd, 3) < 0) {
        std::cerr << "Listen failed" << std::endl;
        close(server_fd);
        return 1;
    }

    std::cout << "Server starting: Listening on port " << PORT << "..." << std::endl;

    // --- Main Accept Loop ---
    while (true) {
        sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int new_socket = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);

        if (new_socket < 0) {
            std::cerr << "Accept failed" << std::endl;
            continue; // Continue to the next iteration
        }

        // --- Read Request (Simplified) ---
        char buffer[BUFFER_SIZE] = {0};
        read(new_socket, buffer, BUFFER_SIZE);
        std::string request(buffer);

        // --- Construct HTTP Response ---
        std::string time_str = get_current_time_utc();
        std::string body = "Hellooo from Confidential Space! The time is: " + time_str + "\n";
        
        std::stringstream http_response;
        http_response << "HTTP/1.1 200 OK\r\n";
        http_response << "Content-Type: text/plain\r\n";
        http_response << "Content-Length: " << body.length() << "\r\n";
        http_response << "\r\n";
        http_response << body;
        
        // --- Send Response ---
        std::string response_str = http_response.str();
        send(new_socket, response_str.c_str(), response_str.length(), 0);
        
        // --- Close the client socket ---
        close(new_socket);
    }

    // --- Close the server socket (unreachable in this loop) ---
    close(server_fd);
    return 0;
}
