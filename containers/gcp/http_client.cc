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

#include "http_client.h"

#include <curl/curl.h>  // Include curl here

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace confidential_federated_compute::gcp {
namespace {
// This callback is used by both GET and POST
static size_t WriteCallback(void* contents, size_t size, size_t nmemb,
                            void* userp) {
  ((std::string*)userp)->append((char*)contents, size * nmemb);
  return size * nmemb;
}
}  // namespace

absl::StatusOr<std::string> PostJsonViaUnixSocket(
    absl::string_view url, absl::string_view socket_path,
    absl::string_view json_payload) {
  CURL* curl = curl_easy_init();
  if (!curl) {
    return absl::InternalError("Failed to initialize curl.");
  }
  std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl_handle(
      curl, curl_easy_cleanup);

  std::string read_buffer;
  struct curl_slist* headers = nullptr;
  std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> header_list_guard(
      headers, curl_slist_free_all);

  headers = curl_slist_append(headers, "Content-Type: application/json");
  headers = curl_slist_append(headers, "Metadata-Flavor: Google");
  header_list_guard.reset(headers);  // Reset guard after potential reallocation

  curl_easy_setopt(curl, CURLOPT_URL, std::string(url).c_str());
  curl_easy_setopt(curl, CURLOPT_UNIX_SOCKET_PATH,
                   std::string(socket_path).c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list_guard.get());
  // Use length-aware option for POST data
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_payload.length());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.data());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &read_buffer);

  CURLcode res = curl_easy_perform(curl);
  long http_code = 0;
  if (res == CURLE_OK) {
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  } else {
    return absl::InternalError(
        absl::StrCat("curl_easy_perform() failed: ", curl_easy_strerror(res)));
  }

  if (http_code != 200) {
    absl::StatusCode status_code = absl::StatusCode::kInternal;
    if (http_code >= 400 && http_code < 500) {
      status_code = absl::StatusCode::kInvalidArgument;
    } else if (http_code >= 500) {
      status_code = absl::StatusCode::kUnavailable;
    }
    return absl::Status(status_code,
                        absl::StrCat("Attestation agent returned HTTP status ",
                                     http_code, ": ", read_buffer));
  }
  return read_buffer;
}

absl::StatusOr<std::string> CurlGet(absl::string_view url) {
  CURL* curl = curl_easy_init();
  if (!curl) {
    return absl::InternalError("Failed to initialize curl.");
  }
  std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl_handle(
      curl, curl_easy_cleanup);
  std::string read_buffer;

  curl_easy_setopt(curl, CURLOPT_URL, std::string(url).c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &read_buffer);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "gcp-client/1.0");

  CURLcode res = curl_easy_perform(curl);
  long http_code = 0;
  if (res == CURLE_OK) {
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  } else {
    return absl::InternalError(
        absl::StrCat("curl_easy_perform() failed: ", curl_easy_strerror(res)));
  }

  if (http_code != 200) {
    return absl::InternalError(absl::StrCat("HTTP GET failed with code ",
                                            http_code, ": ", read_buffer));
  }
  return read_buffer;
}

}  // namespace confidential_federated_compute::gcp
