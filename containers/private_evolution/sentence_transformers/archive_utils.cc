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
#include "archive_utils.h"

#include <iostream>
#include <string>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "fcp/base/monitoring.h"
#include "libarchive/archive.h"
#include "libarchive/archive_entry.h"

namespace confidential_federated_compute::sentence_transformers {
namespace {

absl::Status CopyData(struct archive* ar, struct archive* aw) {
  int r;
  const void* buff;
  size_t size;
  la_int64_t offset;

  for (;;) {
    r = archive_read_data_block(ar, &buff, &size, &offset);
    if (r == ARCHIVE_EOF) return absl::OkStatus();
    if (r < ARCHIVE_OK) return absl::InternalError(archive_error_string(ar));
    r = archive_write_data_block(aw, buff, size, offset);
    if (r < ARCHIVE_OK) {
      return absl::InternalError(archive_error_string(aw));
    }
  }
}

}  // anonymous namespace

absl::StatusOr<std::string> ExtractAll(absl::string_view zip_file_path,
                                       absl::string_view parent) {
  // 1. Setup archive reader
  struct archive* a = archive_read_new();
  archive_read_support_filter_all(a);
  archive_read_support_format_all(a);

  // 2. Setup archive writer (disk writer)
  struct archive* ext = archive_write_disk_new();
  archive_write_disk_set_standard_lookup(ext);

  // Setup cleanup
  auto archive_closer = absl::MakeCleanup([a, ext] {
    archive_read_close(a);
    archive_read_free(a);
    archive_write_close(ext);
    archive_write_free(ext);
  });

  std::string file_name = std::string(zip_file_path);
  int r = archive_read_open_filename(a, file_name.data(), 10240);
  if (r != ARCHIVE_OK) {
    return absl::InvalidArgumentError(archive_error_string(a));
  }

  // 3. Iterate through entries and extract
  std::string first_entry_path;
  bool is_first_entry = true;

  std::string dest_dir = std::string(parent);

  struct archive_entry* entry;
  while (archive_read_next_header(a, &entry) == ARCHIVE_OK) {
    // Construct the full destination path
    const char* pathname_in_archive = archive_entry_pathname(entry);

    std::string full_path = absl::StrCat(dest_dir, "/", pathname_in_archive);
    LOG(INFO) << absl::StrCat("Writing into path:", full_path);

    // Rewrite the entry's pathname to the new location
    archive_entry_set_pathname(entry, full_path.data());

    // Write the header and data to disk
    r = archive_write_header(ext, entry);
    if (r == ARCHIVE_OK) {
      FCP_RETURN_IF_ERROR(CopyData(a, ext));
      r = archive_write_finish_entry(ext);
    } else {
      return absl::InvalidArgumentError(archive_error_string(ext));
    }

    if (is_first_entry) {
      first_entry_path = std::string(full_path);
      first_entry_path = absl::StripSuffix(first_entry_path, "/");
      is_first_entry = false;
    }
  }

  return first_entry_path;
}

}  // namespace confidential_federated_compute::sentence_transformers
