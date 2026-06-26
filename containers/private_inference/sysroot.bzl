# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Repository rule for merging Debian packages into the sysroot."""

def _merged_sysroot_impl(repository_ctx):
    # 1. Download and extract the original sysroot
    repository_ctx.download_and_extract(
        url = "https://github.com/google-parfait/confidential-federated-compute/releases/download/sysroot-20260311/sysroot.tar.xz",
        sha256 = "86ccf9ea2360237ecad01633a86d566ed16e137daf986320ef91ec295ea5f679",
    )

    # Create a temp directory to extract all debs
    repository_ctx.execute(["mkdir", "temp_extract"])

    # Ensure target directory exists in sysroot
    repository_ctx.execute(["mkdir", "-p", "usr/lib/x86_64-linux-gnu"])

    # Loop over all deb files
    for deb_label in repository_ctx.attr.deb_files:
        deb_path = repository_ctx.path(deb_label)

        # Use a unique subdirectory per package to avoid conflicts during extraction
        # We replace special characters in the label name to make a valid directory name
        pkg_dir_name = deb_label.package.replace("/", "_").replace(":", "_") + "_" + deb_label.name
        pkg_dir = "temp_extract/" + pkg_dir_name
        repository_ctx.execute(["mkdir", "-p", pkg_dir])

        # Extract this deb into its pkg_dir
        res = repository_ctx.execute(["ar", "x", deb_path], quiet = True, working_directory = pkg_dir)
        if res.return_code != 0:
            fail("Failed to extract .deb archive %s: %s" % (deb_label, res.stderr))

        data_tar = ""
        for ext in ["xz", "zst", "gz"]:
            tar_name = "data.tar." + ext
            if repository_ctx.path(pkg_dir + "/" + tar_name).exists:
                data_tar = tar_name
                break

        if data_tar == "":
            fail("Could not find data.tar in .deb package %s" % deb_label)

        res = repository_ctx.execute(["tar", "xf", data_tar], quiet = True, working_directory = pkg_dir)
        if res.return_code != 0:
            fail("Failed to extract data.tar for %s: %s" % (deb_label, res.stderr))

        # Copy all shared libraries (.so*) from the extracted contents into the sysroot
        extracted_lib_path = pkg_dir + "/usr/lib/x86_64-linux-gnu"
        if repository_ctx.path(extracted_lib_path).exists:
            res = repository_ctx.execute([
                "bash",
                "-c",
                "cp -d %s/*.so* usr/lib/x86_64-linux-gnu/" % extracted_lib_path,
            ])
            if res.return_code != 0:
                fail("Failed to copy libraries for %s: %s" % (deb_label, res.stderr))

        extracted_lib_path2 = pkg_dir + "/lib/x86_64-linux-gnu"
        if repository_ctx.path(extracted_lib_path2).exists:
            res = repository_ctx.execute([
                "bash",
                "-c",
                "cp -d %s/*.so* usr/lib/x86_64-linux-gnu/" % extracted_lib_path2,
            ])
            if res.return_code != 0:
                fail("Failed to copy libraries from /lib for %s: %s" % (deb_label, res.stderr))

    # Clean up temp_extract
    repository_ctx.execute(["rm", "-rf", "temp_extract"])

merged_sysroot = repository_rule(
    implementation = _merged_sysroot_impl,
    attrs = {
        "deb_files": attr.label_list(mandatory = True, allow_files = True),
    },
)
