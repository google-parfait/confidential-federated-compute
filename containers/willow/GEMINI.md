## General Instructions

- Run tests frequently to catch syntax, compilation, and logic bugs early.
- Always change directory to the `containers/willow` sub-project directory before doing development work:
    ```bash
    cd containers/willow
    ```
- If running tests fails because dependencies are not installed (e.g. bazel) the issue is likely because the command is not being run inside the devcontainer, a docker container which has all required dependencies already installed.
- The devcontainer is defined in `.devcontainer/devcontainer.json` relative to the `containers/willow` directory, and all devcontainer commands should be run from within `containers/willow` using the `--workspace-folder .` flag. For example:
    ```bash
    devcontainer up --workspace-folder .
    devcontainer exec --workspace-folder . /bin/bash
    devcontainer exec --workspace-folder . bazel test :all
    ```
- The devcontainer can be rebuilt with the following command, however always stop and check with the user before doing so because it should not be necessary during normal development:
    ```bash
    devcontainer up --workspace-folder . --remove-existing-container
    ```
- If the `devcontainer` command is missing, suggest the user run the following command to install it. Do not run the command for the user:
    ```bash
    curl -fsSL https://raw.githubusercontent.com/devcontainers/cli/main/scripts/install.sh | sh
    ```
- When you run commands or start a shell inside this devcontainer, you are already at the root of the `willow` Bazel workspace, so you do not need to change directories before building or running tests.
- To format Rust code in the `willow` container project using `@rules_rust//:rustfmt` (which is configured to use the local `.rustfmt.toml` by default), run the following command from within the `containers/willow` directory:
    ```bash
    devcontainer exec --workspace-folder . bazel run @rules_rust//:rustfmt -- //...
    ```
    Alternatively, you can scope it to specific targets, such as:
    ```bash
    devcontainer exec --workspace-folder . bazel run @rules_rust//:rustfmt -- //committee_selector/...:all
    ```
- To format/lint Bazel files (`BUILD`, `WORKSPACE`, `*.bzl`, `MODULE.bazel`), use the `buildifier` tool inside the devcontainer:
    ```bash
    # To format all Bazel files recursively:
    devcontainer exec --workspace-folder . buildifier -r .

    # To check if formatting matches style guide (lint):
    devcontainer exec --workspace-folder . buildifier -mode=check -lint=warn -r .
    ```
- To sort keep-sorted blocks (like in `MODULE.bazel` or `BUILD` files), use the `keep-sorted` tool inside the devcontainer:
    ```bash
    # To sort a specific file:
    devcontainer exec --workspace-folder . keep-sorted MODULE.bazel

    # To sort all files recursively:
    devcontainer exec --workspace-folder . find . -type f -not -path '*/.*' -exec keep-sorted {} +

    # To check if all keep-sorted blocks are sorted recursively (lint):
    devcontainer exec --workspace-folder . find . -type f -not -path '*/.*' -exec keep-sorted --mode=lint {} +
    ```

## Reading Test Logs

- Test logs are also inside the devcontainer's filesystem. If you see a path that looks like `/home/vscode/.cache/bazel/_bazel_vscode/8524201230cf289da6a8f50b894245cf/execroot/_main/bazel-out/k8-opt/testlogs/.../*.log` then it must be read from the devcontainer using a command like:
  ```bash
  devcontainer exec --workspace-folder . cat <path_to_log_file>
  ```


