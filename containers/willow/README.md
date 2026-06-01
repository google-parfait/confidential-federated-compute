# Willow server

This folder contains a Willow specific containers that
implements secure aggregation using Willow protocol - see
https://eprint.iacr.org/2024/936.pdf for the details.

The actual aggregation is handled in the Willow crypto library:
https://github.com/google/secure-aggregation.

## Development Setup

This project uses a Dev Container for a reproducible and pre-configured environment.

### Starting the Dev Container
Always navigate to the sub-project directory first:
```bash
cd containers/willow
```

Then run the devcontainer commands:
```bash
# Start the container
devcontainer up --workspace-folder .

# Open a shell inside the container
devcontainer exec --workspace-folder . /bin/bash

# Run all tests inside the container
devcontainer exec --workspace-folder . bazel test :all
```

For more detailed instructions, developer guidelines, formatting code, and viewing logs inside the devcontainer, refer to [GEMINI.md](GEMINI.md).
