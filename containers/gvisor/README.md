# gVisor container (experimental)

> [!WARNING]
> This container is experimental. It's a prototype for discussion and demos,
> not ready for production use. See [Status and limitations](#status-and-limitations).

An Oak Containers application that runs untrusted, closed-source code once per
record, each time in a fresh [gVisor](https://gvisor.dev) sandbox. The code can
be any Linux root filesystem, for example a Python application with native
dependencies. This container, the enforcer, is open source and can be
attested. The customer code it runs doesn't need to be.

## How it works

1. **Setup.** The customer's root filesystem arrives as a tar in the
   `rootfs_layer` configuration blob (`WriteConfiguration`). The container
   unpacks it once with `bsdtar`. It then writes the sandbox's OCI config
   itself, so the customer only supplies files. The sandbox runs
   `/entrypoint` from that filesystem, which can be any executable, or a
   symlink to one. It gets:
   - a read-only root;
   - a tmpfs `/tmp`;
   - no network.
2. **Per record.** Each input is a Federated Compute checkpoint whose string
   tensor `data` holds records. `DoFn::Do` runs the program once per record
   with `runsc run`, each time in a fresh sandbox. It uses the systrap platform
   because a confidential VM has no KVM. The record goes to the program's
   stdin.
3. **Output.** Each input produces one checkpoint, whose string tensor `data`
   holds the stdout of every run that succeeded. A run that fails, times out,
   or writes more than 1 MiB contributes nothing. The output is encrypted with
   the first reencryption key, so only the pipeline's next node can read it.

The container never parses the records or the outputs. What the bytes mean is
up to the customer's program and the stages around this one. For example,
FedSQL can read the output as a table with a single `data` column.

## Privileges

Oak runs app containers with rootless runc as an unprivileged user. Each
container lives in a user namespace that maps a single uid.

runsc needs two capabilities on top of Oak's defaults (`CAP_AUDIT_WRITE`,
`CAP_KILL`, `CAP_NET_BIND_SERVICE`):

- **`CAP_SYS_ADMIN`:** to create the sandbox's namespaces and its minimal
  chroot.
- **`CAP_SETFCAP`:** to map uid 0 into the nested user namespace it creates.
  Linux 5.12 and later require this.

How they're scoped:

- Both are held only in the container's user namespace.
- [`BUILD`](BUILD) adds them with `oci_runtime_bundle`'s `config_patch`.
- The customer code runs on gVisor's kernel. gVisor's sandbox process drops
  both capabilities before the customer code starts.
- Unpacking the layer needs neither capability. `bsdtar` keeps directories
  writable until it has extracted their contents.

## Building and testing

```sh
cd containers/gvisor
bazelisk build -c opt //:oci_runtime_bundle
bazelisk test //:process_runner_test
```

The container has its own Bazel module. It pins the gVisor release in
[`MODULE.bazel`](MODULE.bazel).

## Status and limitations

- **Fixed entrypoint.** The sandbox always runs `/entrypoint`. Like the
  tensor names, the entrypoint should come from a configuration.
- **Fixed tensor name.** Records come in and go out in the `data` tensor, as
  in other containers such as sentence_transformers. The input and output
  tensor names should come from a configuration, which needs a new config
  proto in the federated-compute repository first.
- **Failures are silent.** The program's stderr never leaves the sandbox. The
  per-run counters `gvisor-unit-ok` and `gvisor-unit-failed`, which are
  visible outside the TEE, are the only sign that a run failed.
- **Cold start per record.** Every record starts a cold sandbox, which took
  0.3–0.4 s in our tests.
  - gVisor's checkpoint/restore could reuse a snapshot taken before any input
    arrives. That measured about 0.13 s locally, but it isn't wired in.
  - It requires non-rootless runsc.
  - In-process random number generators would have to be reseeded for every
    record.
- **Limits.** There's no per-record memory limit, and the per-record deadline
  is 2 minutes.
- **Deprecated bundle rule.** The container uses Oak's deprecated
  `oci_runtime_bundle` rule, because its replacement, `app_bundle`, can't add
  capabilities yet.
