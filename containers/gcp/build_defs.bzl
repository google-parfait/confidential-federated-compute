"""Build definitions for the GCP prototype."""

BuildSettingInfo = provider(
    doc = "Provider for passing build setting values.",
    fields = ["value"],
)

def _image_digest_flag_impl(ctx):
    return [BuildSettingInfo(value = ctx.build_setting_value)]

# Defines a custom string flag we can pass on the command line
image_digest_flag = rule(
    implementation = _image_digest_flag_impl,
    build_setting = config.string(flag = True),
)

IntSettingInfo = provider(
    doc = "Provider for passing int build setting values.",
    fields = ["value"],
)

def _int_flag_impl(ctx):
    return [IntSettingInfo(value = ctx.build_setting_value)]

int_flag = rule(
    implementation = _int_flag_impl,
    build_setting = config.int(flag = True),
)

def _generate_policy_impl(ctx):
    digest = ctx.attr.digest_flag[BuildSettingInfo].value
    out = ctx.actions.declare_file(ctx.attr.out)

    if digest and digest != "skip":
        # Manual override: single digest from --//:server_digest flag.
        content = """verifier_type: {type}
allow_debug: true
allow_outdated_hw_tcb: true
expected_image_digest: "{digest}"
""".format(type = ctx.attr.verifier_type, digest = digest)

        # buildifier: disable=print
        print("Generating policy (%s) [manual digest]:\n%s" % (ctx.attr.verifier_type, content))
        ctx.actions.write(output = out, content = content)
    elif ctx.file.registry_file:
        # Auto mode: run generate_policy.py at execution time for date filtering.
        model_filter = ctx.attr.server_model[BuildSettingInfo].value if ctx.attr.server_model else ""
        attest_filter = ctx.attr.server_attestation[BuildSettingInfo].value if ctx.attr.server_attestation else ""
        max_age = ctx.attr.server_max_age_days[IntSettingInfo].value if ctx.attr.server_max_age_days else 60

        ctx.actions.run_shell(
            outputs = [out],
            inputs = [ctx.file.registry_file],
            tools = [ctx.file._generate_policy_script],
            command = (
                "python3 {script}" +
                " --registry={registry}" +
                " --output={out}" +
                " --verifier_type={type}" +
                " --model={model}" +
                " --attestation={attest}" +
                " --max_age_days={age}"
            ).format(
                script = ctx.file._generate_policy_script.path,
                registry = ctx.file.registry_file.path,
                out = out.path,
                type = ctx.attr.verifier_type,
                model = model_filter,
                attest = attest_filter,
                age = max_age,
            ),
        )
    else:
        # No digest, no registry — generate policy without digest check.
        content = """verifier_type: {type}
allow_debug: true
allow_outdated_hw_tcb: true
""".format(type = ctx.attr.verifier_type)

        # buildifier: disable=print
        print("Generating policy (%s) [no digest check]:\n%s" % (ctx.attr.verifier_type, content))
        ctx.actions.write(output = out, content = content)

    return [DefaultInfo(files = depset([out]))]

# Rule to generate the policy file
generate_policy = rule(
    implementation = _generate_policy_impl,
    attrs = {
        "out": attr.string(mandatory = True),
        "digest_flag": attr.label(mandatory = True),
        "verifier_type": attr.string(mandatory = True, values = ["ITA", "GCA"]),
        "registry_file": attr.label(
            allow_single_file = [".json"],
            doc = "server_image_registry.json for auto-populating digests.",
        ),
        "server_model": attr.label(
            doc = "Build flag to filter registry by model name.",
        ),
        "server_attestation": attr.label(
            doc = "Build flag to filter registry by attestation flavor.",
        ),
        "server_max_age_days": attr.label(
            doc = "Build flag: max age in days for registry entries (default: 60).",
        ),
        "_generate_policy_script": attr.label(
            default = ":generate_policy.py",
            allow_single_file = True,
            doc = "The Python script that generates policy from the registry.",
        ),
    },
)

def define_load_runner(variant, image_and_tag, name = None):
    native.genrule(
        name = "load_and_print_digest_runner_" + variant,
        outs = ["load_and_print_digest_" + variant + ".sh"],
        cmd = """
            set -e # Exit on error
            TARBALL_SCRIPT=$(location :{variant}_tarball)
            IMAGE_TAG="{image_and_tag}"

            # --- STEP 1: Execute the oci_load script ---
            echo "Executing oci_load script: $$TARBALL_SCRIPT" >&2
            $$TARBALL_SCRIPT # This runs 'docker load'
            echo "Image load complete." >&2

            # --- STEP 2: Get the Image ID for the loaded tag ---
            echo "Inspecting tag '$$IMAGE_TAG' to get Image ID..." >&2
            LOADED_ID=$$(docker image inspect "$$IMAGE_TAG" --format '{{{{.Id}}}}' 2>/dev/null)

            if [ -z "$$LOADED_ID" ]; then
                echo "ERROR: Could not inspect tag '$$IMAGE_TAG' to get Image ID after load." >&2
                exit 1
            fi
            echo "Found Image ID for tag '$$IMAGE_TAG': $$LOADED_ID" >&2

            # --- STEP 3: Inspect the specific Image ID to get the RepoDigest (Manifest Digest) ---
            echo "Inspecting loaded Image ID '$$LOADED_ID' for RepoDigest..." >&2
            
            # FIX: Quadruple braces for println and end to survive Python .format()
            DIGEST=$$(docker image inspect "$$LOADED_ID" --format '{{{{range .RepoDigests}}}}{{{{.}}}}{{{{println}}}}{{{{end}}}}' 2>/dev/null | grep -o 'sha256:[a-f0-9]*' | head -n 1 || true)

            # Fallback to the Image ID itself ONLY if RepoDigests is empty for that ID
            if [ -z "$$DIGEST" ]; then
                echo "WARNING: RepoDigest not found for specific Image ID $$LOADED_ID after load. Falling back to Image ID (Config Digest)." >&2
                # Use the ID we got from inspecting the tag
                DIGEST="$$LOADED_ID"
            else
                echo "Found RepoDigest (Manifest Digest) for Image ID $$LOADED_ID: $$DIGEST" >&2
            fi

            # --- STEP 4: Print the final digest ---
            # This should now be the Manifest Digest if available, otherwise the Config Digest.
            echo "==================================================" >&2
            echo "Server Image ({image_and_tag}) Docker Digest: $$DIGEST" >&2
            echo "==================================================" >&2

            # --- STEP 5: Generate the dummy output script for bazel run ---
            printf '#!/bin/bash\\n echo "Wrapper script finished."\\n exit 0\\n' > $@
            chmod +x $@
        """.format(variant = variant, image_and_tag = image_and_tag),
        executable = True,  # Make runnable via `bazel run`
        local = True,  # Allow access to local Docker daemon
        tools = [":" + variant + "_tarball"],
        visibility = ["//visibility:public"],
    )
