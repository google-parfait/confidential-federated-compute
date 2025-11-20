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

def _generate_policy_impl(ctx):
    digest = ctx.attr.digest_flag[BuildSettingInfo].value

    content = """verifier_type: {type}
allow_debug: true
allow_outdated_hw_tcb: true
""".format(type = ctx.attr.verifier_type)

    if digest and digest != "skip":
        content += 'expected_image_digest: "%s"\n' % digest

    # buildifier: disable=print
    print("Generating policy (%s):\n%s" % (ctx.attr.verifier_type, content))

    out = ctx.actions.declare_file(ctx.attr.out)
    ctx.actions.write(output = out, content = content)
    return [DefaultInfo(files = depset([out]))]

# Rule to generate the policy file
generate_policy = rule(
    implementation = _generate_policy_impl,
    attrs = {
        "out": attr.string(mandatory = True),
        "digest_flag": attr.label(mandatory = True),
        "verifier_type": attr.string(mandatory = True, values = ["ITA", "GCA"]),
    },
)

def define_load_runner(variant, tag, name = None):
    native.genrule(
        name = "load_and_print_digest_runner_" + variant,
        outs = ["load_and_print_digest_" + variant + ".sh"],
        cmd = """
            set -e # Exit on error
            TARBALL_SCRIPT=$(location :tarball_{variant})
            IMAGE_TAG="gcp_prototype:{tag}"

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
            echo "Server Image ({variant}) Docker Digest: $$DIGEST" >&2
            echo "==================================================" >&2

            # --- STEP 5: Generate the dummy output script for bazel run ---
            printf '#!/bin/bash\\n echo "Wrapper script finished."\\n exit 0\\n' > $@
            chmod +x $@
        """.format(variant = variant, tag = tag),
        executable = True,  # Make runnable via `bazel run`
        local = True,  # Allow access to local Docker daemon
        tools = [":tarball_" + variant],
        visibility = ["//visibility:public"],
    )
