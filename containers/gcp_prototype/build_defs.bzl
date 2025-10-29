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

    content = """verifier_type: ITA
require_secboot_enabled: true
require_debug_disabled: false
require_sw_tcb_uptodate: true
"""
    if digest and digest != "skip":
        content += 'expected_image_digest: "%s"\n' % digest

    # buildifier: disable=print
    print("Generating policy:\n" + content)

    out = ctx.actions.declare_file(ctx.attr.out)
    ctx.actions.write(output = out, content = content)
    return [DefaultInfo(files = depset([out]))]

# Rule to generate the policy file
generate_policy = rule(
    implementation = _generate_policy_impl,
    attrs = {
        "out": attr.string(mandatory = True),
        # Point this to the image_digest_flag target
        "digest_flag": attr.label(mandatory = True),
    },
)
