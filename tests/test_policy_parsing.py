import pytest
from torchruntime.utils.args import parse_policy_args


def test_default_policy():
    args = ["pkg1"]
    preview, unsupported, cleaned = parse_policy_args(args)
    assert preview is False
    assert unsupported is True
    assert cleaned == ["pkg1"]


def test_stable_policy():
    args = ["--policy", "stable", "pkg1"]
    preview, unsupported, cleaned = parse_policy_args(args)
    assert preview is False
    assert unsupported is False
    assert cleaned == ["pkg1"]


def test_nightly_policy():
    args = ["--policy", "nightly"]
    preview, unsupported, cleaned = parse_policy_args(args)
    assert preview is True
    assert unsupported is True
    assert cleaned == []


def test_preview_policy_alias():
    args = ["--policy", "preview"]
    preview, unsupported, cleaned = parse_policy_args(args)
    assert preview is True
    assert unsupported is True
    assert cleaned == []


def test_policy_equals_syntax():
    args = ["--policy=stable", "pkg1"]
    preview, unsupported, cleaned = parse_policy_args(args)
    assert preview is False
    assert unsupported is False
    assert cleaned == ["pkg1"]


def test_policy_override_preview():
    # stable is p=F, u=F. --preview should make p=T
    args = ["--policy", "stable", "--preview"]
    preview, unsupported, cleaned = parse_policy_args(args)
    assert preview is True
    assert unsupported is False

def test_policy_override_unsupported():
    # nightly is p=T, u=T. --no-unsupported should make u=F
    args = ["--policy", "nightly", "--no-unsupported"]
    preview, unsupported, cleaned = parse_policy_args(args)
    assert preview is True
    assert unsupported is False


def test_unknown_policy():
    args = ["--policy", "nonexistent"]
    with pytest.raises(ValueError, match="Unknown policy"):
        parse_policy_args(args)


def test_missing_policy_arg():
    args = ["--policy"]
    with pytest.raises(ValueError, match="--policy requires an argument"):
        parse_policy_args(args)


def test_mixed_args():
    args = ["torch", "--preview", "--policy", "stable", "--uv"]
    # stable: p=F, u=F
    # --preview: p=T
    # Result: p=T, u=F
    # cleaned: ["torch", "--uv"]
    preview, unsupported, cleaned = parse_policy_args(args)
    assert preview is True
    assert unsupported is False
    assert cleaned == ["torch", "--uv"]
