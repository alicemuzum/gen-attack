import pytest

from gen_attack.cli import build_parser


def test_classify_defaults():
    args = build_parser().parse_args(["classify"])
    assert args.attacks == ["fgsm", "pgd"]
    assert args.samples == 16


def test_detect_rejects_unknown_attack(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["detect", "--attack", "nope"])
    assert "invalid choice" in capsys.readouterr().err
