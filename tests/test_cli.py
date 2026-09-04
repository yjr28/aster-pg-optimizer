from aster.cli.main import build_parser


def test_cli_exposes_primary_commands():
    parser = build_parser()
    for command in ["collect", "train", "optimize", "benchmark"]:
        try:
            parser.parse_args([command, "--help"])
        except SystemExit as exc:
            assert exc.code == 0


def test_train_cli_accepts_robustness_split_regimes():
    parser=build_parser()
    for regime in ("template","parameter","workload"):
        args=parser.parse_args([
            "train","--dataset","dataset.jsonl","--model-out","model.joblib",
            "--split-regime",regime,
        ])
        assert args.split_regime == regime
