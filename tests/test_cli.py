from aster.cli.main import build_parser


def test_cli_exposes_collect_train_optimize_commands():
    parser = build_parser()
    for command in ["collect", "train", "optimize"]:
        try:
            parser.parse_args([command, "--help"])
        except SystemExit as exc:
            assert exc.code == 0
