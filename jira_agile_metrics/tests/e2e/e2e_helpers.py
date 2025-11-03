"""Shared helper functions for e2e tests."""


def write_config_and_get_parser_args(tmp_path, config_yaml, configure_argument_parser):
    """Write config YAML to file and return parser and parsed arguments.

    Args:
        tmp_path: Temporary directory path
        config_yaml: Configuration YAML string
        configure_argument_parser: Function to create argument parser

    Returns:
        Tuple of (parser, args)
    """
    config_path = tmp_path / "e2e_config.yml"
    config_path.write_text(config_yaml, encoding="utf-8")

    parser = configure_argument_parser()
    args = parser.parse_args([str(config_path)])
    return parser, args
