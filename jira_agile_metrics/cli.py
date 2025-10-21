import argparse
import getpass
import logging
import os

from dotenv import load_dotenv
from jira import JIRA

from .calculator import run_calculators
from .config import ConfigError, config_to_options
from .config_main import CALCULATORS
from .querymanager import QueryManager
from .trello import TrelloClient
from .utils import set_chart_context
from .webapp.app import app as webapp

load_dotenv()

logger = logging.getLogger(__name__)


def configure_argument_parser():
    """Configure an ArgumentParser that manages command line options."""

    parser = argparse.ArgumentParser(
        description=("Extract Agile metrics data from JIRA/Trello and produce data and charts.")
    )

    # Basic options
    parser.add_argument("config", metavar="config.yml", nargs="?", help="Configuration file")
    parser.add_argument("-v", dest="verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "-vv",
        dest="very_verbose",
        action="store_true",
        help="Even more verbose output",
    )
    parser.add_argument(
        "-n",
        metavar="N",
        dest="max_results",
        type=int,
        help="Only fetch N most recently updated issues",
    )

    parser.add_argument(
        "--server",
        metavar="127.0.0.1:8080",
        help=(
            "Run as a web server instead of a command line tool, "
            "on the given host and/or port."
            "The remaining options do not apply."
        ),
    )

    # Output directory
    parser.add_argument(
        "--output-directory",
        "-o",
        metavar="metrics",
        help=("Write output files to this directory,rather than the current working directory."),
    )

    # Connection options
    parser.add_argument("--domain", metavar="https://my.jira.com", help="JIRA domain name")
    parser.add_argument("--username", metavar="user", help="JIRA/Trello user name")
    parser.add_argument("--password", metavar="password", help="JIRA password")
    parser.add_argument("--key", metavar="key", help="Trello API key")
    parser.add_argument("--token", metavar="token", help="Trello API password")
    parser.add_argument("--http-proxy", metavar="https://proxy.local", help="URL to HTTP Proxy")
    parser.add_argument(
        "--https-proxy",
        metavar="https://proxy.local",
        help="URL to HTTPS Proxy",
    )
    parser.add_argument(
        "--jira-server-version-check",
        type=bool,
        metavar="True",
        help=(
            "If true it will fetch JIRA server version info first"
            "to determine if some API calls are available"
        ),
    )

    return parser


def main():
    parser = configure_argument_parser()
    args = parser.parse_args()

    if args.server:
        run_server(parser, args)
    else:
        run_command_line(parser, args)


def run_server(parser, args):
    host = None
    port = args.server

    if ":" in args.server:
        (host, port) = args.server.split(":")
    port = int(port)

    set_chart_context("paper")
    webapp.run(host=host, port=port)


def run_command_line(parser, args):
    if not args.config:
        parser.print_usage()
        return

    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=(
            logging.DEBUG
            if args.very_verbose
            else logging.INFO if args.verbose else logging.WARNING
        ),
    )

    # Configuration and settings
    # (command line arguments override config file options)

    logger.debug("Parsing options from %s", args.config)
    try:
        with open(args.config) as config:
            options = config_to_options(
                config.read(), cwd=os.path.dirname(os.path.abspath(args.config))
            )
    except FileNotFoundError:
        print(
            f"Error: Configuration file '{args.config}' not found. "
            "Please provide a valid config file."
        )
        return

    # Allow command line arguments to override options
    override_options(options["connection"], args)
    override_options(options["settings"], args)

    # Set charting context, which determines how charts are rendered
    set_chart_context("paper")

    # Set output directory if required
    output_dir = None
    if "output_directory" in options:
        output_dir = options["output_directory"]
    if args.output_directory:
        output_dir = args.output_directory
    if output_dir:
        logger.info("Changing working directory to %s" % output_dir)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)
    logger.debug(f"[DEBUG] Config file path: {args.config}")
    logger.debug(f"[DEBUG] Initial output_directory in options: {options.get('output_directory')}")
    logger.debug(f"[DEBUG] Current working directory: {os.getcwd()}")
    logger.debug(f"[DEBUG] output_dir value: {output_dir}")

    # Select data source
    jira = None

    if options["connection"]["type"] == "jira":
        jira = get_jira_client(options["connection"])
    elif options["connection"]["type"] == "trello":
        jira = get_trello_client(options["connection"], options["settings"]["type_mapping"])
    else:
        raise ConfigError("Unknown source")
    # Query JIRA and run calculators
    logger.info("Running calculators")
    query_manager = QueryManager(jira, options["settings"])
    run_calculators(CALCULATORS, query_manager, options["settings"])


def override_options(options, arguments):
    """Update `options` dict with settings from `arguments`
    with the same key.
    """
    for key in options.keys():
        if getattr(arguments, key, None) is not None:
            options[key] = getattr(arguments, key)


def get_jira_client(connection):
    url = connection["domain"] or os.environ.get("JIRA_URL")
    username = connection["username"]
    if not username:
        username = os.environ.get("JIRA_USERNAME")
    password = connection["password"]
    if not password:
        password = os.environ.get("JIRA_PASSWORD")
    http_proxy = connection["http_proxy"]
    https_proxy = connection["https_proxy"]
    jira_server_version_check = connection["jira_server_version_check"]
    jira_client_options = connection["jira_client_options"]

    logger.info("Connecting to %s", url)

    if not username:
        username = input("Username: ")

    if not password:
        password = getpass.getpass("Password: ")

    options = {"server": url, "rest_api_version": 3}
    proxies = None

    if http_proxy or https_proxy:
        proxies = {}
        if http_proxy:
            proxies["http"] = http_proxy
        if https_proxy:
            proxies["https"] = https_proxy

    options.update(jira_client_options)

    return JIRA(
        options,
        basic_auth=(username, password),
        proxies=proxies,
        get_server_info=jira_server_version_check,
    )


def get_trello_client(connection, type_mapping):
    username = connection["username"]
    key = connection["key"]
    token = connection["token"]

    if not username:
        username = input("Username: ")

    if not key:
        key = getpass.getpass("Key: ")

    if not token:
        token = getpass.getpass("Token: ")

    return TrelloClient(username, key, token, type_mapping=type_mapping)


if __name__ == "__main__":
    main()
