from __future__ import annotations

import argparse

from cellresponse.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cellresponse")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="M0 data and design audit from pinned metadata")
    audit.add_argument("--config", required=True)
    audit.add_argument(
        "--offline",
        action="store_true",
        help="reuse downloaded metadata, shard index, and expression check",
    )

    args = parser.parse_args(argv)
    if args.command == "audit":
        from cellresponse.data.run_audit import run

        run(load_config(args.config), skip_network=args.offline)


if __name__ == "__main__":
    main()
