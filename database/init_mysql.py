from __future__ import annotations

import argparse
import pathlib
import shlex
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize MySQL schema for foot-analysis")
    parser.add_argument("--host", default="127.0.0.1", help="MySQL host")
    parser.add_argument("--port", type=int, default=3306, help="MySQL port")
    parser.add_argument("--user", default="cesar", help="MySQL user")
    parser.add_argument("--password", default="cesar123", help="MySQL password")
    parser.add_argument(
        "--schema-file",
        default=str(pathlib.Path(__file__).with_name("mysql_schema.sql")),
        help="Path to schema SQL file",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    schema_path = pathlib.Path(args.schema_file).resolve()

    if not schema_path.exists():
        print(f"Schema file not found: {schema_path}")
        return 1

    mysql_cmd = [
        "mysql",
        f"--host={args.host}",
        f"--port={args.port}",
        f"--user={args.user}",
        f"--password={args.password}",
    ]

    try:
        with schema_path.open("rb") as sql_file:
            completed = subprocess.run(mysql_cmd, stdin=sql_file, check=False)
    except FileNotFoundError:
        print("mysql client not found in PATH. Install mysql-client and retry.")
        return 1

    if completed.returncode != 0:
        print("MySQL schema initialization failed.")
        print("Executed command:")
        print(" ".join(shlex.quote(part) for part in mysql_cmd))
        return completed.returncode

    print("MySQL schema initialized successfully on database: foot_analysis_db")
    return 0


if __name__ == "__main__":
    sys.exit(main())