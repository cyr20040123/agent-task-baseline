#!/usr/bin/env python3
"""Send a /newsession request to a running llm_proxy."""

import argparse
import configparser
import json
import os
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description="Trigger /newsession on llm_proxy")
    parser.add_argument("session_name", help="New session name")
    parser.add_argument("session_path", nargs="?", default=None, help="New session path (optional)")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), "llm_proxy.ini")
    config.read(config_path)
    host = config.get("DEFAULT", "host", fallback="127.0.0.1")
    port = config.get("DEFAULT", "port", fallback="8030")

    url = f"http://{host}:{port}/newsession"
    if args.session_path and args.session_path != "":
        body = json.dumps({"session_name": args.session_name, "session_path": args.session_path}).encode()
    else:
        body = json.dumps({"session_name": args.session_name}).encode()

    print(f"\n\nRequest sending to {url} with session_name={args.session_name}, session_path={args.session_path}\n\n")

    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req) as resp:
        result = json.loads(resp.read())

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
