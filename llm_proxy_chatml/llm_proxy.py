#!/usr/bin/env python3
"""OpenAI-compatible LLM proxy with ChatML conversation logging."""

import configargparse
import configparser
import logging
import os
import signal
from datetime import datetime

import uvicorn

from chatml_session import SessionManager
from proxy_server import create_app


def parse_args():
    parser = configargparse.ArgParser(
        default_config_files=["llm_proxy.ini"],
        auto_env_var_prefix="LLM_PROXY_",
        description="OpenAI-compatible LLM proxy with ChatML logging",
    )
    parser.add("--host", default="0.0.0.0", help="Proxy listen host")
    parser.add("--port", default=8030, type=int, help="Proxy listen port")
    parser.add("--base-url", required=True, help="Upstream LLM service base URL")
    parser.add("--api-key", default="", help="Upstream API key")
    parser.add("--log-folder", default="./logs/", help="Log and output directory")
    parser.add("--log-chatml", action="store_true", default=False,
               help="Enable ChatML session recording")
    parser.add("--session-name", default=None, help="Initial session name")

    args = parser.parse_args()

    session_explicit = args.session_name is not None
    if not args.session_name:
        args.session_name = "sess_" + datetime.now().strftime("%m%d_%H%M%S")

    # Write back to ini file
    _save_config(args, "llm_proxy.ini", session_explicit)
    return args


def _save_config(args, path, session_explicit):
    config = configparser.ConfigParser()
    config["DEFAULT"] = {
        "host": args.host,
        "port": str(args.port),
        "base-url": args.base_url,
        "api-key": args.api_key,
        "log-folder": args.log_folder,
        "log-chatml": str(args.log_chatml),
    }
    if session_explicit:
        config["DEFAULT"]["session-name"] = args.session_name
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        config.write(f)


def setup_logging(log_folder):
    os.makedirs(log_folder, exist_ok=True)
    log_name = "llm_proxy_" + datetime.now().strftime("%m%d_%H%M") + ".log"
    log_path = os.path.join(log_folder, log_name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("llm_proxy")


def main():
    args = parse_args()
    logger = setup_logging(args.log_folder)

    session_mgr = SessionManager(args.log_folder, args.session_name, args.log_chatml)

    app = create_app(args.base_url, args.api_key, session_mgr)

    # Graceful shutdown — dump sessions
    def shutdown():
        logger.info("shutting down, dumping ChatML sessions…")
        session_mgr.dump_all()

    def _sig_handler(signum, frame):
        shutdown()
        os._exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    logger.info("starting llm_proxy on %s:%d, upstream=%s, log_chatml=%s, log_folder=%s",
                args.host, args.port, args.base_url, args.log_chatml, args.log_folder)
    print(f"Initial session name: {args.session_name}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                access_log=False)


if __name__ == "__main__":
    main()
