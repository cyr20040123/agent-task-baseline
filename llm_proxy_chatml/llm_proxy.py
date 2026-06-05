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
    parser.add_argument("--host", default="0.0.0.0", help="Proxy listen host")
    parser.add_argument("--port", default=8030, type=int, help="Proxy listen port")
    parser.add_argument("--base-url", required=True, help="Upstream LLM service base URL")
    parser.add_argument("--api-key", default="", help="Upstream API key")
    parser.add_argument("--log-folder", default="./logs/", help="Log and output directory")
    parser.add_argument("--log-chatml", choices=["none", "multi", "single"], default="none",
               help="ChatML recording mode: none (disabled), multi (prefix-matched "
               "multi-turn), single (one entry per request)")
    parser.add_argument("--session-name", default=None, help="Initial session name")
    parser.add_argument("--session-path", default="",
               help="ChatML output path (defaults to --log-folder)")
    parser.add_argument("--temperature", default=-1.0, type=float,
               help="Default temperature for upstream requests. "
               "When non-negative and the request does not already contain "
               "a 'temperature' field, the value is injected. "
               "Default -1.0 (disabled).")
    parser.add_argument("--rl", action="store_true", default=False,
               help="Enable RL-specific ChatML logging. When enabled, "
               "logprobs and token_ids are requested from upstream and "
               "recorded alongside each assistant response.")

    args = parser.parse_args()

    session_explicit = args.session_name is not None
    if not args.session_name:
        args.session_name = "sess_" + datetime.now().strftime("%m%d_%H%M%S")
    if not args.session_path:
        args.session_path = args.log_folder

    # Write back: preserve [DEFAULT] + other sections, update only [RECENT]
    _save_config(args, "llm_proxy.ini", session_explicit)
    return args


def _save_config(args, path, session_explicit):
    # Read existing ini to preserve all sections except [RECENT]
    old = configparser.ConfigParser()
    old.read(path)

    # Build new ini: copy all sections except [RECENT], then write [RECENT]
    ini = configparser.ConfigParser()
    if old.defaults():
        ini["DEFAULT"] = dict(old.defaults())
    for sec in old.sections():
        if sec == "RECENT":
            continue
        ini[sec] = dict(old[sec])

    recent = {
        "host": args.host,
        "port": str(args.port),
        "base-url": args.base_url,
        "api-key": args.api_key,
        "log-folder": args.log_folder,
        "log-chatml": str(args.log_chatml),
        "temperature": str(args.temperature),
        "rl": str(args.rl),
    }
    if session_explicit:
        recent["session-name"] = args.session_name
    ini["RECENT"] = recent

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        ini.write(f)


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

    session_mgr = SessionManager(args.log_folder, args.session_name, args.log_chatml,
                                 args.session_path, rl_enabled=args.rl)

    app = create_app(args.base_url, args.api_key, session_mgr, args.temperature)

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
