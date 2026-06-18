#!/usr/bin/env python3
"""OpenAI-compatible LLM proxy with ChatML conversation logging."""

import argparse
import logging
import os
import signal
from datetime import datetime

import httpx
import uvicorn
import yaml

from chatml_session import SessionManager
from proxy_server import create_app

CONFIG_FILE = "llm_proxy.yaml"

# Sentinel for CLI args not explicitly set
_UNSET = object()

# Hardcoded defaults (lowest priority)
_HARD_DEFAULTS = {
    "host": "0.0.0.0",
    "port": 8030,
    "base_url": "",
    "api_key": "",
    "log_folder": "./logs/",
    "log_chatml": "none",
    "session_name": None,
    "session_path": "",
    "temperature": -1.0,
    "rl": False,
    "default_model": None,
    "override_model": False,
}

# Fields persisted to YAML RECENT (in order)
_CONFIG_FIELDS = [
    "host", "port", "base_url", "api_key", "log_folder", "log_chatml",
    "session_name", "session_path", "temperature", "rl", "default_model",
    "override_model",
]

# Fields that are always written to RECENT (even if matching DEFAULT)
_ALWAYS_RECENT = set()


def _load_yaml(path):
    """Load YAML config file. Returns empty dict if missing or unreadable."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _cli_to_yaml_key(cli_key):
    """Convert argparse dest (underscores) to YAML key (kebab-case)."""
    return cli_key.replace("_", "-")


def _yaml_to_cli_key(yaml_key):
    """Convert YAML key (kebab-case) to argparse dest (underscores)."""
    return yaml_key.replace("-", "_")


def _yaml_value(key, value):
    """Coerce a YAML value to the correct Python type for the given config key."""
    if key in ("port",):
        return int(value)
    if key in ("temperature",):
        return float(value)
    if key in ("rl",):
        return bool(value)
    if value is None:
        return None
    return value


def parse_args():
    # -- CLI parser -------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible LLM proxy with ChatML logging",
    )
    parser.add_argument("--host", default=_UNSET, help="Proxy listen host")
    parser.add_argument("--port", default=_UNSET, type=int, help="Proxy listen port")
    parser.add_argument("--base-url", default=_UNSET, help="Upstream LLM service base URL")
    parser.add_argument("--api-key", default=_UNSET, help="Upstream API key")
    parser.add_argument("--log-folder", default=_UNSET, help="Log and output directory")
    parser.add_argument("--log-chatml", default=_UNSET,
                        choices=["none", "multi", "single"],
                        help="ChatML recording mode: none, multi, single")
    parser.add_argument("--session-name", default=_UNSET, help="Initial session name")
    parser.add_argument("--session-path", default=_UNSET,
                        help="ChatML output path (defaults to --log-folder)")
    parser.add_argument("--temperature", default=_UNSET, type=float,
                        help="Default temperature for upstream requests")
    parser.add_argument("--rl", default=None, action="store_true",
                        help="Enable RL-specific ChatML logging")
    parser.add_argument("--override-model", default=None, action="store_true",
                        help="Force all requests to use --default-model")
    parser.add_argument("--default-model", default=_UNSET,
                        help="Default model for requests with empty or 'none' model")
    parser.add_argument("--preset", default=_UNSET,
                        help="Name of a YAML config group to load (e.g. DEEPSEEK)")

    cli_args = parser.parse_args()

    # -- Load YAML config -------------------------------------------------
    yaml_data = _load_yaml(CONFIG_FILE)
    defaults = yaml_data.get("DEFAULT", {}) or {}
    recents = yaml_data.get("RECENT", {}) or {}
    preset_name = None
    if cli_args.preset is not _UNSET:
        preset_name = cli_args.preset
    preset = yaml_data.get(preset_name, {}) or {} if preset_name else {}

    # -- Resolve each field: CLI > preset > RECENT > DEFAULT > hardcoded --
    def _cli_val(key):
        """Get CLI value for field, returning _UNSET if not explicitly provided."""
        # argparse action="store_true" always returns True/False, never _UNSET,
        # so detect via the actual dest attr
        if key in ("rl", "override_model"):
            return _UNSET if getattr(cli_args, key) is None else getattr(cli_args, key)
        return getattr(cli_args, key)

    resolved = {}
    _base = {}  # DEFAULT-level baseline for diff logging
    session_explicit = False
    for key in _HARD_DEFAULTS:
        yk = _cli_to_yaml_key(key)
        # Compute DEFAULT-level baseline first
        if yk in defaults:
            _base[key] = _yaml_value(key, defaults[yk])
        else:
            _base[key] = _HARD_DEFAULTS[key]
        # Resolve with full priority
        cli_v = _cli_val(key)
        if cli_v is not _UNSET:
            resolved[key] = cli_v
            if key == "session_name":
                session_explicit = True
            continue
        if preset_name and yk in preset:
            resolved[key] = _yaml_value(key, preset[yk])
            continue
        if yk in recents:
            resolved[key] = _yaml_value(key, recents[yk])
            continue
        if yk in defaults:
            resolved[key] = _yaml_value(key, defaults[yk])
            continue
        resolved[key] = _HARD_DEFAULTS[key]

    resolved["_defaults"] = _base

    preset_name_str = preset_name if preset_name else None
    resolved["preset"] = preset_name_str

    # Auto session name if not set
    if not resolved["session_name"]:
        resolved["session_name"] = "sess_" + datetime.now().strftime("%m%d_%H%M%S")
    if not resolved["session_path"]:
        resolved["session_path"] = resolved["log_folder"]

    # Validate required base-url
    if not resolved["base_url"]:
        parser.error("--base-url is required (set via CLI, YAML DEFAULT, or --preset)")

    # -- Write back YAML --------------------------------------------------
    _save_config(resolved, session_explicit)

    # Return a namespace for backward compatibility
    return argparse.Namespace(**resolved)


def _save_config(args, session_explicit):
    """Write back YAML config:
    - RECENT: only fields that differ from DEFAULT (plus _ALWAYS_RECENT)
    - Clean up other non-DEFAULT groups: remove fields matching DEFAULT
    """
    yaml_data = _load_yaml(CONFIG_FILE)
    defaults = yaml_data.get("DEFAULT", {}) or {}

    # Determine which fields differ from DEFAULT
    recent = {}
    for key in _CONFIG_FIELDS:
        yk = _cli_to_yaml_key(key)
        val = args.get(key) if isinstance(args, dict) else getattr(args, key)
        if key == "session_name" and not session_explicit:
            continue  # auto-generated names are not persisted
        if yk in _ALWAYS_RECENT:
            recent[yk] = val
        elif val != defaults.get(yk):
            recent[yk] = val

    # Update RECENT
    if recent:
        yaml_data["RECENT"] = recent
    elif "RECENT" in yaml_data:
        del yaml_data["RECENT"]

    # Clean up other groups: remove fields matching DEFAULT
    for group_name in list(yaml_data.keys()):
        if group_name in ("DEFAULT", "RECENT"):
            continue
        group = yaml_data[group_name]
        if not isinstance(group, dict):
            continue
        cleaned = {}
        for yk, val in group.items():
            if val != defaults.get(yk):
                cleaned[yk] = val
        if cleaned:
            yaml_data[group_name] = cleaned
        else:
            del yaml_data[group_name]

    # Write
    os.makedirs(os.path.dirname(CONFIG_FILE) or ".", exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


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

    # --- log resolved config, marking deviations from DEFAULT ---
    base = getattr(args, "_defaults", {})
    parts = []
    for key in _CONFIG_FIELDS:
        val = getattr(args, key)
        if key in ("api_key",) and val:
            display = val[:8] + "***" if len(val) > 8 else "***"
        else:
            display = val
        marker = "*" if base and val != base.get(key) else " "
        parts.append(f"{marker}{key}={display}")
    logger.info("config: %s", "  ".join(parts))

    session_mgr = SessionManager(args.log_folder, args.session_name, args.log_chatml,
                                 args.session_path, rl_enabled=args.rl)

    # --- fetch available models from upstream ---
    # Try without /v1 prefix first, then with /v1 if that fails.
    # If /v1 works, the upstream is an OpenAI-compatible service that
    # expects the /v1 prefix — update base_url accordingly.
    available_models = []
    base_url = args.base_url.rstrip("/")
    for attempt, suffix in enumerate(("", "/v1")):
        try:
            models_url = f"{base_url}{suffix}/models"
            resp = httpx.get(models_url,
                             headers={"Authorization": f"Bearer {args.api_key}"} if args.api_key else {},
                             timeout=10)
            resp.raise_for_status()
            model_list = resp.json().get("data", [])
            available_models = [m["id"] for m in model_list if m.get("id")]
            if suffix == "/v1":
                args.base_url = base_url + "/v1"
                logger.warning("auto-detected /v1 prefix: base-url updated to '%s'", args.base_url)
            logger.info("available models from upstream (%d): %s",
                        len(available_models), ", ".join(available_models))
            break
        except Exception as e:
            if attempt == 0:
                logger.debug("models fetch without /v1 failed, retrying with /v1: %s", e)
            else:
                logger.warning("failed to fetch models from upstream: %s", e)

    # --- default model ---
    default_model = args.default_model
    if default_model is None and available_models:
        default_model = available_models[0]
        logger.info("default-model auto-set to '%s' (first available)", default_model)
    elif default_model is not None:
        logger.info("default-model set to '%s' (from config)", default_model)

    app = create_app(args.base_url, args.api_key, session_mgr, args.temperature,
                     default_model=default_model, override_model=args.override_model)
    app.state.available_models = available_models

    # Graceful shutdown — dump sessions
    def shutdown():
        logger.info("shutting down, dumping ChatML sessions…")
        path = session_mgr.dump_all()
        logger.info("sessions dumped to %s", path)

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
