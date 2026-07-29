"""Secure API-key configuration for the trace collector.

API keys are sensitive credentials. This module resolves them in this order:

  1. OS credential vault via `keyring` (Windows Credential Manager here) —
     the recommended store for a developer machine: encrypted at rest,
     per-user, never sitting in a plaintext file or shell history.
  2. Environment variable (for CI or an already-managed environment).
  3. A `.env` file at the repo root (weakest option: plaintext on disk;
     it is gitignored, but prefer the vault).

Set a key ONCE, with hidden input (never echoed, never in shell history):

    py -m pip install keyring
    py -m derail.config set-key GEMINI_API_KEY

Then verify / rotate / remove:

    py -m derail.config check GEMINI_API_KEY      # prints masked value only
    py -m derail.config set-key GEMINI_API_KEY    # overwrite = rotate
    py -m derail.config delete-key GEMINI_API_KEY

Rules enforced here: the key value is never printed, logged, or embedded in
error messages (only a ****-masked suffix); nothing in this repo ever writes
a key anywhere except the OS vault. For a real server deployment, use the
platform's secret manager (GCP Secret Manager / AWS Secrets Manager / Vault)
instead of any of the above — this module is for the developer-machine case.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SERVICE = "derail-monitor"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _mask(value: str) -> str:
    """Displayable form of a secret: source-checkable, not recoverable."""
    return f"****{value[-4:]}" if len(value) >= 8 else "****"


def _from_keyring(name: str) -> str | None:
    try:
        import keyring
        return keyring.get_password(_SERVICE, name)
    except Exception:  # noqa: BLE001 — keyring missing/locked -> fall through
        return None


def _from_dotenv(name: str) -> str | None:
    env_path = _REPO_ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{name}=") and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def get_api_key(name: str, required: bool = False) -> str | None:
    """Resolve a key: keyring -> environment -> .env. Never logs the value."""
    for source in (_from_keyring, lambda n: os.environ.get(n) or None,
                   _from_dotenv):
        value = source(name)
        if value:
            return value
    if required:
        raise SystemExit(
            f"{name} is not configured. Store it securely with:\n"
            f"  py -m derail.config set-key {name}\n"
            f"(or set the {name} environment variable / a gitignored .env)")
    return None


def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="py -m derail.config")
    parser.add_argument("command", choices=["set-key", "check", "delete-key"])
    parser.add_argument("name", help="e.g. GEMINI_API_KEY")
    args = parser.parse_args(argv)

    if args.command == "set-key":
        import getpass
        try:
            import keyring
        except ImportError:
            raise SystemExit("pip install keyring first (stores the key in "
                             "the Windows Credential Manager)")
        value = getpass.getpass(f"{args.name} (input hidden): ").strip()
        if not value:
            raise SystemExit("empty value; nothing stored")
        keyring.set_password(_SERVICE, args.name, value)
        print(f"stored {args.name} = {_mask(value)} in the OS credential "
              f"vault (service '{_SERVICE}')")
    elif args.command == "check":
        value = get_api_key(args.name)
        if value is None:
            print(f"{args.name}: NOT configured")
            sys.exit(1)
        source = ("keyring" if _from_keyring(args.name)
                  else "environment" if os.environ.get(args.name) else ".env")
        print(f"{args.name}: configured ({_mask(value)}, source: {source})")
    else:  # delete-key
        try:
            import keyring
        except ImportError:
            raise SystemExit("pip install keyring first (stores the key in "
                             "the Windows Credential Manager)")
        try:
            keyring.delete_password(_SERVICE, args.name)
            print(f"deleted {args.name} from the OS credential vault")
        except keyring.errors.PasswordDeleteError:
            print(f"{args.name} was not in the vault (env/.env copies, if "
                  "any, must be removed manually)")


if __name__ == "__main__":
    _cli()
