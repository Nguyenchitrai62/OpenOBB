"""Two-step OAuth login for google-colab-cli, usable by agents without a TTY.

colab-cli's oauth2 flow prints a URL and then blocks on input() for the code,
which agents and some Windows terminals cannot drive. This splits it in two:

    python tools/colab_login.py url              # prints the consent URL
    python tools/colab_login.py exchange < code   # reads the code from stdin

It reuses colab-cli's bundled public client and scopes, and writes the token
to the same path the CLI reads (~/.config/colab-cli/token.json), so plain
`colab ...` commands work afterwards. Run it with the colab-cli tool's python:
    %APPDATA%/uv/tools/google-colab-cli/Scripts/python.exe tools/colab_login.py url
"""
import json
import os
import sys
from importlib import resources

from google_auth_oauthlib.flow import InstalledAppFlow

from colab_cli.auth import PUBLIC_SCOPES, REMOTE_REDIRECT_URI, TOKEN_CONFIG_PATH

PENDING_PATH = os.path.expanduser("~/.config/colab-cli/pending_login.json")


def _flow(code_verifier=None):
    client_config = json.loads(
        resources.files("colab_cli").joinpath("oauth_config.json").read_text()
    )
    flow = InstalledAppFlow.from_client_config(
        client_config, PUBLIC_SCOPES, code_verifier=code_verifier,
        autogenerate_code_verifier=code_verifier is None,
    )
    flow.redirect_uri = REMOTE_REDIRECT_URI
    return flow


def cmd_url():
    flow = _flow()
    url, state = flow.authorization_url(prompt="consent", token_usage="remote")
    os.makedirs(os.path.dirname(PENDING_PATH), exist_ok=True)
    with open(PENDING_PATH, "w") as f:
        json.dump({"code_verifier": flow.code_verifier, "state": state}, f)
    print(url)


def cmd_exchange():
    code = sys.stdin.read().strip()
    if not code:
        sys.exit("no authorization code on stdin")
    with open(PENDING_PATH) as f:
        pending = json.load(f)
    flow = _flow(pending["code_verifier"])
    flow.fetch_token(code=code)
    with open(TOKEN_CONFIG_PATH, "w") as f:
        f.write(flow.credentials.to_json())
    os.remove(PENDING_PATH)
    print(f"token saved to {TOKEN_CONFIG_PATH}")


if __name__ == "__main__":
    {"url": cmd_url, "exchange": cmd_exchange}[sys.argv[1]]()
