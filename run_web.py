"""Entry point for the additive web UI (no changes to existing pipeline files)."""

from webapp.app import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
