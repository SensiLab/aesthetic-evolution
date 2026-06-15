from __future__ import annotations

import os
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, session, url_for

from pairwise_coverage_app.service import PairwiseCoverageError, PairwiseCoverageService


def create_app(data_dir: Path, target_n: int, state_dir: Path | None = None) -> Flask:
    template_dir = Path(__file__).resolve().parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    app.secret_key = os.environ.get("PAIRWISE_COVERAGE_SECRET", "pairwise-coverage-dev-secret")

    if state_dir is None:
        state_dir = Path(__file__).resolve().parent / "state"

    service = PairwiseCoverageService(
        data_dir=data_dir,
        state_dir=state_dir,
        target_n=target_n,
    )

    def _session_id() -> str:
        current = session.get("session_id")
        if current:
            return str(current)
        generated = str(uuid.uuid4())
        session["session_id"] = generated
        return generated

    @app.get("/")
    def root():
        return redirect(url_for("compare"))

    @app.get("/compare")
    def compare():
        status = service.status_payload()
        forced_pair = session.get("forced_pair")
        if forced_pair and status.get("status") == "active":
            image_a = str(forced_pair.get("image_a", ""))
            image_b = str(forced_pair.get("image_b", ""))
            try:
                service.resolve_image_path(image_a)
                service.resolve_image_path(image_b)
            except PairwiseCoverageError:
                session.pop("forced_pair", None)
            else:
                return render_template(
                    "compare.html",
                    image_a=image_a,
                    image_b=image_b,
                    status=status,
                )

        next_pair = service.next_pair()
        if next_pair is None:
            return render_template("status.html", status=status)

        return render_template(
            "compare.html",
            image_a=next_pair.image_a,
            image_b=next_pair.image_b,
            status=status,
        )

    @app.post("/vote")
    def vote():
        image_a = request.form.get("image_a", "")
        image_b = request.form.get("image_b", "")
        outcome = request.form.get("outcome", "")

        if not image_a or not image_b or not outcome:
            return jsonify({"ok": False, "error": "Missing vote fields"}), 400

        try:
            service.record_vote(
                session_id=_session_id(),
                image_a=image_a,
                image_b=image_b,
                outcome=outcome,
            )
            session.pop("forced_pair", None)
        except PairwiseCoverageError as exc:
            if "already been used" in str(exc):
                return redirect(url_for("compare"))
            return jsonify({"ok": False, "error": str(exc)}), 400
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        return redirect(url_for("compare"))

    @app.post("/undo")
    def undo():
        try:
            undone = service.undo_last_vote(session_id=_session_id())
            session["forced_pair"] = {
                "image_a": undone.image_a,
                "image_b": undone.image_b,
            }
        except PairwiseCoverageError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        return redirect(url_for("compare"))

    @app.get("/status")
    def status():
        return jsonify(service.status_payload())

    @app.get("/progress")
    def progress():
        metrics = service.calculate_progress_metrics()
        if metrics is None:
            return jsonify(
                {
                    "ok": True,
                    "implemented": False,
                    "estimated_comparisons_left": None,
                    "average_deviation": None,
                    "max_deviation": None,
                }
            )

        estimated_comparisons_left, average_deviation, max_deviation = metrics
        return jsonify(
            {
                "ok": True,
                "implemented": True,
                "estimated_comparisons_left": estimated_comparisons_left,
                "average_deviation": average_deviation,
                "max_deviation": max_deviation,
            }
        )

    @app.get("/images/<path:image_id>")
    def image_file(image_id: str):
        try:
            path = service.resolve_image_path(image_id)
        except PairwiseCoverageError:
            abort(404)

        return send_file(path)

    return app
