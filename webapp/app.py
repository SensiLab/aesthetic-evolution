from __future__ import annotations

from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for

from webapp.service import JobService, ValidationError


def create_app() -> Flask:
    workspace_root = Path(__file__).resolve().parents[1]
    app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / "templates"))
    service = JobService(workspace_root=workspace_root)

    default_prompt_path = workspace_root / "config" / "reasoning_prompt.txt"
    default_prompt_text = default_prompt_path.read_text(encoding="utf-8") if default_prompt_path.exists() else ""

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            jobs=[job.to_dict() for job in service.list_jobs()],
            experiments=service.list_experiments(),
            defaults={
                "runs": 5,
                "population_size": 10,
                "param_spec_file": "config/param_spec.yaml",
                "prompt_text": default_prompt_text,
                "processing": "parallel",
                "screen": False,
                "workers": 8,
                "alpha_mode": "biased",
                "alpha": "",
                "mutation_rate": 0.1,
                "mutation_sigma": 0.1,
                "parents_compete": True,
                "competing_parents_rate": 0.1,
                "k": 0.5,
                "ranking_method": "glicko",
                "overwrite": False,
                "sketch_dir": "/home/sjkro1/ARC-Discovery/Harmonograph",
            },
        )

    @app.post("/jobs")
    def create_job():
        payload = request.form.to_dict(flat=True)
        payload["screen"] = "screen" in request.form
        payload["parents_compete"] = "parents_compete" in request.form
        payload["overwrite"] = "overwrite" in request.form

        try:
            record = service.submit(payload)
        except ValidationError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        return redirect(url_for("job_status_page", job_id=record.job_id))

    @app.get("/jobs/<job_id>")
    def job_status_page(job_id: str):
        job = service.get_job(job_id)
        if not job:
            abort(404)

        return render_template("job.html", job=job.to_dict())

    @app.get("/api/jobs/<job_id>")
    def api_job_status(job_id: str):
        job = service.get_job(job_id)
        if not job:
            return jsonify({"error": "not found"}), 404

        return jsonify(job.to_dict())

    @app.get("/api/jobs/<job_id>/log")
    def api_job_log(job_id: str):
        job = service.get_job(job_id)
        if not job:
            return jsonify({"error": "not found"}), 404

        if not job.log_path:
            return jsonify({"log": ""})

        log_path = Path(job.log_path)
        if not log_path.exists():
            return jsonify({"log": ""})

        text = log_path.read_text(encoding="utf-8")
        return jsonify({"log": text[-20000:]})

    @app.get("/experiments/<experiment_name>")
    def experiment_page(experiment_name: str):
        try:
            runs = service.experiment_runs(experiment_name)
        except ValidationError:
            abort(404)

        selected_run = request.args.get("run") or (runs[0] if runs else None)
        artifacts = service.run_artifacts(experiment_name, selected_run) if selected_run else {"plot_files": [], "design_files": [], "param_files": [], "experiment_files": []}

        return render_template(
            "experiment.html",
            experiment_name=experiment_name,
            runs=runs,
            selected_run=selected_run,
            artifacts=artifacts,
        )

    @app.get("/artifacts/<path:subpath>")
    def artifacts(subpath: str):
        experiments_root = workspace_root / "Experiments"
        target = (experiments_root / subpath).resolve()

        if experiments_root not in target.parents and target != experiments_root:
            abort(404)
        if not target.exists() or not target.is_file():
            abort(404)

        return send_file(target)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
