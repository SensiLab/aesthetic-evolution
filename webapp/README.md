# Web UI (Additive)

This webserver is implemented as an additive layer and does **not** modify existing files like `run.py`.

## What this page is for
The webpage gives you a browser-based way to run the same evolution workflow you normally run from `run.py`, while keeping your existing scripts intact.

Use it to:
- Launch new aesthetic evolution jobs with custom parameters.
- Watch job status and logs while a run is executing.
- Open completed experiments and inspect generations.
- View ranking plot images, generated design images, and parameter JSON files.
- Save the run configuration into the experiment output folder.

## Features
- Submit evolution jobs with the same key parameters as `run.py`
- Execute jobs in-process using `aesthetic_evolution()` (no subprocess call to `run.py`)
- One active job at a time via background worker queue
- Browse experiments, generations, ranking plot images, design images, and params JSON
- Save run config snapshot into `Experiments/{experiment}/experiment_config.yaml`

## Start the web UI
1. Install Flask if needed:
   - `pip install flask`
2. Start the server from repo root:
   - `python run_web.py`
3. Open in browser:
   - `http://127.0.0.1:8000`

## Page walkthrough

### 1) Home page (`/`)
The home page has three sections:

- **Run New Job**: Form for creating a run.
- **Recent Jobs**: Status list of recently submitted jobs.
- **Experiments**: Links to existing experiment folders under `Experiments/`.

### 2) Run New Job form
Fill in the form and click **Submit Job**.

Important fields:
- **Experiment Name**: Name of the output folder under `Experiments/`.
- **Runs**: Number of generations to execute.
- **Population Size (even)**: Must be an even integer.
- **Param Spec File**: Usually `config/param_spec.yaml`.
- **Sketch Directory**: Absolute or repo-relative path to your Processing sketch.
- **Prompt Filepath** or **Prompt Text**: provide one (prompt text overrides filepath when present).
- **Processing**: `serial` or `parallel`.
- **Workers**: Parallel worker count (used if processing is parallel).
- **Alpha Mode / Alpha**:
  - `random`: sample alpha each crossover.
  - `fixed`: use provided alpha (0 to 1).
  - `biased`: alpha derived from parent scores.
- **Mutation Rate / Mutation Sigma**: Evolution mutation controls.
- **Tournament k ratio (0..1)**: Fraction converted internally to tournament size.
- **Ranking Method**: `glicko`, `simple`, or `CLIP-IQA`.
- **Screen**: Enable if Processing can access a display.
- **Parents Compete / Competing Parents Rate**: Carry over top parents if enabled.
- **Overwrite Existing Experiment**: Required if the experiment name already exists.

Validation behavior:
- The form is server-validated before queueing.
- If values are invalid (missing paths, bad ranges, odd population size, etc.), submission returns an error.

### 3) Job status page (`/jobs/{job_id}`)
After submit, you are redirected to a job page that shows:
- Current status (`queued`, `running`, `completed`, `failed`).
- Timestamps (created, started, finished).
- Any error message.
- Live log output (auto-refreshes every few seconds).
- Link to open the related experiment.

Execution model:
- One active job runs at a time.
- Additional jobs are queued automatically.

### 4) Experiment browser (`/experiments/{experiment_name}`)
Use this page to inspect outputs after or during execution.

What you can do:
- Select a run (`run0`, `run1`, ...).
- View **Ranking & Generation Plot** images (e.g., population design/ranking/selection plots).
- Open generated design images.
- Open parameter JSON files for each design.

## Output files and where they are saved

For each experiment, outputs are written to:
- `Experiments/{experiment_name}/runN/Images/*.png`
- `Experiments/{experiment_name}/runN/Params/*.json`
- `Experiments/{experiment_name}/experiment_config.yaml`

If prompt text was entered directly in the form, the UI also saves:
- `Experiments/{experiment_name}/web_prompt.txt`

Web UI runtime metadata is stored in:
- `webapp/jobs/{job_id}.json`
- `webapp/jobs/{job_id}.log`

## Typical user workflow
1. Open `/`.
2. Fill and submit **Run New Job**.
3. Watch logs on `/jobs/{job_id}`.
4. Open `/experiments/{experiment_name}`.
5. Switch runs and inspect ranking plots + params.

## Troubleshooting
- **Import error for Flask**: install Flask in your active Python environment.
- **Sketch path not found**: check `Sketch Directory` path is valid on disk.
- **Prompt file not found**: verify `Prompt Filepath` exists, or paste prompt into `Prompt Text`.
- **Experiment already exists**: enable **Overwrite Existing Experiment** if replacement is intended.
- **Long runtime**: expected for larger populations/runs and model-based ranking.

## Notes
- Existing Processing and model runtime dependencies still apply.
- This UI writes only new files under `webapp/` and normal experiment outputs under `Experiments/`.
