# Aesthetic Evolution

An aesthetic evolution system that uses genetic algorithms to evolve generative art designs. The system generates visual designs via Processing sketches, ranks them using a vision language model, and evolves parameters through tournament selection, crossover, and mutation.

## Quick Start

```bash
# 1. Copy and configure environment
cp .env-example .env
# Edit .env: set HF_CACHE_DIR to your local HuggingFace hub cache path

# 2. Set your sketch path and experiment name
#    Edit config/experiment_config.yaml → job.sketch_dir and job.experiment_name

# 3. Run
python run.py
```

See [Prerequisites](#prerequisites) and [Configuration](#configuration) for full details.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
   - [Local](#local)
   - [Docker](#docker)
3. [Configuration](#configuration)
4. [Usage](#usage)
   - [Running an Experiment](#running-an-experiment)
   - [Web UI](#running-with-the-web-app)
   - [Generating a Benchmark Dataset](#generating-a-benchmark-dataset)
   - [Pairwise Voting App](#running-the-standalone-pairwise-voting-app)
   - [Benchmarking / Evaluation](#benchmarking-evaluating-model-rankings-against-human-labels)
5. [Output Structure](#output-structure)
6. [Parameter Specification](#parameter-specification)
7. [Architecture Overview](#architecture-overview)
8. [Performance Notes](#performance-notes)
9. [Troubleshooting](#troubleshooting)
10. [Extending](#extending)
11. [Authors / License](#authors)

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.13+** | Required for all local runs |
| **CUDA 12.8** | Required by PyTorch 2.7.0+cu128 and Flash Attention 2.7.4 |
| **NVIDIA GPU (Ampere or newer)** | Flash Attention 2 requires sm_80+; tested on RTX 5090 |
| **~16 GB disk** | For the Qwen3-VL-8B-Instruct model weights |
| **Processing 4.5.3** | External binary; `processing-java` must be on your PATH |
| **xvfb** | Required for headless rendering on Linux (`screen: False`) |
| **nvidia-container-toolkit** | Docker only — enables GPU access inside containers |

---

## Installation

### Local

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd aesthetic-evolution
   ```

2. **Install Processing 4.5.3**:
   - Download from [processing.org/download](https://processing.org/download)
   - Ensure `processing-java` is on your PATH (e.g. symlink it to `/usr/local/bin/`)
   - Set up your Processing sketch directory (e.g. `~/Harmonograph`)

3. **Install xvfb** (Linux headless rendering):
   ```bash
   sudo apt-get install xvfb
   ```

4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   > **Flash Attention note**: `requirements.txt` includes a pre-built wheel targeting RTX 5090 / CUDA 12.8. If you have a different GPU or CUDA version, build from source or download a matching wheel from the [Flash Attention releases page](https://github.com/Dao-AILab/flash-attention/releases).

5. **Download the Qwen3-VL model** (first run only):
   The model (`Qwen/Qwen3-VL-8B-Instruct`) downloads automatically from HuggingFace on first run, provided `HF_CACHE_DIR` is set or a default cache is available. Requires ~16 GB disk space.

### Docker

Docker builds a self-contained image with Python 3.13, Processing 4.5.3, CUDA 12.8, and all dependencies. The container runs the model **offline** — the model must be pre-downloaded to your host cache before starting.

> **Note**: Docker support is newly added and still being hardened. Some edge cases may require manual workarounds.

```bash
# 1. Install nvidia-container-toolkit on the host
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 2. Configure your HuggingFace cache path
cp .env-example .env
# Open .env and set HF_CACHE_DIR to your local HuggingFace cache directory
# e.g. HF_CACHE_DIR=/home/user/.cache/huggingface/hub

# 3. Pre-download the model (required — container runs with HF_HUB_OFFLINE=1)
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct

# 4. Set sketch_dir to the in-container path in config/experiment_config.yaml:
#   sketch_dir: /app/Harmonograph

# 5. Start services
docker compose up webapp      # Web UI at http://localhost:5000
docker compose up pairwise    # Pairwise voting at http://localhost:5050
docker compose up             # Both services
```

**Volume mounts** (configured in `docker-compose.yml`):

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `./Experiments` | `/app/Experiments` | Experiment outputs |
| `./Data` | `/app/Data` | Benchmark datasets |
| `./config` | `/app/config` | YAML configs and prompt files |
| `../Harmonograph` | `/app/Harmonograph` | Processing sketch source |
| `${HF_CACHE_DIR}` | `/hf-hub-cache` | HuggingFace model cache |

---

## Configuration

### Environment Variables

Copy `.env-example` to `.env` before running Docker:

```
HF_CACHE_DIR=/path/to/your/huggingface/hub/cache
```

This is only required for Docker. For local runs the standard HuggingFace cache location (`~/.cache/huggingface/`) is used automatically.

### experiment_config.yaml

The main config for `run.py` and the web UI. Located at `config/experiment_config.yaml`.

#### Job Section

Settings for Processing sketch execution and the experiment environment:

```yaml
job:
  experiment_name: "experiment"             # Output folder name under Experiments/
  param_spec_file: "config/param_spec.yaml" # Parameter schema file
  sketch_dir: "/path/to/Harmonograph"       # Absolute path to Processing sketch directory
  prompt_filepath: "config/prompt.txt"      # LLM evaluation prompt
  processing: "parallel"                    # "serial" or "parallel"
  screen: false                             # false = headless via xvfb; true = use display
  workers: 8                                # Worker count for parallel generation
```

> **Docker**: set `sketch_dir: /app/Harmonograph` (the in-container mount path).

#### Evolution Section

Genetic algorithm parameters:

```yaml
evo:
  alpha_mode: "biased"          # Crossover weighting: "fixed", "random", or "biased"
  alpha: 0.5                    # Fixed alpha (used only when alpha_mode="fixed")
  runs: 5                       # Number of generations
  population_size: 20           # Must be even
  mutation_rate: 0.1            # Per-parameter mutation probability
  mutation_sigma: 0.1           # Gaussian mutation standard deviation
  parents_compete: false        # Carry top-ranked parents into next generation unchanged
  competing_parents_rate: 0.1   # Fraction of population reserved for elite parents
  k: 0.25                       # Tournament size as fraction of population
  ranking_method: "glicko"      # "glicko", "CLIP-IQA", or "simple"
```

**Alpha mode options**:
- `"fixed"`: Uses the `alpha` value for every crossover.
- `"random"`: Samples alpha uniformly for each crossover.
- `"biased"`: Weights alpha toward the higher-ranked parent.

**Ranking method options**:
- `"glicko"` (recommended): Pairwise LLM comparisons with Glicko-2 rating. Robust but O(N²) comparisons.
- `"CLIP-IQA"` (fast): Direct quality scoring via OpenCLIP — no pairwise comparisons, O(N). Based on [CLIP-IQA paper](https://arxiv.org/abs/2207.12396).
- `"simple"`: Pairwise LLM comparisons with basic win/loss counting.

**Parent competition options**:
- `parents_compete: false`: Entire next population comes from breeding + mutation.
- `parents_compete: true`: Top `competing_parents_rate` fraction copied directly into next generation before breeding fills the rest.

### generation_config.yaml

Lightweight config for `generate.py` (standalone design generation without evolution):

```yaml
param_spec_file: "config/param_spec.yaml"
sketch_dir: "/path/to/Harmonograph"
processing: parallel
screen: false
workers: 8
population_size: 100
```

### LLM Prompt Files

The prompt used for pairwise evaluation is stored in a plain text file specified by `prompt_filepath`. Several examples are included:

| File | Purpose |
|------|---------|
| `config/prompt.txt` | Default aesthetic evaluation prompt |
| `config/jon_prompt.txt` | Prompt matching the human-labelled benchmark |
| `config/butterfly_prompt.txt` | Butterfly-themed aesthetic evaluation |
| `config/reasoning_prompt.txt` | Chain-of-thought reasoning before final answer |

The prompt must instruct the model to output only `"1"` or `"2"` when using Glicko or Simple ranking.

Example (`config/prompt.txt`):
```text
You will be given two images.

IMPORTANT RULES (must be followed):
- Images that are mostly dark blobs or solid dark regions MUST be ranked lower.
- Visible line structure and repeating patterns are REQUIRED for a high score.
- Messy noise or amorphous shapes should be ranked lower.

Task:
Choose which image is more aesthetically pleasing according to the rules above.
Output ONLY '1' or '2'.
```

---

## Usage

### Running an Experiment

1. **Edit `config/experiment_config.yaml`**:
   - Set `job.sketch_dir` to your Processing sketch path.
   - Set `job.experiment_name` to a unique identifier for this run.
   - Tune evolutionary parameters in the `evo` section as needed.

2. **Run**:
   ```bash
   python run.py
   ```

   The script loads `config/experiment_config.yaml`, checks if the experiment already exists (and prompts before overwriting), then runs the full evolutionary loop.

3. **Monitor output**:
   - Progress bars show generation and evaluation progress.
   - Ranked image grids are saved each generation to `Experiments/{name}/run{N}/Images/`.
   - Parameter JSONs are saved to `Experiments/{name}/run{N}/Params/`.
   - A copy of the config is saved to `Experiments/{name}/experiment_config.yaml`.

### Running with the Web App

The web UI runs the same `aesthetic_evolution()` pipeline in-process without modifying `run.py`.

1. **Start the server**:
   ```bash
   python run_web.py   # binds to http://127.0.0.1:8000
   ```

2. **Submit a job** at `http://127.0.0.1:8000`:
   - Fill in the same fields as `experiment_config.yaml`.
   - Enable **Overwrite Existing Experiment** only if intentionally replacing an existing run.

3. **Track execution**:
   - The jobs page shows `queued` / `running` / `completed` / `failed` status with live log output.

4. **Browse outputs**:
   - Navigate to the experiment page to view generation images, ranking plots, and parameter JSON files.

For full web UI details see [webapp/README.md](webapp/README.md).

### Generating a Benchmark Dataset

`generate.py` creates a population of random designs without running the evolutionary loop — useful for building a fixed benchmark for evaluation.

1. **Edit `config/generation_config.yaml`** (set `sketch_dir`, `population_size`, etc.).

2. **Run**:
   ```bash
   python generate.py
   ```

   Output is written to `Data/benchmark/Images/` and `Data/benchmark/Params/`.

### Running the Standalone Pairwise Voting App

A browser-based UI for collecting human pairwise preference labels. Separate from the experiment web UI.

1. **Start the server**:
   ```bash
   python run_pairwise_coverage.py --n 3 --data-dir Data/curated/Images
   ```

   | Flag | Default | Description |
   |------|---------|-------------|
   | `--n` | *(required)* | Minimum times each image must appear globally |
   | `--data-dir` | `Data/curated/Images` | Directory of images to compare |
   | `--state-dir` | `pairwise_coverage_app/state` | Directory for persistent CSV and state files |
   | `--host` | `0.0.0.0` | Bind host |
   | `--port` | `5050` | Bind port |
   | `--debug` | off | Enable Flask debug mode |

2. **Open** `http://127.0.0.1:5050/compare`.

3. **Behavior guarantees**:
   - Unordered pairs are never repeated.
   - Scheduling continues until every image has appeared at least `n` times, or the target is marked impossible.
   - Coverage and no-repeat state persist across restarts via state files in `--state-dir`.

4. **API endpoints**:

   | Endpoint | Method | Description |
   |----------|--------|-------------|
   | `/compare` | GET | Main voting UI |
   | `/vote` | POST | Submit a vote (`image_a`, `image_b`, `outcome`) |
   | `/undo` | POST | Undo the most recent vote for the current session |
   | `/status` | GET | JSON status: counts, deficits, can_undo |
   | `/progress` | GET | Glicko convergence metrics: `estimated_comparisons_left`, `average_deviation`, `max_deviation` |
   | `/images/<image_id>` | GET | Serve an image by ID |

5. **Output files**:
   - `pairwise_coverage_app/state/labels.csv` — pairwise outcomes
   - `pairwise_coverage_app/state/global_state.json` — coverage state snapshot

### Benchmarking: Evaluating Model Rankings Against Human Labels

`evaluate_scores.py` compares a VLM's aesthetic rankings against a human-labelled ground truth. It is also imported by the pairwise coverage app to compute live Glicko convergence metrics.

#### Key Functions

| Function | Description |
|----------|-------------|
| `calc_scores(scores_path, plot)` | Computes Glicko ratings from a pairwise CSV (`image_a`, `image_b`, `outcome`). Returns players sorted by rating. |
| `calc_scores_sampled(scores_path, n, plot, seed)` | Same as above but on a random subset where each design appears in at most `n` comparisons. |
| `compare_rankings(ranking_a, ranking_b, k_values)` | Spearman rho, Kendall tau, and top-k Jaccard between two ordered ID lists. |
| `model_predictions(model_name, prompt_filepath, benchmark_dir, ...)` | Runs the VLM over all pairs and writes predictions to CSV. |
| `CLIP_IQA_Score(image_dir, positive_prompt, negative_prompt)` | Ranks images using CLIP-IQA and returns ordered IDs. |
| `get_structural_complexity(img_path)` | Returns a scalar structural complexity score for a single image. |

#### CLI Usage

```bash
python evaluate_scores.py \
    --benchmark_dir Data/curated/Images \
    --benchmark_scores_csv jon-ranking/labels.csv \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --prompt_filepath config/prompt.txt \
    --model_predictions_csv model_predictions.csv \
    --n 50
```

The script generates model predictions, computes Glicko scores for both human and model rankings, then reports averaged Spearman rho, Kendall tau, and top-5/10/20 Jaccard over 1000 sampled subsets.

#### CSV Format

Both human labels and model predictions use the same format:

```
image_a,image_b,outcome
run_1.png,run_2.png,first    # first image wins
run_3.png,run_4.png,second   # second image wins
run_5.png,run_6.png,draw     # draw
```

The 5-column format with `timestamp,session_id` prefix produced by the pairwise coverage app is also accepted automatically.

---

## Output Structure

```
Experiments/
  {experiment_name}/
    experiment_config.yaml     # Copy of the config used for this run
    run0/                      # Initial (random) generation
      Images/
        run0_0.png
        run0_1.png
        Rankings.png           # Ranked grid visualisation
        ...
      Params/
        run0_0.json            # {"parameters": {...}, "filepath": "..."}
        run0_1.json
        ...
    run1/                      # Generation 1 (after first evaluation + evolution)
      Images/
      Params/
    ...
```

---

## Parameter Specification

Parameters are defined in `config/param_spec.yaml`:

```yaml
parameter_name:
  type: "int" | "float" | "categorical"
  min: <value>          # For int/float
  max: <value>          # For int/float
  values: [...]         # For categorical
  calc: true | false    # If true, derived from other parameters (excluded from mutation/crossover)
```

**Calculated parameters** (derived automatically, not mutated):
- `dt`: `1.0 / (fb * max(frx, fry * sampleRate))`
- `spaceP`: `10^spaceD`

---

## Architecture Overview

### Core Classes

#### `Params` (inherits `Player`)
Genotype storage and genetic operations.

- Stores parameter dictionaries categorised by type (int, float, categorical)
- Inherits `Player` to track Glicko-2 rating and rating deviation (RD)
- `breed(other, alpha, child_name)` — arithmetic mean crossover for numerics; random selection for categoricals
- `mutate()` — Gaussian noise on normalised parameter space, then clip and denormalise
- `write_json(directory, filename, pop_name)` — serialises params to JSON for Processing consumption

#### `DesignGenerator`
Orchestrates Processing sketch execution.

- Manages `Experiments/{name}/run{N}/Images|Params/` folder structure
- Spawns Processing subprocesses; wraps with `xvfb-run` when `screen=False`
- Supports serial or parallel generation via `ProcessPoolExecutor`
- 120s timeout per render to prevent hangs; stderr/stdout captured for debugging
- Subprocess call: `processing-java --sketch={sketch_dir} --run {params_json_path}`

#### `DesignEvolver`
Manages the evolutionary loop and LLM-based evaluation.

- Initialises random population from `param_spec.yaml`
- Coordinates LLM batch evaluation via `Qwen3VLBatchProcessor`
- Updates Glicko ratings based on pairwise comparison outcomes
- Implements tournament selection with rank-based probability sampling
- Orchestrates breeding and mutation to produce the next generation

#### `Qwen3VLBatchProcessor`
Handles batch inference with Qwen3-VL-8B-Instruct.

- `process_batch_chunked()` — processes comparisons in chunks (default `chunk_size=32`) to avoid OOM
- Uses Flash Attention 2 and bfloat16 for memory efficiency

#### `CLIP_IQA`
OpenCLIP-based image quality assessment (used when `ranking_method: CLIP-IQA`).

- RN50 model; compares image features against positive/negative text prompts
- Returns P(good) and P(bad) per image; ranks by P(good)

#### `Player` / Glicko-2
Glicko-2 rating system — tracks rating and RD per design. `update_rating()` applies incremental updates after each chunk of comparisons.

### Evolutionary Loop

```
for each generation:
  1. Evaluate: build all-pairs jobs → batch LLM inference → update ratings → sort population
  2. Evolve:   sample parent pairs by rating probability → breed → mutate → generate new designs
```

For CLIP-IQA, step 1 loads all images and scores them against text prompts (no pairwise comparisons).

---

## Performance Notes

| Parameter | Guidance |
|-----------|---------|
| **GPU memory** | Default `chunk_size=32` for batch inference; reduce if OOM (Glicko/Simple only) |
| **Generation speed** | `parallel` mode is significantly faster; tune `workers` to your CPU core count |
| **Render timeout** | 120s per design; increase in code if your sketch legitimately takes longer |
| **Selection pressure** | Lower `k` → more elitism; higher `k` → more diversity |
| **Parent competition** | Higher `competing_parents_rate` → more exploitation, less diversity |
| **Mutation** | `mutation_rate` controls per-parameter probability; `mutation_sigma` controls magnitude |
| **Ranking speed** | CLIP-IQA is ~100x faster than pairwise methods for large populations (O(N) vs O(N²)) |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `processing-java: command not found` | Processing not on PATH | Add the Processing install directory to PATH, or symlink `processing-java` to `/usr/local/bin/` |
| Processing renders hang or time out | Sketch has an infinite loop, or no display available | Set `screen: false` to use xvfb; verify the sketch runs correctly outside this system |
| CUDA out of memory | Batch size too large | Reduce `chunk_size` in `Qwen3VLBatchProcessor` (default 32) |
| `ModuleNotFoundError: flash_attn` | Flash Attention wheel mismatch for your GPU/CUDA | Build Flash Attention from source, or download a matching wheel from the [releases page](https://github.com/Dao-AILab/flash-attention/releases) |
| Docker: model not found / HF offline error | Model not pre-downloaded before container start | Run `huggingface-cli download Qwen/Qwen3-VL-8B-Instruct` on the host, then restart |
| Docker: GPU unavailable inside container | nvidia-container-toolkit not installed or Docker not restarted | `sudo apt-get install nvidia-container-toolkit && sudo systemctl restart docker` |
| Experiment already exists prompt | Reusing an existing experiment name | Enable **Overwrite Existing Experiment** in the web UI, or delete `Experiments/{name}/` manually |
| Odd population size error | `population_size` must be even for breeding pairs | Set `population_size` to an even number in config |
| LLM outputs something other than "1" or "2" | Prompt not constraining output format | Ensure your prompt explicitly instructs the model to output only `1` or `2`; consider using `config/prompt.txt` as a template |

---

## Extending

- **New parameter types**: Add entries to `config/param_spec.yaml` and update type categorisation in `Params.__init__`.
- **Custom selection**: Modify `evolve_population()` in `generate_designs.py`.
- **Multi-objective optimisation**: Extend `calc_ranks()` in `utils.py` for multiple fitness criteria.
- **Checkpointing**: Add serialisation of `population_params` between runs (not currently implemented).

---

## Citation

If you use this work, please cite:

```bibtex
@article{krol2026evolving,
  title={Evolving to the Aesthetics of a Vision-Language Model},
  author={Krol, Stephen James and McCormack, Jon},
  journal={arXiv preprint arXiv:2606.00112},
  year={2026}
}
```

## Authors

- Stephen Krol (@sjkro1) — January 2026

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE.txt) (CC BY-NC-SA 4.0).

You are free to:
- **Share**: Copy and redistribute the material in any medium or format
- **Adapt**: Remix, transform, and build upon the material

Under the following terms:
- **Attribution**: You must give appropriate credit
- **NonCommercial**: You may not use the material for commercial purposes
- **ShareAlike**: Derivatives must use the same license

See [LICENSE.txt](LICENSE.txt) for the full license text.
