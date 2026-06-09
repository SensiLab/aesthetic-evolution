# Aesthetic Evolution

An aesthetic evolution system that uses genetic algorithms to evolve generative art designs. The system generates visual designs via Processing sketches, ranks them using the Qwen3-VL vision language model with Glicko-2 rating system, and evolves parameters through tournament selection, crossover, and mutation.

## Project Overview

This project implements an evolutionary algorithm for generative art creation. The core pipeline operates as follows:

1. **Generate**: Create a population of designs using Processing sketches with randomized parameters
2. **Evaluate**: Rank designs using GPU-accelerated batch inference with Qwen3-VL vision LLM
3. **Select**: Use tournament selection to choose high-performing parents
4. **Breed**: Create offspring through arithmetic mean crossover
5. **Mutate**: Apply Gaussian mutations to parameters
6. **Repeat**: Iterate through multiple generations

The system supports both serial and parallel design generation, headless rendering via `xvfb`, and efficient GPU batch processing with Flash Attention 2 for memory optimization.

### Key Features

- **Genetic Algorithm**: Tournament selection, arithmetic mean crossover, and Gaussian mutation
- **Parent Elitism Option**: Optionally retain top-ranked parents in the next generation using `parents_compete` and `competing_parents_rate`
- **Multiple Ranking Systems**: 
  - **Glicko-2**: Probabilistic ranking using Glicko-2 algorithm for robust pairwise comparison evaluation
  - **CLIP-IQA**: Vision-language model-based quality assessment without pairwise comparisons
  - **Simple**: Basic win/loss counting from pairwise comparisons
- **Vision-based Ranking**: Automated aesthetic evaluation using Qwen3-VL-8B-Instruct with pairwise comparisons (for Glicko/Simple) or CLIP for direct quality scoring (CLIP-IQA)
- **Flexible Parameters**: YAML-based parameter specification with int, float, and categorical types
- **Parallel Processing**: Multi-worker design generation and chunked GPU batch inference
- **Headless Rendering**: Support for server environments without displays
- **Web Interface (Additive)**: Browser UI for job submission, live logs, and generation/result browsing without changing `run.py`
- **Standalone Design Generation**: `generate.py` generates a random population without running evolution, useful for building benchmark datasets
- **Benchmarking and Evaluation**: `evaluate_scores.py` compares model rankings against human rankings using Glicko scoring, sampled evaluation, Spearman/Kendall correlation, and top-k Jaccard similarity

## Installation and Setup

### Requirements

- **Python**: 3.13+ (based on requirements)
- **CUDA**: 12.8 (required for GPU acceleration)
- **Processing**: External installation required for sketch rendering
- **xvfb**: For headless rendering on Linux servers

### CUDA Version

This project requires **CUDA 12.8**. The dependencies include:

- PyTorch 2.7.0+cu128
- Flash Attention 2.7.4 (custom wheel for RTX 5090, CUDA 12.8)
- NVIDIA CUDA Runtime 12.8.57
- NVIDIA CUDA NVRTC 12.8.61

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd aesthetic-evolution
   ```

2. **Install Processing**:
   - Download and install Processing from [processing.org](https://processing.org/download)
   - Ensure `processing-java` is available in your PATH
   - Set up your Processing sketch directory (e.g., `/home/user/Harmonograph`)

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: The `requirements.txt` includes a local Flash Attention wheel. You may need to:
   - Build Flash Attention from source for your GPU/CUDA version, or
   - Download a compatible pre-built wheel from the [Flash Attention releases](https://github.com/Dao-AILab/flash-attention/releases)

4. **Install xvfb** (for headless rendering):
   ```bash
   sudo apt-get install xvfb  # Ubuntu/Debian
   ```

5. **Download the Qwen3-VL model**:
   The model (`Qwen/Qwen3-VL-8B-Instruct`) will be automatically downloaded from HuggingFace on first run. Ensure you have:
   - Sufficient disk space (~16GB for the model)
   - HuggingFace token configured if needed for gated models

## Architecture Overview

### Three Main Classes

#### 1. `Params` Class (inherits from `Player`)
**Purpose**: Genotype storage, genetic operations, and Glicko rating tracking

**Key Responsibilities**:
- Inherit from `Player` class to store Glicko rating and rating deviation (RD)
- Store parameter dictionaries with type-aware categorization (int, float, categorical)
- Handle breeding via arithmetic mean crossover with configurable alpha weighting
- Implement Gaussian mutation (σ=0.1) on normalized parameter space
- Manage calculated parameters (`dt`, `spaceP`) that are derived from other params
- Write parameter JSON files for Processing consumption

**Key Methods**:
- `breed(other, alpha, child_name)`: Creates offspring using weighted arithmetic mean for numeric params and random selection for categorical
- `mutate()`: Applies Gaussian noise to normalized parameters, then clips and denormalizes
- `write_json(directory, filename, pop_name)`: Serializes parameters to JSON with filepath metadata

#### 2. `DesignGenerator` Class
**Purpose**: Orchestrate Processing sketch execution to render designs

**Key Responsibilities**:
- Manage experiment folder structure (`Experiments/{name}/run{N}/Images|Params/`)
- Spawn Processing subprocesses (with optional xvfb wrapper for headless mode)
- Support serial or parallel design generation via `ProcessPoolExecutor`
- Handle timeouts (20s limit) to prevent stuck renders
- Write parameter JSON files before invoking Processing

**Key Methods**:
- `generate_population(pop_name, params)`: Batch generates all designs for a population
- `generate_image(jobs)`: Static method that executes `processing-java --sketch={dir} --run {params.json}`
- `_initialise_design(name)`: Creates directory structure for a generation

**Processing Integration**:
- Parallel mode: Uses `ProcessPoolExecutor` with configurable worker count
- Headless mode: Wraps command with `xvfb-run -a --server-args=-screen 0 1024x768x24`
- Subprocess call: `processing-java --sketch={sketch_dir} --run {params_json_path}`

#### 3. `DesignEvolver` Class
**Purpose**: Manage the evolutionary loop and LLM-based evaluation

**Key Responsibilities**:
- Initialize random population parameters from `param_spec.yaml`
- Coordinate LLM batch evaluation via `Qwen3VLBatchProcessor`- Update Glicko ratings based on pairwise comparison outcomes- Implement tournament selection with rank-based probability sampling
- Orchestrate breeding and mutation to create next generation
- Track population state across generations

**Key Methods**:
- `_generate_initial_params()`: Creates random initial population respecting parameter ranges
- `evaluate_population(plot)`: Builds pairwise comparison jobs, processes via LLM, updates Glicko ratings incrementally
- `_process_batch_chunked(jobs, chunk_size)`: Processes comparisons in chunks, updates Glicko scores after each chunk
- `evolve_population(generator, plot)`: Samples parents by Glicko rating probability, breeds children, applies mutation, generates new designs
- `_calc_ranking_probabilities(n, k)`: Computes selection probabilities based on tournament size k

**Ranking Systems**:

*For Glicko and Simple methods:*
- Uses `Qwen3VLBatchProcessor` with chunked batch inference (default chunk_size=32)
- Processes pairwise comparisons:
  - **Glicko**: Updates ratings incrementally after each chunk; each design starts with rating=1500, deviation=350
  - **Simple**: Counts wins/losses from all pairwise comparisons
- Ratings converge as more comparisons are processed (Glicko only), with uncertainty (RD) decreasing over time
- Expects LLM to output only "1" or "2" for each comparison

*For CLIP-IQA method:*
- Uses OpenCLIP (RN50) for direct image quality assessment
- No pairwise comparisons needed - evaluates each image independently
- Compares image features against "Good Design" vs "Bad Design" text prompts
- Returns probability scores: P(good) and P(bad) for each image
- Based on [Exploring CLIP for Assessing the Look and Feel of Images](https://arxiv.org/abs/2207.12396)
- Significantly faster than pairwise methods (O(N) vs O(N²) comparisons)

### Generation Workflow in `run.py`

The `run.py` script orchestrates the complete evolutionary pipeline. It loads configuration from `config/experiment_config.yaml` and calls the `aesthetic_evolution()` function from `generate_designs.py`.

**Workflow Steps**:

1. **Initialize Generator**:
   - Create `DesignGenerator` with experiment name, Processing sketch directory, and execution mode (serial/parallel)
   - Configure headless rendering and worker count

2. **Initialize Evolver**:
   - Create `DesignEvolver` with parameter spec, LLM prompt, population size, and tournament size (k=5)
   - Generate initial random population parameters
   - Load Qwen3-VL model onto GPU

3. **Generate Initial Population** (Generation 0):
   - `generator.generate_population()` renders all designs from initial random parameters
   - Writes JSON files and executes Processing sketches
   - Saves images to `Experiments/{name}/run0/Images/`

4. **Evolutionary Loop** (for each generation 1 to `runs`):
   ```python
   for run in range(1, runs):
       evolver.evaluate_population(plot=True)     # Step A: Evaluate
       evolver.evolve_population(generator, True)  # Step B: Evolve & Generate
   ```

   **Step A - Evaluation**:
   
   *For Glicko/Simple methods:*
   - Build all-pairs pairwise comparison jobs (N×(N-1)/2 comparisons for population size N)
   - Process comparisons in GPU batches using Qwen3-VL
   - Update Glicko ratings incrementally after each chunk (or count wins/losses for simple ranking)
   - Ratings reflect both wins/losses and uncertainty (rating deviation) for Glicko
   - Sort population by rating/score (best to worst)
   
   *For CLIP-IQA method:*
   - Load all images and compute CLIP embeddings
   - Compare against positive ("Good Design") and negative ("Bad Design") text prompts
   - Rank by P(good) probability score
   - Much faster: O(N) evaluations vs O(N²) pairwise comparisons
   
   - Plot ranked image grid

   **Step B - Evolution**:
   - Optionally retain top-ranked parents unchanged in next generation (`parents_compete=True`)
   - Number retained = `int(competing_parents_rate * population_size)`
   - Sample parent pairs using Glicko rating-based probability (higher rating → higher selection chance)
   - Breed offspring via arithmetic mean crossover (α determined by `alpha_mode`)
   - Apply mutation to offspring parameters
   - Reset offspring Glicko ratings to initial values (1500 rating, 350 RD)
   - Increment generation counter
   - Generate new population via `generator.generate_population()`

5. **Final Evaluation**:
   - After last generation, evaluate final population to rank results

### Configuration

Two configuration files are used:

- **[config/experiment_config.yaml](config/experiment_config.yaml)**: Full evolutionary experiment configuration (used by `run.py` and `run_web.py`)
- **[config/generation_config.yaml](config/generation_config.yaml)**: Lightweight config for standalone design generation (used by `generate.py`); only needs `param_spec_file`, `sketch_dir`, `processing`, `screen`, `workers`, and `population_size`

`experiment_config.yaml` is organized into two sections:

#### Job Configuration

The `job` section contains settings related to the Processing sketch and execution environment:

```yaml
job:
  experiment_name: "experiment"                  # Experiment identifier
  param_spec_file: "config/param_spec.yaml"      # Parameter schema file
  sketch_dir: "/path/to/Harmonograph"            # Processing sketch directory
  prompt_filepath: "config/prompt.txt"           # Path to LLM prompt file
  processing: "parallel"                          # "serial" or "parallel"
  screen: False                                   # Use xvfb for headless rendering
  workers: 8                                      # Parallel workers
```

#### Evolution Configuration

The `evo` section contains evolutionary algorithm parameters:

```yaml
evo:
  alpha_mode: "biased"        # Crossover mode: "fixed", "random", or "biased"
  alpha: 0.5                  # Fixed alpha value (used when alpha_mode="fixed")
  runs: 5                     # Number of generations
  population_size: 20         # Population size (must be even)
  mutation_rate: 0.1          # Probability of each parameter mutating
  mutation_sigma: 0.1         # Standard deviation for Gaussian mutation
   parents_compete: false      # Keep top parents in next generation unchanged
   competing_parents_rate: 0.1 # Fraction of population reserved for competing parents
  k: 0.25                     # Percentage of population in tournament selection
  ranking_method: "glicko"    # Ranking method: "glicko", "CLIP-IQA", or "simple"
```

**Note**: When using `ranking_method: "CLIP-IQA"`, the system uses OpenCLIP RN50 for direct quality assessment instead of LLM-based pairwise comparisons. This is significantly faster (O(N) vs O(N²)) but evaluates images independently rather than through comparative ranking.

**Alpha Mode Options**:
- `"fixed"`: Uses the specified `alpha` value for all crossovers
- `"random"`: Randomly samples alpha for each crossover
- `"biased"`: Biases alpha toward higher-ranked parent

**Parent Competition Options**:
- `parents_compete: false`: Entire next population is produced from sampled breeding + mutation.
- `parents_compete: true`: Top-ranked parents are copied directly into next generation before breeding.
- `competing_parents_rate`: Proportion of population copied as elite parents when `parents_compete` is enabled.

**Ranking Method Options**:
- `"glicko"` (recommended for robust ranking): Uses Glicko-2 rating system with incremental updates, accounts for rating uncertainty via pairwise comparisons
- `"CLIP-IQA"` (recommended for speed): Uses CLIP vision-language model to directly score image quality without pairwise comparisons; based on [CLIP-IQA paper](https://arxiv.org/abs/2207.12396)
- `"simple"`: Uses basic win/loss counting from pairwise comparisons

**LLM Prompt File**:

The prompt is stored in a separate file specified by `prompt_filepath`. Several example prompts are included in `config/`:

| File | Purpose |
|------|---------|
| `config/prompt.txt` | Default aesthetic evaluation prompt |
| `config/jon_prompt.txt` | Prompt used for the human-labelled benchmark dataset |
| `config/butterfly_prompt.txt` | Butterfly-themed aesthetic prompt |
| `config/reasoning_prompt.txt` | Prompt that encourages chain-of-thought reasoning before the final answer |

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

## Usage

### Running an Experiment

1. **Configure parameters**:
   - Edit [config/experiment_config.yaml](config/experiment_config.yaml):
     - **Job section**: Set `sketch_dir` to your Processing sketch path, `experiment_name`, and execution settings
     - **Evo section**: Configure evolutionary parameters (`population_size`, `runs`, `mutation_rate`, `k`, `alpha_mode`)
   - Edit [config/prompt.txt](config/prompt.txt) to customize LLM aesthetic evaluation criteria
   - Note: Script will prompt if experiment already exists

2. **Execute**:
   ```bash
   python run.py
   ```
   
   The script will:
   - Load configuration from `config/experiment_config.yaml`
   - Check if experiment already exists and prompt for overwrite
   - Display loaded configuration
   - Run the evolutionary process
   - Save configuration to the experiment directory

3. **Monitor output**:
   - Progress bars show design generation and evaluation
   - Ranked image grids saved to `Experiments/{name}/run{N}/Images/`
   - Parameter JSONs stored in `Experiments/{name}/run{N}/Params/`
   - Configuration backup saved to `Experiments/{name}/experiment_config.yaml`

### Generating a Benchmark Dataset

`generate.py` creates a population of random designs without running the evolutionary loop. This is useful for building a fixed benchmark set for evaluation.

1. **Configure generation settings** in [config/generation_config.yaml](config/generation_config.yaml):
   ```yaml
   param_spec_file: "config/param_spec.yaml"
   sketch_dir: "/path/to/Harmonograph"
   processing: parallel
   screen: False
   workers: 8
   population_size: 100
   ```

2. **Execute**:
   ```bash
   python generate.py
   ```

   Output is written to `Data/benchmark/Images/` and `Data/benchmark/Params/`.

### Running with the Web App

The repository includes an additive web app that executes the same pipeline in-process.

1. **Start the web server**:
   ```bash
   python run_web.py
   ```

2. **Open the UI**:
   - `http://127.0.0.1:8000`

3. **Submit a job in the form**:
   - Fill the same core fields as `run.py` (`experiment_name`, `runs`, `population_size`, `param_spec_file`, `sketch_dir`, prompt, ranking, mutation, selection settings).
   - Set `parents_compete` and `competing_parents_rate` if you want elite parents retained each generation.
   - Enable overwrite only when intentionally replacing an existing experiment.

4. **Track execution**:
   - Jobs page shows `queued` / `running` / `completed` / `failed` status.
   - Live logs are available per job.

5. **Browse outputs**:
   - Open experiment pages to view generation images, ranking plots, and parameter JSON files.
   - Artifacts are read from the same `Experiments/{name}/run{N}/` structure used by CLI runs.

For additional web UI details, see [webapp/README.md](webapp/README.md).

### Running the Standalone Pairwise Voting App

This app is separate from the experiment web UI and presents two images per round with three choices: first image, second image, or draw.

1. **Start the standalone server**:
   ```bash
   python run_pairwise_coverage.py --n 3 --data-dir Data/benchmark/Images
   ```

   Full CLI options:
   | Flag | Default | Description |
   |------|---------|-------------|
   | `--n` | *(required)* | Minimum times each image must be shown globally |
   | `--data-dir` | `Data/curated/Images` | Directory of images to compare |
   | `--state-dir` | `pairwise_coverage_app/state` | Persistent CSV and state directory |
   | `--host` | `0.0.0.0` | Host to bind |
   | `--port` | `5050` | Port to bind |
   | `--debug` | off | Enable Flask debug mode |

2. **Open the app**:
   - `http://127.0.0.1:5050/compare`

3. **Behavior guarantees**:
   - Unordered pairs are never repeated.
   - The app keeps scheduling comparisons until every image has been shown at least `n` times, or marks the target impossible.
   - Coverage and no-repeat constraints are global across restarts/sessions via persistent state files.

4. **API endpoints**:
   | Endpoint | Method | Description |
   |----------|--------|-------------|
   | `/compare` | GET | Main voting UI |
   | `/vote` | POST | Submit a vote (`image_a`, `image_b`, `outcome`) |
   | `/undo` | POST | Undo the most recent vote for the current session |
   | `/status` | GET | JSON status payload (counts, deficits, can_undo) |
   | `/progress` | GET | Glicko convergence metrics: `estimated_comparisons_left`, `average_deviation`, `max_deviation` |
   | `/images/<image_id>` | GET | Serve an image file by ID |

5. **Output files**:
   - CSV outcomes: `pairwise_coverage_app/state/labels.csv`
   - State snapshot: `pairwise_coverage_app/state/global_state.json`

Use `--state-dir` if you want separate datasets/runs to use independent tracking files.

### Benchmarking: Evaluating Model Rankings Against Human Labels

`evaluate_scores.py` provides tools to compare a VLM's aesthetic rankings against a human-labelled ground truth. It is also imported by `pairwise_coverage_app` to compute live Glicko convergence metrics on the `/progress` endpoint.

#### Key Functions

| Function | Description |
|----------|-------------|
| `calc_scores(scores_path, plot)` | Computes Glicko ratings from a CSV of pairwise outcomes (`image_a`, `image_b`, `outcome`). Returns players sorted by rating. |
| `calc_scores_sampled(scores_path, n, plot, seed)` | Same as `calc_scores` but on a random subset where each design appears in at most `n` comparisons. Pass `seed` for reproducibility. |
| `compare_rankings(ranking_a, ranking_b, k_values)` | Computes Spearman rho, Kendall tau, and top-k overlap/Jaccard between two ordered ID lists. |
| `model_predictions(model_name, prompt_filepath, benchmark_dir, ...)` | Runs the VLM over all pairs in `benchmark_dir` and writes predictions to a CSV. |
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

The script runs the full pipeline: generates model predictions, computes Glicko scores for both human and model rankings, then reports averaged Spearman rho, Kendall tau, and top-5/10/20 Jaccard over 1000 sampled subsets (each design appears in at most `n` comparisons per sample). If `traditional-results.csv` exists, it also benchmarks structural complexity metrics.

#### CSV Format

Both human labels (`--benchmark_scores_csv`) and model predictions (`--model_predictions_csv`) use the same format:

```
image_a,image_b,outcome
run_1.png,run_2.png,first    # first image wins
run_3.png,run_4.png,second   # second image wins
run_5.png,run_6.png,draw     # draw
```

The 5-column format (with `timestamp,session_id` prefix) produced by the pairwise coverage app is also accepted automatically.

### Output Structure

```
Experiments/
  {experiment_name}/
    experiment_config.yaml   # Backup of configuration used
    run0/                    # Initial generation
      Images/
        run0_0.png
        run0_1.png
        ...
      Params/
        run0_0.json         # Contains "parameters" dict + "filepath"
        run0_1.json
        ...
    run1/                    # Generation 1
      Images/
      Params/
    ...
```

## Parameter Specification

Parameters are defined in [config/param_spec.yaml](config/param_spec.yaml) with:

```yaml
parameter_name:
  type: "int" | "float" | "categorical"
  min: <value>          # For numeric types
  max: <value>          # For numeric types
  values: [...]         # For categorical type
  calc: true | false    # If true, computed from other params
```

**Calculated Parameters**:
- `dt`: Computed as `1.0 / (fb * max(frx, fry * sampleRate))`
- `spaceP`: Computed as `10^spaceD`

These are automatically derived and excluded from mutation/crossover.

## Dependencies

Key packages:
- `torch==2.7.0+cu128`: PyTorch with CUDA 12.8
- `transformers`: HuggingFace Transformers for Qwen3-VL (for Glicko/Simple ranking)
- `flash-attn==2.7.4`: Flash Attention 2 for memory-efficient inference (for Glicko/Simple ranking)
- `qwen-vl-utils`: Vision-language model utilities (for Glicko/Simple ranking)
- `open_clip_torch`: OpenCLIP for CLIP-IQA ranking method
- `numpy`, `tqdm`, `pyyaml`: Standard scientific Python stack

See [requirements.txt](requirements.txt) for complete list.

## Performance Notes

- **GPU Memory**: Default chunk_size=32 for batch processing; reduce if OOM occurs (applies to Glicko/Simple methods)
- **Generation Time**: Parallel mode significantly faster; adjust `workers` based on CPU cores
- **Timeout**: 20s limit per design render to prevent hangs; may need adjustment for complex sketches
- **Selection Pressure**: Tournament parameter `k` (as percentage) controls selection pressure; lower k = more elitism
- **Parent Competition**: Higher `competing_parents_rate` increases exploitation (stability of strong designs) but can reduce diversity
- **Mutation**: `mutation_rate` controls per-parameter mutation probability; `mutation_sigma` controls magnitude
- **Ranking Speed**: CLIP-IQA is ~100x faster than pairwise methods for large populations (O(N) vs O(N²) comparisons)

## Extending

- **New Parameter Types**: Add to `config/param_spec.yaml` and update `Params` class categorization
- **Custom Selection**: Modify `evolve_population()` sampling strategy
- **Multi-objective Optimization**: Extend `calc_ranks()` for multiple fitness criteria
- **Checkpointing**: Add serialization of `population_params` between runs (not currently implemented)

## Authors

- Stephen Krol (@sjkro1) - January 2026

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE.txt) (CC BY-NC-SA 4.0).

You are free to:
- **Share**: Copy and redistribute the material in any medium or format
- **Adapt**: Remix, transform, and build upon the material

Under the following terms:
- **Attribution**: You must give appropriate credit
- **NonCommercial**: You may not use the material for commercial purposes
- **ShareAlike**: If you remix, transform, or build upon the material, you must distribute your contributions under the same license

See [LICENSE.txt](LICENSE.txt) for the full license text.
