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
- **Multiple Ranking Systems**: 
  - **Glicko-2**: Probabilistic ranking using Glicko-2 algorithm for robust pairwise comparison evaluation
  - **CLIP-IQA**: Vision-language model-based quality assessment without pairwise comparisons
  - **Simple**: Basic win/loss counting from pairwise comparisons
- **Vision-based Ranking**: Automated aesthetic evaluation using Qwen3-VL-8B-Instruct with pairwise comparisons (for Glicko/Simple) or CLIP for direct quality scoring (CLIP-IQA)
- **Flexible Parameters**: YAML-based parameter specification with int, float, and categorical types
- **Parallel Processing**: Multi-worker design generation and chunked GPU batch inference
- **Headless Rendering**: Support for server environments without displays

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
   - Sample parent pairs using Glicko rating-based probability (higher rating → higher selection chance)
   - Breed offspring via arithmetic mean crossover (α determined by `alpha_mode`)
   - Apply mutation to offspring parameters
   - Reset offspring Glicko ratings to initial values (1500 rating, 350 RD)
   - Increment generation counter
   - Generate new population via `generator.generate_population()`

5. **Final Evaluation**:
   - After last generation, evaluate final population to rank results

### Configuration

All experiment parameters are configured in [config/experiment_config.yaml](config/experiment_config.yaml), which is organized into two sections:

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
  k: 0.25                     # Percentage of population in tournament selection
  ranking_method: "glicko"    # Ranking method: "glicko", "CLIP-IQA", or "simple"
```

**Note**: When using `ranking_method: "CLIP-IQA"`, the system uses OpenCLIP RN50 for direct quality assessment instead of LLM-based pairwise comparisons. This is significantly faster (O(N) vs O(N²)) but evaluates images independently rather than through comparative ranking.

**Alpha Mode Options**:
- `"fixed"`: Uses the specified `alpha` value for all crossovers
- `"random"`: Randomly samples alpha for each crossover
- `"biased"`: Biases alpha toward higher-ranked parent

**Ranking Method Options**:
- `"glicko"` (recommended for robust ranking): Uses Glicko-2 rating system with incremental updates, accounts for rating uncertainty via pairwise comparisons
- `"CLIP-IQA"` (recommended for speed): Uses CLIP vision-language model to directly score image quality without pairwise comparisons; based on [CLIP-IQA paper](https://arxiv.org/abs/2207.12396)
- `"simple"`: Uses basic win/loss counting from pairwise comparisons

**LLM Prompt File**:

The prompt is now stored in a separate file (e.g., [config/prompt.txt](config/prompt.txt)):

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
