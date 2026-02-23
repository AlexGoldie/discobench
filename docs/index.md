# DiscoGen

<!-- [![Release](https://img.shields.io/github/v/release/AlexGoldie/discogen)](https://img.shields.io/github/v/release/AlexGoldie/discogen)
[![Build status](https://img.shields.io/github/actions/workflow/status/AlexGoldie/discogen/main.yml?branch=main)](https://github.com/AlexGoldie/discogen/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/AlexGoldie/discogen/branch/main/graph/badge.svg)](https://codecov.io/gh/AlexGoldie/discogen)
[![Commit activity](https://img.shields.io/github/commit-activity/m/AlexGoldie/discogen)](https://img.shields.io/github/commit-activity/m/AlexGoldie/discogen)
[![License](https://img.shields.io/github/license/AlexGoldie/discogen)](https://img.shields.io/github/license/AlexGoldie/discogen) -->

**DiscoGen** is a modular benchmark for automated algorithm discovery in machine learning.

## What is DiscoGen?

DiscoGen is a procedural generator of machine learning tasks for algorithm discovery agents and AI scientist systems. DiscoGen has a modular setup, an emphasis on discovering algorithms that transfer to meta-test datasets, and supports a huge diversity of >1 billion tasks! We hope DiscoGen helps drive the frontier of research in algorithm discovery by providing a large-scale, open-ended landscape for optimising, understanding and evaluating AI research agents!

### Key Features

- **Modular Architecture**: Break down ML algorithms into composable components!
- **Multiple Domains**: Support for reinforcement learning, language modelling, computer vision, bayesian optimization, and more!
- **Flexible Configuration**: Easy switching between baseline and experimental implementations!
- **LLM-Ready**: Designed for automated algorithm discovery using AI agents!
- **Extensible**: Very simple to introduce new domains, expand current ones, and add new datasets or evaluation regimes!

## Quick Start

### Installation

Install from source:

```bash
git clone git@github.com:AlexGoldie/discogen.git
cd discogen
make install
```
or install from pip:
```bash
pip install discogen
```
### Basic Usage

List available domains:
```bash
uv run discogen get-domains
```

Create a full task-domain codebase (with baseline implementations):
```bash
uv run discogen create-task --task-domain OnPolicyRL
```

Create an example task for algorithm discovery:
```bash
uv run discogen create-task --task-domain OnPolicyRL --example
```

See the full [Usage Guide](usage.md) for detailed instructions.

## Available Domains

DiscoGen currently supports the following task domains:

- **[OnPolicyRL](domains.md#onpolicyrl)**: On-policy reinforcement learning (PPO-style algorithms)
- **[OffPolicyRL](domains.md#offpolicyrl)**: Off-policy reinforcement learning (DQN-style algorithms)
- **[LanguageModelling](domains.md#languagemodelling)**: Pre-training language models
- **[ComputerVisionClassification](domains.md#computervisionclassification)**: Image classification tasks
- **[BayesianOptimisation](domains.md#bayesianoptimisation)**: Black-box optimization
- **[BrainSpeechDetection](domains.md#brainspeechdetection)**: Neural signal analysis
- **[ModelUnlearning](domains.md#modelunlearning)**: LLM unlearning tasks
- **[UnsupervisedEnvironmentDesign](domains.md#unsupervisedenvironmentdesign)**: Environment curriculum learning
- **[ContinualLearning](domains.md#continuallearning)**: Learning under non-stationarity
- **[GreenhouseGasPrediction](domains.md#greenhousegasprediction)**: Predicting atmospheric greenhouse gas concentrations

See the [Domains](domains.md) page for detailed information about each domain.

## How It Works

### 1. Modular Components

Each task domain is decomposed into modules. For example, OnPolicyRL includes:
- `loss.py`: Objective function (e.g., PPO loss)
- `networks.py`: Neural network architectures
- `optim.py`: Optimization algorithms
- `train.py`: Training loop logic

### 2. Base and Edit Implementations

Each module has two versions:
- **Base**: Fully implemented, tested baseline
- **Edit**: Template with function signatures for customization

### 3. Configuration-Driven

Control which modules use baseline vs. custom implementations via YAML config:

```yaml
change_optim: true   # Use custom optimizer
change_loss: false   # Use baseline loss
change_networks: false
change_train: false
```

### 4. Task Generation

DiscoGen assembles the configured modules into a complete, runnable task in `task_src/`:

```bash
discogen create-task --task-domain OnPolicyRL
cd task_src/OnPolicyRL
python run_main.py
```

## Documentation

### For Users
- **[Usage Guide](usage.md)**: CLI commands, Python API, and workflows
- **[Domains](domains.md)**: Available task domains and their modules

### For Contributors
- **[Contributing Overview](how_to/overview.md)**: How to add new tasks to DiscoGen
- **[Dataset Integration](how_to/dataset_integration.md)**: Adding new datasets to tasks

## Example Use Cases

### Algorithm Discovery with LLMs

Use DiscoGen to have AI agents discover new ML algorithms:
1. Configure which modules should be generated by the LLM.
2. LLM writes implementations for those modules.
3. Evaluate performance across multiple tasks.
4. Iterate and refine based on results.

### Optimising Algorithm Discovery Agents

Use DiscoGen to continuously sample new tasks, and optimise the agent to maximise performance over these:
1. Sample a new DiscoGen task.
2. Use an algorithm discovery agent to develop new algorithms.
3. Update the agent scaffold, prompt or weights in response to the score of its algorithm.
4. Sample a new DiscoGen task and repeat.

## Project Structure

```
discogen/
├── tasks/              # Task domain implementations
│   ├── OnPolicyRL/
│   ├── LanguageModelling/
│   └── ...
├── utils/              # Core utilities
├── create_task.py      # Task generation logic
├── create_config.py    # Configuration utilities
└── cli.py              # Command-line interface

task_src/               # Generated task files (after running create-task)
```

## Contributing

We welcome contributions! DiscoGen grows stronger with more tasks and domains.

- Found a bug? [Open an issue](https://github.com/AlexGoldie/discogen/issues)
- Want to add a task? See the [Contributing Guide](how_to/overview.md)
- Adding datasets? Check the [Dataset Integration Guide](how_to/dataset_integration.md)

## Citation

If you use DiscoGen in your research, please cite:

```bibtex
    @article{goldie2025discogen,
      title={DiscoGen: Procedural Generation of Algorithm Discovery Tasks in Machine Learning},
      author={Alexander D. Goldie and Zilin Wang and Adrian Hayler and Deepak Nathani and Edan Toledo and Ken Thampiratwong and Aleksandra Kalisz and Michael Beukman and Alistair Letcher and Shashank Reddy and Clarisse Wibault and Theo Wolf and Charles O'Neill and Jakob N. Foerster and Shimon Whiteson and Roberta Raileanu},
      year={2025}
    }
```

## Links

- **GitHub Repository**: [https://github.com/AlexGoldie/discogen](https://github.com/AlexGoldie/discogen)
- **Documentation**: [https://AlexGoldie.github.io/discogen](https://AlexGoldie.github.io/discogen)
- **Blog**: [https://alexgoldie.github.io/discogen-blog/](https://alexgoldie.github.io/discogen-blog/)
- **PyPI Package**: Coming soon

## License

This project is licensed under the terms specified in the [LICENSE](https://github.com/AlexGoldie/discogen/blob/main/LICENSE) file.
