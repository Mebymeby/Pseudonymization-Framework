# A General Pseudonymization Framework for Cloud-Based LLMs: Replacing Privacy Information in Controlled Text Generation

[![arXiv](https://img.shields.io/badge/arXiv-2502.15233-b31b1b.svg)](https://arxiv.org/abs/2502.15233)
[![Code](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/Mebymeby/Pseudonymization-Framework)
![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains the implementation code for the paper:  
**A General Pseudonymization Framework for Cloud-Based LLMs: Replacing Privacy Information in Controlled Text Generation**

Framework Diagram:  
![Method Overview](./Method%20Overview/img.png)

## Quick Start

1. Download required model files to `<project_path>/Model/`. Refer to `<project_path>/Config/model_config` for model specifications.
2. Download datasets to `<project_path>/Dataset/`. Required datasets are listed in `<project_path>/Config/dataset_config`.
3. Either:
   - Download source code for evaluation metrics to `<project_path>/Metrics/`, or
   - Directly import the `evaluate` library in your code
4. Install dependencies
5. Execute the inference script from `<project_path>` directory. For all available parameters, see `<project_path>/Utils/argparse_util.py` and `run_cmd.md`

## Key Features
- Pseudonymization framework for privacy preservation in remote LLM interactions
- Multiple entity detect,generate and replace strategies
- Customizable privacy detection thresholds
- Support for multiple methods combinations
- Comprehensive evaluation metrics integration

## Project Structure
```
.
├── Config/               # Configuration files
│   ├── model_config      # Model specifications
│   └── dataset_config    # Dataset requirements
├── Dataset/              # Dataset storage (create empty directory)
├── Model/                # Pretrained models
├── Output/               # Result, Logs and Intermediate result
├── Metrics/              # Evaluation metrics from evaluate lib
├── Utils/                # Utility functions
│   └── argparse_util.py  # Argument parser configuration
└── main.py               # Main execution script
```

## License
This project is licensed under the [MIT License](LICENSE).