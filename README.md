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
5. Execute the inference script from `<project_path>` directory
    ```python
    # Example command for squad dataset with ner_rand_direct scheme:
    python main.py --dataset squad \
                   --data_split validation \
                   --data_size 1000 \
                   --eval_model Qwen2.5-14B-Instruct \
                   --eval_model_gpu 0 \
                   --comment ner_rand_direct \
                   --embedding_model all-mpnet-base-v2 \
                   --ner_model ner_pipe \
                   --ner_keys entity \
                   --entity_score_threshold 0.8 \
                   --entity_map_method ner_dict_select \
                   --rep_gen_method str_replace \
                   --recover_method str_replace
    ```
   For all available parameters, see `<project_path>/Utils/argparse_util.py`



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

## Citation
If you find this work useful, please cite our paper:
```bibtex
@misc{anonymous2024pseudonymization,
  title         = {A General Pseudonymization Framework for Cloud-Based LLMs: Replacing Privacy Information in Controlled Text Generation},
  author        = {Anonymous},
  year          = {2025},
  archivePrefix = {arXiv},
  eprint        = {2502.15233},
  primaryClass  = {cs.CL}
}
```

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For any inquiries, please:
- Open an issue on GitHub
- Contact the authors at: hou_work@yeah.net or longzi@sztu.edu.cn