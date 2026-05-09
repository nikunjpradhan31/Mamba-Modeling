# Mamba-Modeling

Mamba-Modeling is a machine learning research project exploring the application of State Space Models (specifically Mamba) combined with Graph Isomorphism Networks (GIN) for molecular toxicity prediction on the Tox21 dataset.

## Context & Architecture

This repository implements a **Hybrid GIN-Mamba Model** (`GINMambaHybrid`) to leverage both graph structural information and sequence modeling capabilities. The model runs Graph Isomorphism Network with Edges (GINE) and Mamba in parallel, combining their embeddings via specialized **Fusion Layers** (`fusion_layer.py`). Because Mamba models operate on 1D sequences, a key challenge in applying them to graph-structured molecular data is determining the optimal sequence traversal or node ordering.

The project features:
- **ZINC Pretraining**: Support for pretraining the model on the ZINC dataset (`pretrain_zinc.py`) prior to Tox21 finetuning.
- **Mamba2 Integration**: Support for the `mamba2_minimal` implementation for sequence processing.
- **Parallel Execution & Fusion**: The hybrid model parallelizes GINE and Mamba branches, merging them via flexible fusion strategies.

To investigate this, the project evaluates various **Node Ordering Strategies**:
- `random`: Random permutation of nodes
- `atomic_number`: Sorting nodes based on their atomic number
- `electronegativity`: Sorting nodes based on elemental electronegativity
- `degree`: Sorting nodes by their degree within the molecular graph
- `learned`: A parameterized, learnable ordering function

The codebase also supports a baseline standalone GIN ablation for comparative analysis.

## Key Technologies
- **PyTorch & PyTorch Geometric (PyG)**: For graph neural networks and overall deep learning framework.
- **Mamba-SSM**: For the State Space Model layers.
- **RDKit**: For cheminformatics and molecular feature extraction.
- **Scikit-learn / Numpy / Pandas**: For metrics, data manipulation, and training utility.

## Project Structure
- `src/models/`: Contains the `GINMambaHybrid` architecture and the various fusion layers.
- `src/mamba2_minimal/`: Contains a minimal implementation of Mamba2.
- `src/ordering/`: Implements the various node sequence ordering strategies.
- `src/data/`: Handles `Tox21Dataset` loading, feature processing, and scaffold splitting.
- `src/training/`: Training and evaluation loops with metrics (ROC-AUC, PRC-AUC, F1-Score).
- `configs/`: YAML configuration files for hyperparameter management (`default.yaml`, `pretrain.yaml`).
- `main.py` / `run_experiments.py`: Main entry points for training and experimentation.
- `pretrain_zinc.py`: Script for pretraining the model on the ZINC dataset.
- `generate_graphs.ipynb`: Notebook for visualizing training/validation curves and ROC-AUC / PR-AUC metrics.

## Usage

You can run the training pipeline via `main.py` using different orderings or model types:

```bash
# Pretrain on ZINC dataset
python pretrain_zinc.py --config configs/pretrain.yaml

# Run hybrid model with atomic number ordering
python main.py --model_type hybrid --ordering atomic_number

# Run standalone GIN baseline
python main.py --model_type gin

# Run with learnable node ordering
python main.py --model_type hybrid --ordering learned
```