# Graph Recommender System (MovieLens)
\* UC IRVINE EECS 298 COURSE PROJECT

An implementation of a Graph Neural Network recommender system using **PyTorch Geometric (PyG)** and **GraphSAGE**. This project supports both **MovieLens-100k** and **MovieLens-1M** datasets.

---

##  1. Quick Start (Environment Setup)

This project uses `uv` for high-performance dependency management. Please follow these steps to set up your environment in minutes.

### Step 1: Install uv (Skip if already installed)

**Mac / Linux:**
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

```

**Windows (PowerShell):**

```powershell
powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"

```

*(Note: You may need to restart your terminal or run `source $HOME/.cargo/env` after installation.)*

### Step 2: Initialize Environment

Run the following command in the project root directory:

```bash
uv sync

```

*This will automatically create a virtual environment and install Python 3.14+, PyTorch, PyG, and all required dependencies.*

---

##  2. Data Preparation

Please download the datasets and place them **strictly following the directory structure below**.

### Download Links

* **MovieLens 100k**: [Download ml-100k.zip](https://www.google.com/search?q=https://files.grouplens.org/datasets/movielens/ml-100k.zip)
* **MovieLens 1M**: [Download ml-1m.zip](https://www.google.com/search?q=https://files.grouplens.org/datasets/movielens/ml-1m.zip)

### Directory Structure

After unzipping, ensure your `data/` folder looks exactly like this:

```text
graph-rec-movielens/
├── src/
├── ...
└── data/
    ├── ml-100k/       <-- Contains u.data, u.item, etc.
    │   ├── u.data
    │   ├── u.item
    │   └── ...
    └── ml-1m/         <-- Contains ratings.dat, movies.dat, etc.
        ├── movies.dat
        ├── ratings.dat
        └── ...

```

---

##  3. Verification

Once the setup is complete, run the env tester to verify the environment and data paths:

```bash
uv run env_tester.py

```

##  4. Dataset Experiments and Analysis

Run the data loader to verify the environment and data paths:

```bash
uv run src/data_loader.py

```

**Expected Output:**

1. Terminal displays the `Graph Summary` and `HeteroData` structure.
2. A `plots/` folder is automatically generated in the project root, containing data distribution visualizations.

---

##  5. Running the Pipeline & Experiments

### Step 1: Verify Data Processing (Optional)
You can verify the environment and data paths by running the data loader directly:
```bash
uv run src/data_loader.py

```

**Expected Output:** The terminal will display the bipartite `HeteroData` structure, and a `plots/` folder will be automatically generated containing data distribution visualizations.

### Step 2: Train the GraphSAGE Model

Execute the full training, validation, and testing pipeline:

```bash
uv run src/trainer.py

```

**Expected Output:**

1. **Hardware Acceleration:** The script automatically detects and utilizes the best available hardware (`mps` for Mac Apple Silicon, `cuda` for NVIDIA GPUs, or `cpu`).
2. **Graph Splitting:** It performs a strict `RandomLinkSplit` to ensure no data leakage between Train, Validation, and Test edges.
3. **Training Logs:** You will see epoch-by-epoch progress tracking **Train Loss (MSE)**, **Val RMSE**, **Recall@10**, and **NDCG@10**.
4. **Checkpointing:** The best model weights are automatically saved to a `checkpoints/` directory.
5. **Final Evaluation:** After training, the model evaluates the unseen Test Set. For the ML-100k dataset, you can expect a Test RMSE of ~1.01, Recall@10 of ~0.86, and NDCG@10 of ~0.84.



---

##  6. Troubleshooting

Below are the setup issues [@mythy-xie](https://github.com/mythy-xie) encountered during the encoding process. Recommend checking these preparations in advance.

**Q: `FileNotFoundError` when running the script?**

A: Please check your `data/` directory structure. Ensure you didn't create nested folders (e.g., `data/ml-100k/ml-100k/` is incorrect).

**Q: Stuck at `Building wheel for torch-scatter`?**

A: This indicates a compilation issue. Ensure you have a C++ compiler installed.

* **Mac Users**: Run `xcode-select --install` in your terminal.

