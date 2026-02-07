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

*This will automatically create a virtual environment and install Python 3.10+, PyTorch, PyG, and all required dependencies.*

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

##  4. Experiments (to be finished)

Run the data loader to verify the environment and data paths:

```bash
uv run src/data_loader.py

```

**Expected Output:**

1. Terminal displays the `Graph Summary` and `HeteroData` structure.
2. A `plots/` folder is automatically generated in the project root, containing data distribution visualizations.

---

##  5. Troubleshooting

Below are the setup issues @mythy-xie encountered during the encoding process. Recommend checking these preparations in advance.

**Q: `FileNotFoundError` when running the script?**

A: Please check your `data/` directory structure. Ensure you didn't create nested folders (e.g., `data/ml-100k/ml-100k/` is incorrect).

**Q: Stuck at `Building wheel for torch-scatter`?**

A: This indicates a compilation issue. Ensure you have a C++ compiler installed.

* **Mac Users**: Run `xcode-select --install` in your terminal.

