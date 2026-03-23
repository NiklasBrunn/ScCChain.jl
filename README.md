# ScCChain

scCChain (**s**ingle-**c**ell **C**ommunication **Chain**s) is a Julia framework for
detecting and localizing cell–cell communication programs in spatial transcriptomics data.

It constructs a multi-layer cell graph from gene expression and spatial coordinates,
discovers interpretable communication programs via structured dimensionality reduction,
samples communication chains by weighted random walks, and trains a transformer to
prioritize programs and localize communication hotspots within tissue.

## Installation

Requires Julia ≥ 1.10.8

```julia
using Pkg
Pkg.add(url="https://github.com/NiklasBrunn/ScCChain.jl")
```

Or for development:

```bash
git clone https://github.com/NiklasBrunn/ScCChain.jl.git
cd ScCChain.jl
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

### Python Dependencies

scCChain uses PythonCall.jl to call scanpy/anndata for data preprocessing. A private Conda
environment is managed automatically via CondaPkg — Python dependencies (`scanpy`, `anndata`,
`numpy`, `pandas`, `scipy`, `h5py`, `pyarrow`, `openpyxl`) are installed on first `using ScCChain`. The
initial import may take a few minutes while the environment resolves. No manual Python, pip, or
conda setup is required.

## Quick Start

```julia
using ScCChain

# Load data and LR pair database
scdata = load_scdata("path/to/sample.h5ad")
db = load_lrpair_db(; species="human")

# Build multi-layer cell graph
pcs = pca(scdata; k=30)
graph = build_cell_graph(scdata, db; radius=300, alpha=0.00002, dim_red=pcs)

# Discover communication programs
programs = discover_programs_bae(graph; n_zdims=8, seed=42)

# Sample communication chains
chain_result = sample_chains(graph, programs; n_samples=5, n_steps=5, seed=42)

# Train transformer model
model = train_model(chain_result.chains, programs, scdata; n_epochs=1000, seed=42)
result = predict(model, chain_result.chains, scdata)
```

## Reproducing Paper Figures

The `analysis/` directory contains scripts that reproduce manuscript figures.
Paper results were produced with Julia 1.10.8.

### Setup

Follow the instructions in the preprocessing tutorial notebook to download the
example Visium and Xenium data.

Both datasets require the `Cell_Barcode_Type_Matrices.xlsx` file (per-spot/cell type
annotations) — download it from the dataset webpage and place a copy in each input
directory before running the notebook.

Run the preprocessing tutorial notebook to preprocess the raw Visium and Xenium
datasets and to save the results to `data/examples/`:

```bash
julia --project=. -e "using IJulia; notebook(dir=\"tutorials\")"
# Open and run: 01_preprocessing_example_data.ipynb
```

This produces:

- `data/examples/visium/visium_preprocessed.h5ad`
- `data/examples/xenium/xenium_preprocessed.h5ad`

### Running

```bash
# Figure 2 — Visium breast cancer communication program analysis
julia --project=. analysis/figure_2.jl

# Figures 3 & 4 — Xenium breast cancer CXCL12–CXCR4 analysis
julia --project=. analysis/figure_3_4.jl
```

Output panels are saved to `output/figure_2/panels/` and `output/figure_3_4/panels/`.

## Repository Structure

```
src/          Package source
  io/         Data loading & database parsing
  graph/      Cell graph construction
  programs/   Communication program discovery
  chains/     Communication chain generation
  model/      Transformer model architecture & training
  eval/       Evaluation metrics
  plotting/   Visualization
  utils/      Shared utilities
test/         Tests
tutorials/    Tutorial notebooks
analysis/     Paper figure reproduction scripts
data/         Bundled databases (CellChat, PPI)
```

## Citation

If you use scCChain in your work, please cite:

Brunn, N., Guitart, L. C., Farhadyar, K., Fullio, C. L., Kailer, J., Vogel, T., Hackenberg, M., & Binder, H. (2026). Mapping spatial cell-cell communication programs by tailoring chains of cells for transformer neural networks. *bioRxiv*, 2026.03.18.712664. https://doi.org/10.64898/2026.03.18.712664v1

```bibtex
@article{brunn2026mapping,
  title={Mapping spatial cell-cell communication programs by tailoring chains of cells for transformer neural networks},
  author={Brunn, Niklas and Guitart, Laia C and Farhadyar, Kiana and Fullio, Camila L and Kailer, Jakob and Vogel, Tanja and Hackenberg, Maren and Binder, Harald},
  journal={bioRxiv},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
