"""
Communication program discovery submodule.

Applies structured dimensionality reduction over the per-communication-layer scores
of the multi-layer cell graph to produce **communication programs** (CPs): sparse,
interpretable latent factors where each loading directly corresponds to the
contribution of one ligand–receptor pair.

CPs are derived from the cell graph and serve a dual role:
1. **Guide chain construction** — programs tell the chain sampler which communication
   layers to emphasize when sampling chains.
2. **Transformer input features** — program scores for each cell in a chain are used
   as input features to the transformer model.

Key invariants:
- Encoder weights remain sparse and signed.
- Non-negative split loadings are derived deterministically from encoder weights.
- Each loading corresponds to one ligand–receptor pair's contribution.
- Programs must remain complementary (different programs capture different LR subsets).
"""
module Programs

using LinearAlgebra
using Statistics
using StatsBase
using DataFrames

include("types.jl")
include("transforms.jl")
include("feature_matrix.jl")
include("basis_selection.jl")
include("communication_layers.jl")
include("bae_optimization.jl")
include("api.jl")

export ProgramResult,
    discover_programs,
    discover_programs_bae,
    programs_to_communication_layers,
    top_features_per_program

end # module Programs
