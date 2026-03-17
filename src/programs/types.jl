struct PairFeatureMatrix
    X::Matrix{Float32}
    sender_index::Vector{Int}
    receiver_index::Vector{Int}
    communication_names::Vector{String}
end

"""
    ProgramResult

Result container for communication program discovery via BAE.

# Fields
- `encoder_weights::Matrix{Float32}`: sparse encoder weight matrix `(n_features, n_programs)`
- `loadings_split_nonnegative::Matrix{Float32}`: non-negative split loadings `(n_features, 2*n_programs)`
- `latent::Matrix{Float32}`: latent representation `(n_pairs, n_programs)`
- `cluster_probs::Matrix{Float32}`: soft cluster assignment probabilities `(n_programs, n_pairs)`
- `cluster_labels::Vector{Int}`: hard cluster assignments per pair
- `latent_split_softmax::Matrix{Float32}`: softmax-transformed split latent
- `pair_metadata::NamedTuple`: `(sender_index, receiver_index)` cell ID vectors
- `basis_selection::NamedTuple`: basis selection results (enabled, keep, drop, stats, ...)
- `cp_mapping::Dict{Int,Int}`: mapping from dense program indices to original indices
- `metadata::Dict{String,Any}`: additional metadata (communication_names, sparsity, ...)
- `top_features::Dict{String,DataFrame}`: ranked features per program from `_top_features_from_loadings`
"""
Base.@kwdef struct ProgramResult
    encoder_weights::Matrix{Float32}
    loadings_split_nonnegative::Matrix{Float32}
    latent::Matrix{Float32}
    cluster_probs::Matrix{Float32}
    cluster_labels::Vector{Int}
    latent_split_softmax::Matrix{Float32}
    pair_metadata::NamedTuple
    basis_selection::NamedTuple
    cp_mapping::Dict{Int,Int}
    metadata::Dict{String,Any}
    top_features::Dict{String,DataFrame}
end

Base.@kwdef mutable struct BAEHyperparameters
    zdim::Int = 10
    n_runs::Int = 1
    max_iter::Int = 1000
    tol::Union{Nothing,Float32} = 1.0f-5
    batchsize::Int = 512
    η::Float32 = 1.0f-2
    λ::Float32 = 1.0f-1
    ϵ::Float32 = 1.0f-3
    M::Int = 1
end

mutable struct BAEModel
    encoder_weights::Matrix{Float32}
    decoder::Any
    hp::BAEHyperparameters
end
