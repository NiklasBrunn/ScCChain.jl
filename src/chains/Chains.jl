"""
Communication chain generation submodule.

Samples communication chains by weighted random walks on the multi-layer cell graph,
guided by communication programs. Each chain is a directed sequence of cells starting
from a receiver cell.

Key principle (guilty-by-association): communication-silent cells (cells that do not express any ligand) can still appear
in chains, contributing contextual signal from transcriptionally similar neighbors.
"""
module Chains

using DataFrames
using Graphs
using LinearAlgebra
using Random
using SimpleWeightedGraphs
using Statistics
using SparseArrays
using StatsBase: Weights, sample
using ..Graph: CellGraph
using ..IO: scData, cell_annotation, expression_matrix, spatial_coords
using ..Model: ModelResult
using ..Programs: programs_to_communication_layers

Base.@kwdef struct ChainResult
    chains::Vector{Vector{Int}}
    stacked_matrix::Matrix{Vector{Int}}
    communication_labels::Vector{String}
    metadata::Dict{String,Any}
end

function _maybe_get(obj, keys::Vector{Symbol})
    for key in keys
        if obj isa AbstractDict
            if haskey(obj, key)
                return obj[key]
            end
            str_key = String(key)
            if haskey(obj, str_key)
                return obj[str_key]
            end
        elseif hasproperty(obj, key)
            return getproperty(obj, key)
        end
    end
    return nothing
end

function _resolve_chain_layers(graph::CellGraph, programs)
    if programs === nothing
        return graph.communication_layers, graph.communication_names, :raw_graph
    end

    meta = _maybe_get(programs, [:metadata])
    isnothing(meta) &&
        throw(ArgumentError("programs must include metadata with sender/receiver indices"))

    sender = _maybe_get(meta, [:sender_index, :sender_cell_IDs])
    receiver = _maybe_get(meta, [:receiver_index, :receiver_cell_IDs])
    isnothing(sender) && throw(ArgumentError("programs.metadata missing sender_index"))
    isnothing(receiver) && throw(ArgumentError("programs.metadata missing receiver_index"))

    cluster_probs = _maybe_get(programs, [:cluster_probs, :latent_split_softmax])
    isnothing(cluster_probs) &&
        throw(ArgumentError("programs must provide cluster_probs or latent_split_softmax"))

    n_cells = size(graph.similarity_layer, 1)
    pair_meta = (sender_index = Int.(sender), receiver_index = Int.(receiver))
    cpmat = permutedims(Float32.(cluster_probs))
    layers = programs_to_communication_layers(
        cpmat,
        n_cells,
        pair_meta;
        CP_cutoff = false,
        self_comm = true,
        materialize_dense = false,
        cutoff = 1.0f-5,
    )

    labels = String["CP_$(i)" for i = 1:length(layers)]
    return layers, labels, :program_adjusted
end

mutable struct _CellMultiGraph
    layer1::SimpleWeightedDiGraph{Int,Float64}
    layer2::SimpleWeightedDiGraph{Int,Float64}
    q::Vector{Float64}
end

function _as_sparse_float64(mat)
    if issparse(mat)
        return sparse(Float64.(mat))
    end
    return sparse(Float64.(mat))
end

function _clip_negative_sparse(mat::SparseMatrixCSC{Float64,Int})
    clipped = copy(mat)
    clipped.nzval .= max.(clipped.nzval, 0.0)
    dropzeros!(clipped)
    return clipped
end

function _multi_weighted_randomwalk(
    cell_graph::_CellMultiGraph,
    start_cell::Integer,
    nsteps::Integer;
    ntop_nbrs::Union{Int,Nothing} = nothing,
    communication_threshold::Union{Float64,Nothing} = nothing,
    celltype_receiver_indices::Union{Vector{Int},Nothing} = nothing,
)
    visited_cells = Vector{Int}()
    visited_layers = Vector{Int}()
    sizehint!(visited_cells, nsteps)
    sizehint!(visited_layers, max(0, nsteps - 1))

    currs = Int(start_cell)
    i = 1
    q0 = copy(cell_graph.q)
    communication_layer_used = false

    while i <= nsteps + 1
        push!(visited_cells, currs)

        selected_layer_index = 2
        selected_layer = nothing
        nbrs = Int[]
        weights = Float64[]

        if communication_threshold === nothing
            if i > 1
                selected_layer_index = sample(Weights(cell_graph.q))
            end
            selected_layer =
                selected_layer_index == 1 ? cell_graph.layer1 : cell_graph.layer2

            if selected_layer_index == 1
                cell_graph.q = [0.0, 1.0]
            end

            nbrs = collect(outneighbors(selected_layer, currs))

            if isempty(nbrs) && (selected_layer_index == 1)
                selected_layer_index = 2
                selected_layer = cell_graph.layer2
                cell_graph.q = q0
                nbrs = collect(outneighbors(selected_layer, currs))
            end

            isempty(nbrs) && break
            weights = [get_weight(selected_layer, currs, nbr) for nbr in nbrs]
        else
            if !communication_layer_used
                comm_nbrs = collect(outneighbors(cell_graph.layer1, currs))
                comm_weights =
                    [get_weight(cell_graph.layer1, currs, nb) for nb in comm_nbrs]

                if celltype_receiver_indices !== nothing
                    filter_inds = findall(nb -> nb in celltype_receiver_indices, comm_nbrs)
                    comm_nbrs = comm_nbrs[filter_inds]
                    comm_weights = comm_weights[filter_inds]
                end

                valid_inds = findall(w -> w > communication_threshold, comm_weights)
                valid_comm_nbrs = comm_nbrs[valid_inds]
                valid_comm_wts = comm_weights[valid_inds]

                if !isempty(valid_comm_nbrs)
                    selected_layer_index = 1
                    selected_layer = cell_graph.layer1
                    nbrs = valid_comm_nbrs
                    weights = valid_comm_wts
                    communication_layer_used = true
                end
            end

            if selected_layer_index == 2
                selected_layer = cell_graph.layer2
                nbrs = collect(outneighbors(selected_layer, currs))
                isempty(nbrs) && break
                weights = [get_weight(selected_layer, currs, nb) for nb in nbrs]
            end
        end

        push!(visited_layers, selected_layer_index)

        if !isempty(nbrs)
            mask = nbrs .!= currs
            nbrs = nbrs[mask]
            weights = weights[mask]
        end

        isempty(nbrs) && break

        if ntop_nbrs !== nothing
            top_n = min(ntop_nbrs, length(nbrs))
            top_inds = partialsortperm(weights, 1:top_n; rev = true)
            nbrs = nbrs[top_inds]
            weights = weights[top_inds]
        end

        total_weight = sum(weights)
        total_weight <= 0 && break
        probabilities = weights ./ total_weight
        currs = nbrs[sample(Weights(probabilities))]
        i += 1
    end

    return visited_cells[1:(i-1)], visited_layers[1:(i-2)]
end

function _cut_chain_indices(stacked_matrix::Matrix{Vector{Int}})
    chains = Vector{Vector{Int}}()
    for i = 1:size(stacked_matrix, 1)
        comm_idx = findfirst(==(1), stacked_matrix[i, 2])
        isnothing(comm_idx) && continue
        cutoff = min(length(stacked_matrix[i, 1]), comm_idx + 1)
        push!(chains, Int.(stacked_matrix[i, 1][1:cutoff]))
    end
    return chains
end

function _generate_chains_for_start_cells(
    multi_cell_graph::_CellMultiGraph,
    start_cells::Vector{Int};
    communication_threshold::Union{Float64,Nothing} = nothing,
    nsteps::Int = 8,
    ntop_nbrs::Union{Int,Nothing} = nothing,
    nsamples::Int = 1,
    celltype_receiver_indices::Union{Vector{Int},Nothing} = nothing,
    remove_noninteracting_paths::Bool = true,
)
    all_visited_cells = Dict{Int,Vector{Vector{Int}}}()
    all_visited_layers = Dict{Int,Vector{Vector{Int}}}()

    distribution_visited_cells = Array{Vector{Int}}(undef, nsamples)
    distribution_visited_layers = Array{Vector{Int}}(undef, nsamples)
    q0 = copy(multi_cell_graph.q)

    for cell in start_cells
        for i = 1:nsamples
            multi_cell_graph.q = q0
            distribution_visited_cells[i], distribution_visited_layers[i] =
                _multi_weighted_randomwalk(
                    multi_cell_graph,
                    cell,
                    nsteps;
                    ntop_nbrs = ntop_nbrs,
                    communication_threshold = communication_threshold,
                    celltype_receiver_indices = celltype_receiver_indices,
                )
        end

        accepted_paths = if remove_noninteracting_paths
            [i for (i, layers) in enumerate(distribution_visited_layers) if any(==(1), layers)]
        else
            collect(1:nsamples)
        end

        all_visited_cells[cell] = distribution_visited_cells[accepted_paths]
        all_visited_layers[cell] = distribution_visited_layers[accepted_paths]
    end

    keys_sorted = sort(collect(keys(all_visited_cells)))
    if isempty(keys_sorted)
        return Vector{Vector{Int}}(), Matrix{Vector{Int}}(undef, 0, 2)
    end

    cells_col = vcat([all_visited_cells[key] for key in keys_sorted]...)
    layers_col = vcat([all_visited_layers[key] for key in keys_sorted]...)
    stacked_matrix = hcat(cells_col, layers_col)
    chains = [Int.(stacked_matrix[i, 1]) for i = 1:size(stacked_matrix, 1)]
    if remove_noninteracting_paths
        chains = _cut_chain_indices(stacked_matrix)
    end
    return chains, stacked_matrix
end

function _sample_across_layers(
    communication_layers::Vector,
    molecular_layer,
    communication_labels::Vector{String};
    start_from::String = "sender",
    q0::Vector{Float64} = [0.2, 0.8],
    n_samples::Int = 1,
    n_steps::Int = 8,
    q::Float64 = 0.8,
    cells_interest::Union{Nothing,Vector{Int},String} = nothing,
    ntop_nbrs::Union{Int,Nothing} = nothing,
    communication_threshold::Union{Float64,Nothing} = nothing,
    celltype_receiver_indices::Union{Vector{Int},Nothing} = nothing,
    remove_noninteracting_paths::Bool = true,
    filter_negative_values::Bool = true,
)
    start_from in ("sender", "receiver") ||
        throw(ArgumentError("start_from must be \"sender\" or \"receiver\""))
    length(q0) == 2 ||
        throw(ArgumentError("q0 must contain exactly two layer probabilities"))
    all(>=(0), q0) || throw(ArgumentError("q0 weights must be non-negative"))
    sum(q0) > 0 || throw(ArgumentError("q0 weights must sum to a positive value"))
    0.0 <= q <= 1.0 || throw(ArgumentError("q must be in [0, 1]"))

    selected_cells_interest = Int[]
    quantile_mode = false
    n_cells = size(molecular_layer, 1)
    if cells_interest isa String
        cells_interest == "quantile" ||
            throw(ArgumentError("If cells_interest is a String, it must be \"quantile\""))
        quantile_mode = true
    elseif isnothing(cells_interest)
        quantile_mode = true
    elseif cells_interest isa Vector{Int}
        isempty(cells_interest) &&
            throw(ArgumentError("cells_interest cannot be empty when passed explicitly"))
        all(1 .<= cells_interest .<= n_cells) ||
            throw(ArgumentError("cells_interest contains out-of-bounds cell index"))
        selected_cells_interest = copy(cells_interest)
    else
        throw(
            ArgumentError(
                "cells_interest must be nothing, \"quantile\", or Vector{Int}; got $(typeof(cells_interest))",
            ),
        )
    end

    mol_mat = _as_sparse_float64(molecular_layer)
    if filter_negative_values
        mol_mat = _clip_negative_sparse(mol_mat)
    end
    mol_graph = SimpleWeightedDiGraph(mol_mat)

    chains = Vector{Vector{Int}}()
    stacked_matrix = Matrix{Vector{Int}}(undef, 0, 2)
    sampled_communication_labels = String[]
    per_layer_counts = Dict{String,Int}()
    skipped_layers = Dict{String,String}()

    for inter in eachindex(communication_layers)
        layer_name = communication_labels[inter]
        comm_mat = _as_sparse_float64(communication_layers[inter])
        if start_from == "receiver"
            comm_mat = sparse(transpose(comm_mat))
        end
        comm_graph = SimpleWeightedDiGraph(comm_mat)
        multi_cell_graph = _CellMultiGraph(comm_graph, mol_graph, copy(q0))

        selected = if quantile_mode
            rowsums = vec(sum(comm_mat, dims = 2))
            feasible_cells = findall(>(0), rowsums)
            if isempty(feasible_cells)
                skipped_layers[layer_name] = "no_feasible_start_cells"
                Int[]
            else
                quant = quantile(view(rowsums, feasible_cells), q)
                out = findall(>(quant), rowsums)
                if isempty(out)
                    skipped_layers[layer_name] = "empty_after_quantile_threshold"
                end
                Int.(out)
            end
        else
            selected_cells_interest
        end

        if isempty(selected)
            per_layer_counts[layer_name] = 0
            continue
        end

        layer_chains, layer_stacked = _generate_chains_for_start_cells(
            multi_cell_graph,
            selected;
            communication_threshold = communication_threshold,
            nsteps = n_steps,
            ntop_nbrs = ntop_nbrs,
            nsamples = n_samples,
            celltype_receiver_indices = celltype_receiver_indices,
            remove_noninteracting_paths = remove_noninteracting_paths,
        )

        append!(chains, layer_chains)
        if size(layer_stacked, 1) > 0
            stacked_matrix = vcat(stacked_matrix, layer_stacked)
        end
        append!(sampled_communication_labels, fill(layer_name, length(layer_chains)))
        per_layer_counts[layer_name] = length(layer_chains)
    end

    return chains,
    stacked_matrix,
    sampled_communication_labels,
    per_layer_counts,
    skipped_layers
end

"""
    sample_chains(graph, programs; kwargs...)

Compatibility overload that forwards to keyword-based `programs` dispatch.

# Arguments
- `graph`: multi-layer cell graph (from `build_cell_graph`)
- `programs`: communication programs or `nothing`

# Returns
- [`ChainResult`](@ref)
"""
function sample_chains(graph::CellGraph, programs; kwargs...)
    return sample_chains(graph; programs = programs, kwargs...)
end

"""
    sample_chains(graph; programs=nothing, seed=42, kwargs...)

Sample communication chains using either the raw cell graph (`programs=nothing`) or
program-adjusted communication layers (`programs` provided).

# Arguments
- `graph::CellGraph`: multi-layer cell graph
- `programs`: optional communication programs
- `seed::Int`: random seed for reproducibility

# Returns
- [`ChainResult`](@ref)
"""
function sample_chains(graph::CellGraph; programs = nothing, seed::Int = 42, kwargs...)
    layers, layer_labels, mode = _resolve_chain_layers(graph, programs)
    Random.seed!(seed)

    q0 = Float64.(get(kwargs, :q0, [0.2, 0.8]))
    n_samples = Int(get(kwargs, :n_samples, 1))
    n_steps = Int(get(kwargs, :n_steps, 8))
    q = Float64(get(kwargs, :q, 0.8))
    cells_interest = get(kwargs, :cells_interest, nothing)
    ntop_nbrs = get(kwargs, :ntop_nbrs, nothing)
    communication_threshold = get(kwargs, :communication_threshold, nothing)
    celltype_receiver_indices = get(kwargs, :celltype_receiver_indices, nothing)
    remove_noninteracting_paths = Bool(get(kwargs, :remove_noninteracting_paths, true))
    start_from = String(get(kwargs, :start_from, "sender"))
    filter_negative_values = Bool(get(kwargs, :filter_negative_values, true))

    chains, stacked_matrix, communication_labels, per_layer_counts, skipped_layers =
        _sample_across_layers(
            layers,
            graph.similarity_layer,
            layer_labels;
            start_from = start_from,
            q0 = q0,
            n_samples = n_samples,
            n_steps = n_steps,
            q = q,
            cells_interest = cells_interest,
            ntop_nbrs = ntop_nbrs,
            communication_threshold = communication_threshold,
            celltype_receiver_indices = celltype_receiver_indices,
            remove_noninteracting_paths = remove_noninteracting_paths,
            filter_negative_values = filter_negative_values,
        )

    metadata = Dict{String,Any}(
        "source_mode" => mode,
        "seed" => seed,
        "params" => Dict(
            "q0" => q0,
            "n_samples" => n_samples,
            "n_steps" => n_steps,
            "q" => q,
            "cells_interest" => cells_interest,
            "ntop_nbrs" => ntop_nbrs,
            "communication_threshold" => communication_threshold,
            "remove_noninteracting_paths" => remove_noninteracting_paths,
            "start_from" => start_from,
        ),
        "per_layer_counts" => per_layer_counts,
        "skipped_layers" => skipped_layers,
    )

    length(chains) == size(stacked_matrix, 1) ||
        throw(ArgumentError("Output mismatch: chains and stacked_matrix row counts differ"))
    length(chains) == length(communication_labels) || throw(
        ArgumentError("Output mismatch: chains and communication_labels lengths differ"),
    )

    return ChainResult(
        chains = chains,
        stacked_matrix = stacked_matrix,
        communication_labels = communication_labels,
        metadata = metadata,
    )
end

function _validate_chain_metadata_inputs(
    chains::Vector{<:Vector{<:Integer}},
    communication_labels::Vector{String},
    pathway_labels,
    cell_annotation,
)
    length(chains) == length(communication_labels) ||
        throw(ArgumentError("chains and communication_labels must have matching lengths"))

    for (i, chain) in enumerate(chains)
        isempty(chain) && throw(ArgumentError("chains[$i] must not be empty"))
        all(>(0), chain) ||
            throw(ArgumentError("chains[$i] must contain positive cell IDs"))
    end

    if pathway_labels !== nothing
        length(pathway_labels) == length(chains) ||
            throw(ArgumentError("pathway_labels must match the number of chains"))
    end

    if cell_annotation !== nothing && !isempty(chains)
        max_cell_id = maximum(maximum(chain) for chain in chains)
        length(cell_annotation) >= max_cell_id ||
            throw(ArgumentError("cell_annotation must cover all referenced chain cell IDs"))
    end
end

"""
    construct_chain_metadata(chains, communication_labels; pathway_labels=nothing, cell_annotation=nothing)

Construct chain-level metadata with sender/receiver columns for downstream analysis.

# Arguments
- `chains::Vector{<:Vector{<:Integer}}`: sampled chains, where receiver is `chain[end]`
- `communication_labels::Vector{String}`: communication label per chain
- `pathway_labels`: optional pathway label per chain
- `cell_annotation`: optional cell-type vector indexed by cell ID

# Returns
- `DataFrame`: chain metadata table with ID/length columns and optional annotations

# Methods
- `construct_chain_metadata(chains, communication_labels; ...)` — explicit vectors
- `construct_chain_metadata(chain_result; scdata, ...)` — extracts from `ChainResult` and `scData`
"""
function construct_chain_metadata(
    chains::Vector{<:Vector{<:Integer}},
    communication_labels::Vector{String};
    pathway_labels = nothing,
    cell_annotation = nothing,
)
    _validate_chain_metadata_inputs(
        chains,
        communication_labels,
        pathway_labels,
        cell_annotation,
    )

    receiver_cell_ids = [chain[end] for chain in chains]
    first_sender_cell_ids = [chain[1] for chain in chains]
    penultimate_sender_cell_ids =
        Union{Int,Missing}[length(chain) > 1 ? chain[end-1] : missing for chain in chains]

    metadata = DataFrame(
        communication_labels = communication_labels,
        receiver_cell_ids = receiver_cell_ids,
        first_sender_cell_ids = first_sender_cell_ids,
        penultimate_sender_cell_ids = penultimate_sender_cell_ids,
        chain_lengths = length.(chains),
    )

    if pathway_labels !== nothing
        metadata[!, :pathway_labels] = pathway_labels
    end

    if cell_annotation !== nothing
        metadata[!, :receiver_cell_types] = cell_annotation[receiver_cell_ids]
        metadata[!, :first_sender_cell_types] = cell_annotation[first_sender_cell_ids]

        penultimate_sender_cell_types = Union{String,Missing}[
            ismissing(cell_id) ? missing : cell_annotation[cell_id] for
            cell_id in penultimate_sender_cell_ids
        ]
        metadata[!, :penultimate_sender_cell_types] = penultimate_sender_cell_types
    end

    return metadata
end

function construct_chain_metadata(
    chain_result::ChainResult;
    scdata::scData,
    pathway_labels = nothing,
    annotation_col = :auto,
)
    return construct_chain_metadata(
        chain_result.chains,
        chain_result.communication_labels;
        pathway_labels = pathway_labels,
        cell_annotation = cell_annotation(scdata; column = annotation_col),
    )
end

function _validate_chain_metadata_alignment(
    metadata::DataFrame,
    chains::Vector{<:Vector{<:Integer}},
)
    nrow(metadata) == length(chains) ||
        throw(ArgumentError("metadata rows must match the number of chains"))
    return nothing
end

function _validate_matrix_with_chain_ids(
    chains::Vector{<:Vector{<:Integer}},
    matrix::AbstractMatrix,
    matrix_name::String,
)
    all(isfinite, matrix) || throw(ArgumentError("$matrix_name contains non-finite values"))
    n_cells = size(matrix, 1)

    for (i, chain) in enumerate(chains)
        all(1 .<= chain .<= n_cells) ||
            throw(ArgumentError("chain $i has out-of-bounds cell IDs for $matrix_name"))
    end

    return nothing
end

function _get_max_attention_sender_positions(
    attention_per_head::AbstractArray{T,3},
    chains::Vector{<:Vector{<:Integer}},
) where {T<:Real}
    size(attention_per_head, 3) == length(chains) || throw(
        ArgumentError("attention_per_head third dimension must match number of chains"),
    )
    all(isfinite, attention_per_head) ||
        throw(ArgumentError("attention_per_head contains non-finite values"))

    n_sender_positions = size(attention_per_head, 2)
    sender_positions = Vector{Union{Int,Missing}}(undef, length(chains))
    sender_scores = Vector{Union{Float64,Missing}}(undef, length(chains))

    for i in eachindex(chains)
        n_valid_senders = length(chains[i]) - 1
        if n_valid_senders <= 0
            sender_positions[i] = missing
            sender_scores[i] = missing
            continue
        end

        n_valid_senders <= n_sender_positions || throw(
            ArgumentError(
                "chain $i expects $n_valid_senders sender positions, but attention has $n_sender_positions",
            ),
        )

        attention_slice = @view attention_per_head[:, 1:n_valid_senders, i]
        mean_attention = vec(mean(attention_slice; dims = 1))
        max_position = argmax(mean_attention)

        sender_positions[i] = max_position
        sender_scores[i] = mean_attention[max_position]
    end

    return sender_positions, sender_scores
end

function _chain_targets_from_expr(
    chains::Vector{<:Vector{<:Integer}},
    expr::AbstractMatrix{T},
) where {T<:Real}
    _validate_matrix_with_chain_ids(chains, expr, "expr")
    n_genes = size(expr, 2)
    y_true = Matrix{Float64}(undef, n_genes, length(chains))

    for (i, chain) in enumerate(chains)
        receiver_id = chain[end]
        y_true[:, i] = Float64.(view(expr, receiver_id, :))
    end

    return y_true
end

function _baseline_adjusted_mse_dims2(
    y_true::AbstractMatrix{T},
    y_pred::AbstractMatrix{S},
) where {T<:Real,S<:Real}
    size(y_true) == size(y_pred) ||
        throw(ArgumentError("y_true and y_pred must have matching dimensions"))
    all(isfinite, y_true) || throw(ArgumentError("y_true contains non-finite values"))
    all(isfinite, y_pred) || throw(ArgumentError("y_pred contains non-finite values"))

    n_chains = size(y_true, 2)
    baseline_center = vec(mean(y_true; dims = 2))
    mse = Vector{Float64}(undef, n_chains)
    adjusted_mse = Vector{Float64}(undef, n_chains)

    for i = 1:n_chains
        true_i = @view y_true[:, i]
        pred_i = @view y_pred[:, i]
        mse_i = mean((pred_i .- true_i) .^ 2)
        baseline_i = mean((baseline_center .- true_i) .^ 2)
        mse[i] = mse_i
        adjusted_mse[i] = mse_i / (baseline_i + eps(eltype(baseline_i)))
    end

    return mse, adjusted_mse
end

function _top_k_mask(values::AbstractVector{<:Real}, pct::Integer)
    0 <= pct <= 100 || throw(ArgumentError("error percentage must be in [0, 100]"))
    n_values = length(values)
    k = ceil(Int, pct / 100 * n_values)
    if k == 0
        return falses(n_values)
    end

    top_k_inds = partialsortperm(values, 1:k)
    mask = falses(n_values)
    @inbounds mask[top_k_inds] .= true
    return mask
end

"""
    add_max_attention_to_chain_metadata!(metadata, attention_per_head, chains; cell_annotation=nothing)

Add sender cell and score derived from per-head attention to chain metadata.

# Arguments
- `metadata::DataFrame`: chain metadata table
- `attention_per_head::AbstractArray{<:Real,3}`: tensor `(n_heads, n_sender_positions, n_chains)`
- `chains::Vector{<:Vector{<:Integer}}`: sampled chains
- `cell_annotation`: optional cell-type vector indexed by cell ID

# Returns
- `DataFrame`: input metadata with appended max-attention sender columns

# Methods
- `add_max_attention_to_chain_metadata!(metadata, attention_per_head, chains; ...)` — explicit arrays
- `add_max_attention_to_chain_metadata!(metadata, model_result, chain_result; scdata, ...)` — extracts from result types
"""
function add_max_attention_to_chain_metadata!(
    metadata::DataFrame,
    attention_per_head::AbstractArray{<:Real,3},
    chains::Vector{<:Vector{<:Integer}};
    cell_annotation = nothing,
)
    _validate_chain_metadata_alignment(metadata, chains)
    max_positions, max_scores =
        _get_max_attention_sender_positions(attention_per_head, chains)

    max_sender_cell_ids = Union{Int,Missing}[
        ismissing(pos) ? missing : chains[i][pos] for (i, pos) in enumerate(max_positions)
    ]

    metadata[!, :max_attention_sender_positions] = max_positions
    metadata[!, :max_attention_sender_cell_ids] = max_sender_cell_ids
    metadata[!, :max_attention_scores] = max_scores

    if cell_annotation !== nothing
        nonmissing_ids = collect(skipmissing(max_sender_cell_ids))
        if !isempty(nonmissing_ids) && length(cell_annotation) < maximum(nonmissing_ids)
            throw(ArgumentError("cell_annotation must cover max_attention_sender_cell_ids"))
        end
        metadata[!, :max_attention_sender_cell_types] = Union{String,Missing}[
            ismissing(cell_id) ? missing : cell_annotation[cell_id] for
            cell_id in max_sender_cell_ids
        ]
    end

    return metadata
end

function add_max_attention_to_chain_metadata!(
    metadata::DataFrame,
    model_result::ModelResult,
    chain_result::ChainResult;
    scdata::scData,
    annotation_col = :auto,
)
    return add_max_attention_to_chain_metadata!(
        metadata,
        model_result.attention_per_head,
        chain_result.chains;
        cell_annotation = cell_annotation(scdata; column = annotation_col),
    )
end

"""
    add_chain_model_errors_to_metadata!(metadata, chains, expr, predictions; mode=:mse, error_pcts=collect(10:10:100))

Add receiver-targeted chain-wise model errors to metadata using compatibility-equivalent formulas.

# Arguments
- `metadata::DataFrame`: chain metadata table
- `chains::Vector{<:Vector{<:Integer}}`: sampled chains
- `expr::AbstractMatrix{<:Real}`: cell-by-gene expression matrix
- `predictions::AbstractMatrix{<:Real}`: gene-by-chain predicted receiver expression
- `mode`: `:mse` or `:adj_mse`
- `error_pcts`: optional error percentile columns to append

# Returns
- `DataFrame`: input metadata with `:chain_wise_errors` and optional top-k columns

# Methods
- `add_chain_model_errors_to_metadata!(metadata, chains, expr, predictions; ...)` — explicit arrays
- `add_chain_model_errors_to_metadata!(metadata, chain_result, model_result, scdata; ...)` — extracts from result types
"""
function add_chain_model_errors_to_metadata!(
    metadata::DataFrame,
    chains::Vector{<:Vector{<:Integer}},
    expr::AbstractMatrix{<:Real},
    predictions::AbstractMatrix{<:Real};
    mode::Union{Symbol,String} = :mse,
    error_pcts::Union{Nothing,AbstractVector{<:Integer}} = collect(10:10:100),
)
    _validate_chain_metadata_alignment(metadata, chains)
    y_true = _chain_targets_from_expr(chains, expr)
    y_pred = Float64.(predictions)

    size(y_pred) == size(y_true) || throw(
        ArgumentError(
            "predictions must have size (n_genes, n_chains) matching chain receiver targets",
        ),
    )
    all(isfinite, y_pred) || throw(ArgumentError("predictions contains non-finite values"))

    mode_symbol = mode isa String ? Symbol(mode) : mode
    mse, adjusted_mse = _baseline_adjusted_mse_dims2(y_true, y_pred)

    if mode_symbol == :mse
        metadata[!, :chain_wise_errors] = mse
    elseif mode_symbol == :adj_mse
        metadata[!, :chain_wise_errors] = adjusted_mse
    else
        throw(ArgumentError("mode must be :mse or :adj_mse"))
    end

    if error_pcts !== nothing
        for error_pct in error_pcts
            metadata[!, Symbol(string(error_pct))] =
                _top_k_mask(metadata[!, :chain_wise_errors], error_pct)
        end
    end

    return metadata
end

function add_chain_model_errors_to_metadata!(
    metadata::DataFrame,
    chain_result::ChainResult,
    model_result::ModelResult,
    scdata::scData;
    mode::Union{Symbol,String} = :mse,
    error_pcts::Union{Nothing,AbstractVector{<:Integer}} = collect(10:10:100),
)
    return add_chain_model_errors_to_metadata!(
        metadata,
        chain_result.chains,
        Float32.(expression_matrix(scdata)),
        model_result.predictions;
        mode = mode,
        error_pcts = error_pcts,
    )
end

function _require_metadata_columns(metadata::DataFrame, columns::Vector{Symbol})
    metadata_cols = propertynames(metadata)
    missing_cols = Symbol[col for col in columns if !(col in metadata_cols)]
    isempty(missing_cols) ||
        throw(ArgumentError("metadata is missing required columns: $(missing_cols)"))
    return nothing
end

function _validate_ids_for_lookup(values, n_cells::Int, column_name::String)
    for value in values
        if ismissing(value)
            continue
        end
        1 <= value <= n_cells ||
            throw(ArgumentError("$column_name contains out-of-bounds cell ID: $value"))
    end
    return nothing
end

function _cosine_similarity(u::AbstractVector{<:Real}, v::AbstractVector{<:Real})
    denom = norm(u) * norm(v)
    if denom == 0.0
        return 0.0
    end
    return dot(u, v) / denom
end

"""
    add_distances_to_chain_metadata!(metadata, coords)

Add sender-to-receiver Euclidean distance columns to chain metadata.

# Arguments
- `metadata::DataFrame`: chain metadata table with sender/receiver ID columns
- `coords::AbstractMatrix{<:Real}`: cell-by-dimension coordinate matrix

# Returns
- `DataFrame`: input metadata with distance columns appended

# Methods
- `add_distances_to_chain_metadata!(metadata, coords)` — explicit coordinate matrix
- `add_distances_to_chain_metadata!(metadata, scdata)` — extracts coordinates from `scData`
"""
function add_distances_to_chain_metadata!(
    metadata::DataFrame,
    coords::AbstractMatrix{<:Real},
)
    all(isfinite, coords) || throw(ArgumentError("coords contains non-finite values"))
    _require_metadata_columns(
        metadata,
        [:first_sender_cell_ids, :receiver_cell_ids, :penultimate_sender_cell_ids],
    )

    n_cells = size(coords, 1)
    _validate_ids_for_lookup(
        metadata[!, :first_sender_cell_ids],
        n_cells,
        "first_sender_cell_ids",
    )
    _validate_ids_for_lookup(metadata[!, :receiver_cell_ids], n_cells, "receiver_cell_ids")
    _validate_ids_for_lookup(
        metadata[!, :penultimate_sender_cell_ids],
        n_cells,
        "penultimate_sender_cell_ids",
    )
    if :max_attention_sender_cell_ids in propertynames(metadata)
        _validate_ids_for_lookup(
            metadata[!, :max_attention_sender_cell_ids],
            n_cells,
            "max_attention_sender_cell_ids",
        )
    end

    n_chains = nrow(metadata)
    first_sender_distances = Vector{Float64}(undef, n_chains)
    penultimate_sender_distances = Vector{Union{Float64,Missing}}(undef, n_chains)

    for i = 1:n_chains
        first_sender = metadata[i, :first_sender_cell_ids]
        receiver = metadata[i, :receiver_cell_ids]
        first_sender_distances[i] =
            norm(view(coords, first_sender, :) .- view(coords, receiver, :))

        penultimate = metadata[i, :penultimate_sender_cell_ids]
        if ismissing(penultimate)
            penultimate_sender_distances[i] = missing
        else
            penultimate_sender_distances[i] =
                norm(view(coords, penultimate, :) .- view(coords, receiver, :))
        end
    end

    metadata[!, :first_sender_receiver_distance] = first_sender_distances
    metadata[!, :penultimate_sender_receiver_distance] = penultimate_sender_distances

    if :max_attention_sender_cell_ids in propertynames(metadata)
        max_attention_distances = Vector{Union{Float64,Missing}}(undef, n_chains)
        for i = 1:n_chains
            sender = metadata[i, :max_attention_sender_cell_ids]
            receiver = metadata[i, :receiver_cell_ids]
            if ismissing(sender)
                max_attention_distances[i] = missing
            else
                max_attention_distances[i] =
                    norm(view(coords, sender, :) .- view(coords, receiver, :))
            end
        end
        metadata[!, :max_attention_sender_receiver_distance] = max_attention_distances
    end

    return metadata
end

add_distances_to_chain_metadata!(metadata::DataFrame, scdata::scData) =
    add_distances_to_chain_metadata!(metadata, Float32.(spatial_coords(scdata)))

"""
    add_similarities_to_chain_metadata!(metadata, expr)

Add sender-to-receiver cosine similarity columns to chain metadata.

# Arguments
- `metadata::DataFrame`: chain metadata table with sender/receiver ID columns
- `expr::AbstractMatrix{<:Real}`: cell-by-feature matrix

# Returns
- `DataFrame`: input metadata with similarity columns appended

# Methods
- `add_similarities_to_chain_metadata!(metadata, expr)` — explicit expression matrix
- `add_similarities_to_chain_metadata!(metadata, scdata; layer=nothing)` — extracts from `scData`
"""
function add_similarities_to_chain_metadata!(
    metadata::DataFrame,
    expr::AbstractMatrix{<:Real},
)
    all(isfinite, expr) || throw(ArgumentError("expr contains non-finite values"))
    _require_metadata_columns(
        metadata,
        [:first_sender_cell_ids, :receiver_cell_ids, :penultimate_sender_cell_ids],
    )

    n_cells = size(expr, 1)
    _validate_ids_for_lookup(
        metadata[!, :first_sender_cell_ids],
        n_cells,
        "first_sender_cell_ids",
    )
    _validate_ids_for_lookup(metadata[!, :receiver_cell_ids], n_cells, "receiver_cell_ids")
    _validate_ids_for_lookup(
        metadata[!, :penultimate_sender_cell_ids],
        n_cells,
        "penultimate_sender_cell_ids",
    )
    if :max_attention_sender_cell_ids in propertynames(metadata)
        _validate_ids_for_lookup(
            metadata[!, :max_attention_sender_cell_ids],
            n_cells,
            "max_attention_sender_cell_ids",
        )
    end

    n_chains = nrow(metadata)
    first_sender_similarities = Vector{Float64}(undef, n_chains)
    penultimate_sender_similarities = Vector{Union{Float64,Missing}}(undef, n_chains)

    for i = 1:n_chains
        first_sender = metadata[i, :first_sender_cell_ids]
        receiver = metadata[i, :receiver_cell_ids]
        first_sender_similarities[i] =
            _cosine_similarity(view(expr, first_sender, :), view(expr, receiver, :))

        penultimate = metadata[i, :penultimate_sender_cell_ids]
        if ismissing(penultimate)
            penultimate_sender_similarities[i] = missing
        else
            penultimate_sender_similarities[i] =
                _cosine_similarity(view(expr, penultimate, :), view(expr, receiver, :))
        end
    end

    metadata[!, :first_sender_receiver_similarity] = first_sender_similarities
    metadata[!, :penultimate_sender_receiver_similarity] = penultimate_sender_similarities

    if :max_attention_sender_cell_ids in propertynames(metadata)
        max_attention_similarities = Vector{Union{Float64,Missing}}(undef, n_chains)
        for i = 1:n_chains
            sender = metadata[i, :max_attention_sender_cell_ids]
            receiver = metadata[i, :receiver_cell_ids]
            if ismissing(sender)
                max_attention_similarities[i] = missing
            else
                max_attention_similarities[i] =
                    _cosine_similarity(view(expr, sender, :), view(expr, receiver, :))
            end
        end
        metadata[!, :max_attention_sender_receiver_similarity] = max_attention_similarities
    end

    return metadata
end

function add_similarities_to_chain_metadata!(
    metadata::DataFrame,
    scdata::scData;
    layer = nothing,
)
    return add_similarities_to_chain_metadata!(
        metadata,
        Float32.(expression_matrix(scdata; layer = layer)),
    )
end

"""
    subset_chain_metadata!(metadata, error_pct=75; receiver_type=nothing, sender_type=nothing)

Subset metadata to chains selected by an error percentile mask with optional type filters.

# Arguments
- `metadata::DataFrame`: chain metadata table with error mask columns
- `error_pct::Int`: percentile mask column to use (for example `50` uses column `Symbol("50")`)
- `receiver_type`: optional receiver type filter (uses `:receiver_cell_types`)
- `sender_type`: optional sender type filter (uses `:first_sender_cell_types`)

# Returns
- `DataFrame`: filtered metadata rows
"""
function subset_chain_metadata!(
    metadata::DataFrame,
    error_pct::Int = 75;
    receiver_type::Union{Nothing,String} = nothing,
    sender_type::Union{Nothing,String} = nothing,
)
    selector_col = Symbol(string(error_pct))
    selector_col in propertynames(metadata) ||
        throw(ArgumentError("metadata is missing selector column: $selector_col"))

    selection = Vector{Bool}(metadata[!, selector_col])

    if receiver_type !== nothing
        :receiver_cell_types in propertynames(metadata) ||
            throw(ArgumentError("metadata is missing :receiver_cell_types"))
        selection .&= metadata[!, :receiver_cell_types] .== receiver_type
    end

    if sender_type !== nothing
        :first_sender_cell_types in propertynames(metadata) ||
            throw(ArgumentError("metadata is missing :first_sender_cell_types"))
        selection .&= metadata[!, :first_sender_cell_types] .== sender_type
    end

    return metadata[selection, :]
end

export ChainResult,
    sample_chains,
    construct_chain_metadata,
    add_max_attention_to_chain_metadata!,
    add_chain_model_errors_to_metadata!,
    add_distances_to_chain_metadata!,
    add_similarities_to_chain_metadata!,
    subset_chain_metadata!

end # module Chains
