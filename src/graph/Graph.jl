"""
Cell graph construction submodule.

Builds a multi-layer weighted cell graph G = (V, L, E, w) where:
- V = cells/spots/bins
- L = {sim} ∪ {ℓ₁, …, ℓₖ} (similarity layer + one layer per LR pair)
- Edges are weighted by spatial distance and expression co-scores

The graph construction approach is oriented on the NICHES framework
(Raredon et al., Bioinformatics, 2023; doi:10.1093/bioinformatics/btac775).
"""
module Graph

using LinearAlgebra
using SparseArrays
using Statistics
using Graphs
using SimpleWeightedGraphs
using NearestNeighbors
using Distances
using DataFrames
using ..IO: scData, expression_matrix, spatial_coords, var_table, LRPairDB

struct CellGraph
    similarity_layer::SparseMatrixCSC{Float32,Int}
    communication_layers::Vector{SparseMatrixCSC{Float32,Int}}
    communication_names::Vector{String}
    kept_communication_indices::Vector{Int}
    dropped_communication_indices::Vector{Int}
    dropped_communication_reasons::Vector{String}
    metadata::Dict{String,Any}
end

function _validate_graph_inputs(
    expr::AbstractMatrix{<:Real},
    coords::AbstractMatrix{<:Real},
    gene_names::Union{Nothing,AbstractVector{<:AbstractString}};
    n_neighbors::Union{Nothing,Int},
    radius::Union{Nothing,Real},
    alpha::Real,
)
    size(expr, 1) == size(coords, 1) ||
        throw(ArgumentError("expr rows must equal coords rows"))
    size(coords, 2) == 2 || throw(ArgumentError("coords must be n_cells x 2"))
    isnothing(gene_names) && throw(ArgumentError("Matrix API requires `gene_names`"))
    length(gene_names) == size(expr, 2) ||
        throw(ArgumentError("gene_names length mismatch"))
    isnothing(radius) || radius > 0 || throw(ArgumentError("radius must be positive"))
    isnothing(n_neighbors) ||
        n_neighbors > 0 ||
        throw(ArgumentError("n_neighbors must be positive"))
    isnothing(radius) ||
        isnothing(n_neighbors) ||
        throw(ArgumentError("Set only one of n_neighbors or radius"))
    alpha >= 0 || throw(ArgumentError("alpha must be non-negative"))
    all(isfinite, coords) || throw(ArgumentError("coords must be finite"))
    return nothing
end

function _candidate_edges(
    coords::AbstractMatrix{T};
    n_neighbors::Union{Nothing,Int},
    radius::Union{Nothing,Real},
) where {T<:Real}
    n_cells = size(coords, 1)
    tree = KDTree(permutedims(coords))
    FT = float(T)

    rows = Int[]
    cols = Int[]
    dists = FT[]

    if !isnothing(radius)
        for i = 1:n_cells
            neigh = inrange(tree, @view(coords[i, :]), radius)
            for j in neigh
                push!(rows, i)
                push!(cols, j)
                push!(dists, FT(euclidean(@view(coords[i, :]), @view(coords[j, :]))))
            end
        end
        return rows, cols, dists
    end

    if isnothing(n_neighbors) || n_cells <= 1
        return rows, cols, dists
    end

    k = min(n_cells, max(1, n_neighbors + 1))
    for i = 1:n_cells
        idxs, ds = knn(tree, @view(coords[i, :]), k, true)
        kept = 0
        for t in eachindex(idxs)
            j = idxs[t]
            if j != i
                kept += 1
                kept <= n_neighbors || continue
            end
            push!(rows, i)
            push!(cols, j)
            push!(dists, FT(ds[t]))
        end
    end
    return rows, cols, dists
end

function _cosine_similarity_row(expr::AbstractMatrix{T}, i::Int, j::Int) where {T<:Real}
    xi = @view expr[i, :]
    xj = @view expr[j, :]
    FT = float(T)
    denom = sqrt(sum(abs2, xi)) * sqrt(sum(abs2, xj))
    denom <= eps(FT) && return zero(FT)
    return FT(dot(xi, xj) / (denom + eps(FT)))
end

function _to_float32_sparse(mat::SparseMatrixCSC{<:Real,Int})
    return SparseMatrixCSC{Float32,Int}(
        mat.m,
        mat.n,
        copy(mat.colptr),
        copy(mat.rowval),
        Float32.(mat.nzval),
    )
end

function _symmetric_sparse_max(
    rows::Vector{Int},
    cols::Vector{Int},
    vals::Vector{T},
    n_cells::Int,
) where {T<:Real}
    pair_weights = Dict{Tuple{Int,Int},T}()

    @inbounds for k in eachindex(rows)
        i = rows[k]
        j = cols[k]
        key = i <= j ? (i, j) : (j, i)
        v = vals[k]
        if !haskey(pair_weights, key) || v > pair_weights[key]
            pair_weights[key] = v
        end
    end

    sym_rows = Int[]
    sym_cols = Int[]
    sym_vals = T[]
    sizehint!(sym_rows, 2 * length(pair_weights))
    sizehint!(sym_cols, 2 * length(pair_weights))
    sizehint!(sym_vals, 2 * length(pair_weights))

    for ((i, j), v) in pair_weights
        push!(sym_rows, i)
        push!(sym_cols, j)
        push!(sym_vals, v)
        if i != j
            push!(sym_rows, j)
            push!(sym_cols, i)
            push!(sym_vals, v)
        end
    end

    return sparse(sym_rows, sym_cols, sym_vals, n_cells, n_cells)
end

function _build_similarity_layer(
    expr::AbstractMatrix{T},
    rows::Vector{Int},
    cols::Vector{Int},
    dists::Vector{<:Real},
    n_cells::Int,
    alpha::Float64;
    allow_self_loops::Bool,
    dim_red::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
    symmetrize::Bool = true,
) where {T<:Real}
    FT = float(T)
    if isempty(rows)
        return spzeros(Float32, n_cells, n_cells)
    end

    sim_input = isnothing(dim_red) ? expr : dim_red

    vals = Vector{FT}(undef, length(rows))
    for i in eachindex(rows)
        distance_weight = FT(exp(-alpha * dists[i]^2))
        vals[i] = _cosine_similarity_row(sim_input, rows[i], cols[i]) * distance_weight
    end

    if symmetrize
        sim = _symmetric_sparse_max(rows, cols, vals, n_cells)
    else
        sim = sparse(rows, cols, vals, n_cells, n_cells)
    end
    if !allow_self_loops
        sim[diagind(sim)] .= zero(FT)
        dropzeros!(sim)
    end
    return _to_float32_sparse(sim)
end

function _geometric_mean(values::Vector{T}) where {T<:Real}
    isempty(values) && return zero(T)
    return prod(values)^(one(T) / length(values))
end

function _communication_score_mode(mode_raw)
    mode_str = if mode_raw isa Symbol
        String(mode_raw)
    elseif mode_raw isa AbstractString
        String(mode_raw)
    else
        throw(
            ArgumentError(
                "communication_score_mode must be Symbol or String; got $(typeof(mode_raw))",
            ),
        )
    end
    mode = Symbol(lowercase(mode_str))
    mode in (:geometric_mean, :product, :minimum) || throw(
        ArgumentError(
            "communication_score_mode must be one of: geometric_mean, product, minimum",
        ),
    )
    return mode
end

function _communication_score(
    lig_vals::Vector{T},
    rec_vals::Vector{T},
    mode::Symbol,
) where {T<:Real}
    if mode === :geometric_mean
        return sqrt(_geometric_mean(lig_vals) * _geometric_mean(rec_vals))
    end
    if mode === :product
        return prod(lig_vals) * prod(rec_vals)
    end
    if mode === :minimum
        if isempty(lig_vals) || isempty(rec_vals)
            return zero(T)
        end
        return min(minimum(lig_vals), minimum(rec_vals))
    end
    throw(ArgumentError("Unsupported communication score mode: $(mode)"))
end

function _build_communication_layers(
    expr::AbstractMatrix{T},
    lrpair_db::LRPairDB,
    gene_names::Vector{String},
    rows::Vector{Int},
    cols::Vector{Int},
    dists::Vector{<:Real},
    n_cells::Int,
    alpha::Float64;
    communication_score_mode::Symbol,
    allow_self_loops::Bool,
    filter_gene_names::Union{Nothing,AbstractVector{<:AbstractString}} = nothing,
) where {T<:Real}
    FT = float(T)
    gene_name_set = Set(gene_names)
    filter_set =
        isnothing(filter_gene_names) ? gene_name_set : Set(String.(filter_gene_names))

    communication_layers = SparseMatrixCSC{Float32,Int}[]
    communication_names = String[]
    kept_communication_indices = Int[]
    dropped_communication_indices = Int[]
    dropped_communication_reasons = String[]

    distance_weights = FT.(exp.(-alpha .* (dists .^ 2)))

    for (idx, rec) in enumerate(lrpair_db.records)
        ligand_set = Set(rec.ligands)
        receptor_set = Set(rec.receptors)

        # Check gene presence against filter set (e.g. HVGs) if provided
        missing_filter_lig = setdiff(ligand_set, filter_set)
        missing_filter_rec = setdiff(receptor_set, filter_set)
        if !isempty(missing_filter_lig) || !isempty(missing_filter_rec)
            push!(dropped_communication_indices, idx)
            push!(
                dropped_communication_reasons,
                "Missing ligand or receptor genes in filter set",
            )
            continue
        end

        # Also check that genes exist in the expression matrix gene names
        missing_ligands = setdiff(ligand_set, gene_name_set)
        missing_receptors = setdiff(receptor_set, gene_name_set)
        if !isempty(missing_ligands) || !isempty(missing_receptors)
            push!(dropped_communication_indices, idx)
            push!(dropped_communication_reasons, "Missing ligand or receptor genes")
            continue
        end

        lig_inds = findall(g -> g in ligand_set, gene_names)
        rec_inds = findall(g -> g in receptor_set, gene_names)
        if isempty(lig_inds) || isempty(rec_inds)
            push!(dropped_communication_indices, idx)
            push!(dropped_communication_reasons, "Missing ligand or receptor genes")
            continue
        end

        vals = Vector{FT}(undef, length(rows))
        for k in eachindex(rows)
            sender = rows[k]
            receiver = cols[k]
            lig_vals = FT[expr[sender, g] for g in lig_inds]
            rec_vals = FT[expr[receiver, g] for g in rec_inds]
            vals[k] =
                _communication_score(lig_vals, rec_vals, communication_score_mode) *
                distance_weights[k]
        end

        layer = sparse(rows, cols, vals, n_cells, n_cells)
        if !allow_self_loops
            layer[diagind(layer)] .= zero(FT)
            dropzeros!(layer)
        end

        push!(communication_layers, _to_float32_sparse(layer))
        push!(communication_names, rec.name)
        push!(kept_communication_indices, idx)
    end

    return communication_layers,
    communication_names,
    kept_communication_indices,
    dropped_communication_indices,
    dropped_communication_reasons
end

function _scdata_gene_names(scdata::scData)
    var = var_table(scdata)
    col_names = String.(names(var))
    if "index" in col_names
        return String.(var[!, Symbol("index")])
    end
    if "gene_symbol" in col_names
        return String.(var[!, Symbol("gene_symbol")])
    end
    throw(
        ArgumentError("Could not resolve gene names from var index or gene_symbol column"),
    )
end

"""
    build_cell_graph(expr, coords, lrpair_db; kwargs...)

Build a multi-layer weighted cell graph from expression data, spatial coordinates,
and a ligand–receptor pair database.

# Arguments
- `expr::AbstractMatrix{<:Real}`: log-normalized expression matrix (n_cells × n_genes)
- `coords::AbstractMatrix{<:Real}`: spatial coordinates in µm (n_cells × 2)
- `lrpair_db::LRPairDB`: ligand–receptor pair database
- `gene_names`: vector of gene names aligned to expression matrix columns (required)
- `dim_red`: optional pre-computed dimensionality reduction (n_cells × k) used for
  similarity layer computation instead of `expr` (e.g. PCA of HVGs)
- `filter_gene_names`: optional gene name vector for communication filtering; only
  LR pairs whose ligand AND receptor genes all appear in this set are kept.
  Communication scores are still computed using `expr` and `gene_names`.
- `symmetrize_similarity::Bool=true`: if `false`, skip similarity layer symmetrization
- `promote_float64::Bool=true`: if `true`, promote inputs to Float64 for numerical
  stability; set to `false` to operate in the input element type (e.g. Float32)

# Returns
- `CellGraph` with similarity layer, communication layers, and metadata
"""
function build_cell_graph(expr, coords, lrpair_db; kwargs...)
    gene_names = get(kwargs, :gene_names, nothing)
    n_neighbors =
        haskey(kwargs, :radius) && !haskey(kwargs, :n_neighbors) ? nothing :
        get(kwargs, :n_neighbors, 50)
    radius = get(kwargs, :radius, nothing)
    alpha = Float64(get(kwargs, :alpha, 0.0))
    communication_score_mode =
        _communication_score_mode(get(kwargs, :communication_score_mode, :geometric_mean))
    allow_self_loops_similarity = Bool(get(kwargs, :allow_self_loops_similarity, true))
    allow_self_loops_communication =
        Bool(get(kwargs, :allow_self_loops_communication, true))
    dim_red = get(kwargs, :dim_red, nothing)
    filter_gene_names = get(kwargs, :filter_gene_names, nothing)
    symmetrize_similarity = Bool(get(kwargs, :symmetrize_similarity, true))
    promote_float64 = Bool(get(kwargs, :promote_float64, true))

    gene_names_vec = isnothing(gene_names) ? nothing : String.(gene_names)
    n_neighbors_val = isnothing(n_neighbors) ? nothing : Int(n_neighbors)
    radius_val = isnothing(radius) ? nothing : Float64(radius)

    _validate_graph_inputs(
        expr,
        coords,
        gene_names_vec;
        n_neighbors = n_neighbors_val,
        radius = radius_val,
        alpha = alpha,
    )

    if !isnothing(dim_red)
        size(dim_red, 1) == size(expr, 1) ||
            throw(ArgumentError("dim_red rows must match expr rows"))
    end

    n_cells = size(expr, 1)
    if promote_float64
        expr_f = Float64.(expr)
        coords_f = Float64.(coords)
        dim_red_f = isnothing(dim_red) ? nothing : Float64.(dim_red)
    else
        expr_f = float.(expr)
        coords_f = float.(coords)
        dim_red_f = isnothing(dim_red) ? nothing : float.(dim_red)
    end

    candidate_rows, candidate_cols, candidate_dists =
        _candidate_edges(coords_f; n_neighbors = n_neighbors_val, radius = radius_val)
    similarity_layer = _build_similarity_layer(
        expr_f,
        candidate_rows,
        candidate_cols,
        candidate_dists,
        n_cells,
        alpha;
        allow_self_loops = allow_self_loops_similarity,
        dim_red = dim_red_f,
        symmetrize = symmetrize_similarity,
    )

    communication_layers,
    communication_names,
    kept_communication_indices,
    dropped_communication_indices,
    dropped_communication_reasons = _build_communication_layers(
        expr_f,
        lrpair_db,
        gene_names_vec,
        candidate_rows,
        candidate_cols,
        candidate_dists,
        n_cells,
        alpha;
        communication_score_mode = communication_score_mode,
        allow_self_loops = allow_self_loops_communication,
        filter_gene_names = filter_gene_names,
    )

    # Store distance scores sparse matrix for pair feature matrix row ordering
    gk_vals = exp.(-alpha .* (candidate_dists .^ 2))
    _distance_scores = sparse(candidate_rows, candidate_cols, gk_vals, n_cells, n_cells)

    metadata = Dict{String,Any}(
        "n_neighbors" => n_neighbors_val,
        "radius" => radius_val,
        "alpha" => alpha,
        "communication_score_mode" => String(communication_score_mode),
        "allow_self_loops_similarity" => allow_self_loops_similarity,
        "allow_self_loops_communication" => allow_self_loops_communication,
        "dim_red" => !isnothing(dim_red),
        "filter_gene_names" => !isnothing(filter_gene_names),
        "symmetrize_similarity" => symmetrize_similarity,
        "_distance_scores" => _distance_scores,
    )

    return CellGraph(
        similarity_layer,
        communication_layers,
        communication_names,
        kept_communication_indices,
        dropped_communication_indices,
        dropped_communication_reasons,
        metadata,
    )
end

"""
    build_cell_graph(scdata::scData, lrpair_db::LRPairDB; kwargs...)

Dispatch overload that extracts expression and coordinates from `scdata` and forwards
to the matrix-based `build_cell_graph` implementation.

# Arguments
- `scdata::scData`: spatial data handle
- `lrpair_db::LRPairDB`: ligand–receptor pair database

# Returns
- Same as matrix-based `build_cell_graph(expr, coords, lrpair_db; kwargs...)`
"""
function build_cell_graph(scdata::scData, lrpair_db::LRPairDB; kwargs...)
    expr = expression_matrix(scdata)
    coords = spatial_coords(scdata)
    if haskey(kwargs, :gene_names)
        return build_cell_graph(expr, coords, lrpair_db; kwargs...)
    end
    gene_names = _scdata_gene_names(scdata)
    return build_cell_graph(expr, coords, lrpair_db; kwargs..., gene_names = gene_names)
end

export CellGraph, build_cell_graph

end # module Graph
