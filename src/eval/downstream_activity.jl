# Downstream gene activity scoring, oriented on the CellNEST approach
# (Zohora et al., Nature Methods, 2025; doi:10.1038/s41592-025-02721-3).

"""
    downstream_gene_activity_score(expr, chains, genenames, downstream_genes; chain_inds=nothing) -> Vector{Float64}

Compute the mean expression of downstream genes in each chain's receiver cell.

The receiver cell is the last element of each chain (`chain[end]`).

# Arguments
- `expr::AbstractMatrix{<:Real}`: cells-by-genes expression matrix
- `chains::Vector{<:Vector{<:Integer}}`: sampled communication chains
- `genenames::AbstractVector{<:AbstractString}`: gene names matching columns of `expr`
- `downstream_genes::AbstractVector{<:AbstractString}`: target genes to score
- `chain_inds::Union{Nothing,AbstractVector{Int}}=nothing`: optional subset of chain
  indices to score. Defaults to all chains.

# Returns
- `Vector{Float64}`: one score per selected chain (mean downstream gene expression
  in the receiver cell)
"""
function downstream_gene_activity_score(
    expr::AbstractMatrix{<:Real},
    chains::Vector{<:Vector{<:Integer}},
    genenames::AbstractVector{<:AbstractString},
    downstream_genes::AbstractVector{<:AbstractString};
    chain_inds::Union{Nothing,AbstractVector{Int}} = nothing,
)
    gene_inds = [findfirst(isequal(g), genenames) for g in downstream_genes]
    filter!(!isnothing, gene_inds)

    isempty(gene_inds) &&
        throw(ArgumentError("None of the downstream genes were found in genenames."))

    inds = isnothing(chain_inds) ? eachindex(chains) : chain_inds

    receivers = [last(chains[i]) for i in inds]
    scores = @views mean(expr[receivers, gene_inds], dims = 2)
    return vec(scores)
end

"""
    get_avg_expression(expr, genenames, target_genes) -> (avg_per_cell, matched_genes, quartiles)

Compute per-cell average expression of target genes with quartile summary.

# Arguments
- `expr::AbstractMatrix{<:Real}`: cells-by-genes expression matrix
- `genenames::AbstractVector{<:AbstractString}`: gene names matching columns of `expr`
- `target_genes::AbstractVector{<:AbstractString}`: genes to average

# Returns
- `avg_per_cell::Vector{Float64}`: mean expression per cell across matched genes
- `matched_genes::Vector{String}`: subset of `target_genes` found in `genenames`
- `quartiles::NamedTuple`: `(q25, q50, q75)` of `avg_per_cell` across all cells.
  Returns `(Float64[], String[], (q25=NaN, q50=NaN, q75=NaN))` if no genes match.
"""
function get_avg_expression(
    expr::AbstractMatrix{<:Real},
    genenames::AbstractVector{<:AbstractString},
    target_genes::AbstractVector{<:AbstractString},
)
    tset = Set(target_genes)
    gene_idx = findall(name -> name in tset, genenames)

    if isempty(gene_idx)
        return Float64[], String[], (q25 = NaN, q50 = NaN, q75 = NaN)
    end

    avg_per_cell = vec(mean(expr[:, gene_idx], dims = 2))

    qs = quantile(avg_per_cell, (0.25, 0.50, 0.75))
    quartiles = (q25 = qs[1], q50 = qs[2], q75 = qs[3])

    return avg_per_cell, genenames[gene_idx], quartiles
end

# ── scData convenience overloads ──────────────────────────────────────

function _scdata_gene_names(scdata::scData)
    vt = var_table(scdata)
    if hasproperty(vt, :index)
        return String.(vt.index)
    elseif hasproperty(vt, :gene_symbol)
        return String.(vt.gene_symbol)
    end
    throw(ArgumentError("Cannot resolve gene names from scdata var table"))
end

"""
    downstream_gene_activity_score(scdata, chains, downstream_genes; chain_inds=nothing)

Convenience overload that extracts expression and gene names from `scdata`.
"""
function downstream_gene_activity_score(
    scdata::scData,
    chains::Vector{<:Vector{<:Integer}},
    downstream_genes::AbstractVector{<:AbstractString};
    chain_inds::Union{Nothing,AbstractVector{Int}} = nothing,
)
    expr = expression_matrix(scdata)
    genenames = _scdata_gene_names(scdata)
    return downstream_gene_activity_score(
        expr,
        chains,
        genenames,
        downstream_genes;
        chain_inds = chain_inds,
    )
end

"""
    get_avg_expression(scdata, target_genes; cell_inds=nothing)

Convenience overload that extracts expression and gene names from `scdata`.
Optionally subset rows via `cell_inds`.
"""
function get_avg_expression(
    scdata::scData,
    target_genes::AbstractVector{<:AbstractString};
    cell_inds::Union{Nothing,AbstractVector{Int}} = nothing,
)
    expr = expression_matrix(scdata)
    genenames = _scdata_gene_names(scdata)
    if !isnothing(cell_inds)
        expr = expr[cell_inds, :]
    end
    return get_avg_expression(expr, genenames, target_genes)
end
