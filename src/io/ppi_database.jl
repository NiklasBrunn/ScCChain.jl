_PPI_DATABASE_ROOT =
    normpath(joinpath(@__DIR__, "..", "..", "data", "databases", "CellNEST"))

"""
    default_ppi_database_path(species::String) -> String

Return the bundled CellNEST PPI database CSV path for `species`.

# Arguments
- `species::String`: `"human"` or `"mouse"`

# Returns
- Path to the bundled CSV file
"""
function default_ppi_database_path(species::String)
    species in ("human", "mouse") ||
        throw(ArgumentError("species must be \"human\" or \"mouse\", got \"$species\""))
    return joinpath(_PPI_DATABASE_ROOT, "$(species)_signaling_ppi.csv")
end

"""
    load_ppi_database(species::String; data_path=nothing) -> DataFrame

Load a CellNEST protein-protein interaction (PPI) database.

# Arguments
- `species::String`: `"human"` or `"mouse"`
- `data_path::Union{String,Nothing}=nothing`: custom directory containing
  `{species}_signaling_ppi.csv`. Uses bundled data when `nothing`.

# Returns
- `DataFrame` with columns `:source`, `:target`, `:experimental_score`
"""
function load_ppi_database(species::String; data_path::Union{String,Nothing} = nothing)
    species in ("human", "mouse") ||
        throw(ArgumentError("species must be \"human\" or \"mouse\", got \"$species\""))

    if isnothing(data_path)
        path = default_ppi_database_path(species)
    else
        path = joinpath(data_path, "$(species)_signaling_ppi.csv")
    end

    isfile(path) || throw(ArgumentError("PPI database file not found: $path"))
    return CSV.read(path, DataFrame)
end

"""
    extract_downstream_genes(ppi::DataFrame, source_genes; top_n=10, top_percent=nothing, include_immediate=true, verbose=false) -> Vector{String}

Collect second-order downstream genes from a PPI network.

Traverses two hops: source -> immediate targets -> downstream targets. Downstream
targets are ranked by their maximum experimental score across all paths.

# Arguments
- `ppi::DataFrame`: PPI database with columns `:source`, `:target`, `:experimental_score`
- `source_genes::Union{String,Vector{String}}`: receptor gene(s) to query
- `top_n::Union{Int,Nothing}=10`: number of top downstream genes to return.
  Takes precedence if both `top_n` and `top_percent` are given.
- `top_percent::Union{Real,Nothing}=nothing`: percentage of downstream genes to return
- `include_immediate::Bool=true`: whether to include direct targets in the result
- `verbose::Bool=false`: print info messages

# Returns
- `Vector{String}` of gene names
"""
function extract_downstream_genes(
    ppi::DataFrame,
    source_genes::Union{String,Vector{String}};
    top_n::Union{Int,Nothing} = 10,
    top_percent::Union{Real,Nothing} = nothing,
    include_immediate::Bool = true,
    verbose::Bool = false,
)
    isnothing(top_percent) &&
        isnothing(top_n) &&
        throw(ArgumentError("Either top_percent or top_n must be provided."))

    source_set = source_genes isa String ? [source_genes] : collect(String, source_genes)

    # Step 1: immediate targets of the source genes
    immediate_df = filter(row -> row.source in source_set, ppi)
    immediate_genes = unique(immediate_df.target)

    if isempty(immediate_genes)
        @warn "No immediate targets found for the provided source genes."
        return String[]
    end

    # Step 2: second-order downstream targets
    downstream_edges = filter(row -> row.source in immediate_genes, ppi)

    if nrow(downstream_edges) == 0
        @warn "No downstream targets found from immediate genes."
        return include_immediate ? collect(String, immediate_genes) : String[]
    end

    # Step 3: deduplicate by max score per target
    downstream_df = select(downstream_edges, :target, :experimental_score => :score)
    agg = combine(groupby(downstream_df, :target), :score => maximum => :max_score)

    # Step 4: sort descending by score
    sorted_agg = sort(agg, :max_score, rev = true)

    # Step 5: select top genes
    if !isnothing(top_percent) && !isnothing(top_n)
        @warn "Both top_percent and top_n provided; using top_n."
        n_select = min(top_n, nrow(sorted_agg))
    elseif !isnothing(top_percent)
        pct = float(top_percent)
        n_select = Int(floor(nrow(sorted_agg) * pct / 100))
        if n_select == 0 && nrow(sorted_agg) > 0 && pct > 0
            n_select = 1
        end
    else
        n_select = min(top_n, nrow(sorted_agg))
    end

    if n_select <= 0
        @warn "Top selection size computed as 0; returning immediate-only set if requested."
        return include_immediate ? collect(String, immediate_genes) : String[]
    end

    top_genes = sorted_agg.target[1:n_select]

    # Step 6: optionally include immediate targets
    result =
        include_immediate ? unique(vcat(immediate_genes, top_genes)) :
        collect(String, top_genes)

    if verbose
        @info "Collected $(length(result)) gene(s): $(length(source_set)) source(s), " *
              "$(length(immediate_genes)) immediate, $(length(top_genes)) top downstream."
    end

    return collect(String, result)
end
