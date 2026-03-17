# Internal helper: split comma-separated gene symbols into a Vector{String}
function _split_genes(s::AbstractString)
    genes = [String(strip(x)) for x in split(s, ",")]
    filter!(!isempty, genes)
    return genes
end

"""
    load_cellchat_db(path; species="human", communication_type=nothing, pathways=nothing, lrpairs=nothing)
    load_cellchat_db(; species="human", communication_type=nothing, pathways=nothing, lrpairs=nothing)

Load CellChat ligand-receptor database from a CSV file.

All filters are applied sequentially (AND logic) in the order: `communication_type`,
`pathways`, `lrpairs`.

# Arguments
- `path::String`: path to the CellChat CSV file
- `species::String`: `"human"` or `"mouse"` (stored in the returned `LRPairDB`)
- `communication_type::Union{Nothing,String,Vector{String}}`: if provided, filter to
  LR pairs whose `annotation` column matches one of the given types
  (e.g. `"Secreted Signaling"`)
- `pathways::Union{Nothing,String,Vector{String}}`: if provided, filter to LR pairs
  whose pathway name matches one of the given values (e.g. `"CXCL"`)
- `lrpairs::Union{Nothing,String,Vector{String}}`: if provided, filter to
  LR pairs whose name matches one of the given values (e.g. `"CXCL12—CXCR4"`)

# Returns
- [`LRPairDB`](@ref) with `source = :cellchat`
"""
function load_cellchat_db(
    path::String;
    species::String = "human",
    communication_type::Union{Nothing,String,Vector{String}} = nothing,
    pathways::Union{Nothing,String,Vector{String}} = nothing,
    lrpairs::Union{Nothing,String,Vector{String}} = nothing,
)

    species in ("human", "mouse") ||
        throw(ArgumentError("species must be \"human\" or \"mouse\", got \"$species\""))
    isfile(path) || throw(ArgumentError("File not found: $path"))

    df = CSV.read(path, DataFrame)

    records = LRPairRecord[]
    for row in eachrow(df)
        ligand_str = ismissing(row.ligand_symbol) ? "" : String(row.ligand_symbol)
        receptor_str = ismissing(row.receptor_symbol) ? "" : String(row.receptor_symbol)

        ligands = _split_genes(ligand_str)
        receptors = _split_genes(receptor_str)

        # LR pair name: ligand subunits joined by _, em-dash, receptor subunits joined by _
        name = join(ligands, "_") * "—" * join(receptors, "_")

        pathway =
            hasproperty(row, :pathway_name) && !ismissing(row.pathway_name) ?
            String(row.pathway_name) : nothing

        meta = Dict{String,Any}()
        if hasproperty(row, :annotation)
            meta["annotation"] = ismissing(row.annotation) ? "" : String(row.annotation)
        end
        if hasproperty(row, :version) && !ismissing(row.version)
            meta["version"] = String(row.version)
        end

        push!(records, LRPairRecord(name, ligands, receptors, pathway, meta))
    end

    if communication_type !== nothing
        allowed =
            communication_type isa String ? Set([communication_type]) :
            Set(communication_type)
        filter!(r -> get(r.metadata, "annotation", "") in allowed, records)
    end

    if pathways !== nothing
        allowed = pathways isa String ? Set([pathways]) : Set(pathways)
        filter!(r -> r.pathway !== nothing && r.pathway in allowed, records)
    end

    if lrpairs !== nothing
        allowed = lrpairs isa String ? Set([lrpairs]) : Set(lrpairs)
        filter!(r -> r.name in allowed, records)
    end

    return LRPairDB(records, :cellchat, species)
end

"""
    load_cellchat_db(; species="human", communication_type=nothing, pathways=nothing, lrpairs=nothing)

Load the bundled CellChat ligand-receptor database CSV for `species`.
"""
function load_cellchat_db(;
    species::String = "human",
    communication_type::Union{Nothing,String,Vector{String}} = nothing,
    pathways::Union{Nothing,String,Vector{String}} = nothing,
    lrpairs::Union{Nothing,String,Vector{String}} = nothing,
)
    path = default_cellchat_path(species)
    isfile(path) || throw(ArgumentError("Bundled CellChat database not found: $path"))
    return load_cellchat_db(
        path;
        species = species,
        communication_type = communication_type,
        pathways = pathways,
        lrpairs = lrpairs,
    )
end
