"""
Data loading and ligand-receptor pair database parsing submodule.

Supports CellChat ligand-receptor databases and custom user-provided databases
(CSV and Excel formats).
"""
module IO

using DataFrames
using CSV
using JSON

include("types.jl")
include("scdata_types.jl")
include("scdata_h5ad.jl")
include("preprocessing.jl")
include("example_datasets.jl")
include("paths.jl")
include("cellchat.jl")
include("custom_db.jl")
include("ppi_database.jl")

"""
    load_lrpair_db(path; format=:cellchat, kwargs...)
    load_lrpair_db(; format=:cellchat, species="human", kwargs...)

Load a ligand–receptor pair database from a file.

# Arguments
- `path::String`: path to the database file (positional overload)
- `format::Symbol`: database format — `:cellchat` (default) or `:custom`
- `species::String`: `"human"` or `"mouse"` (used by pathless overload)

# Returns
- [`LRPairDB`](@ref)
"""
function load_lrpair_db(path::String; format::Symbol = :cellchat, kwargs...)
    if format == :cellchat
        return load_cellchat_db(path; kwargs...)
    elseif format == :custom
        return load_custom_lrpair_db(path; kwargs...)
    end

    throw(ArgumentError("Unsupported format: $format. Expected :cellchat or :custom"))
end

function load_lrpair_db(; format::Symbol = :cellchat, species::String = "human", kwargs...)
    if format == :cellchat
        return load_cellchat_db(; species = species, kwargs...)
    end

    throw(ArgumentError("Unsupported format: $format. Expected :cellchat or :custom"))
end

export LRPairRecord, LRPairDB
export scData
export load_scdata
export preprocess_scdata
export expression_matrix, raw_expression_matrix
export obs_table, var_table
export cell_annotation
export spatial_coords
export obsm, varm, obsp, varp
export layers, layer
export uns
export load_example_dataset_manifest, example_dataset_path, resolve_example_dataset
export n_lrpairs, lrpair_names, all_ligands, all_receptors, to_dataframe
export default_cellchat_path
export load_cellchat_db
export load_lrpair_db
export load_custom_lrpair_db
export merge_lrpair_dbs
export load_ppi_database, default_ppi_database_path
export extract_downstream_genes

end # module IO
