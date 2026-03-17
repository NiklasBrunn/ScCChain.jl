"""
    LRPairRecord

A single LR pair entry from a ligand-receptor pair database.

# Fields
- `name::String`: LR pair identifier, e.g. `"TGFB1—TGFBR1_TGFBR2"`
- `ligands::Vector{String}`: ligand gene symbols (multi-subunit if complex)
- `receptors::Vector{String}`: receptor gene symbols (multi-subunit if complex)
- `pathway::Union{String,Nothing}`: signaling pathway name, or `nothing`
- `metadata::Dict{String,Any}`: format-specific extras (annotation, version, etc.)
"""
struct LRPairRecord
    name::String
    ligands::Vector{String}
    receptors::Vector{String}
    pathway::Union{String,Nothing}
    metadata::Dict{String,Any}
end

"""
    LRPairDB

A collection of [`LRPairRecord`](@ref)s loaded from a ligand-receptor pair database.

# Fields
- `records::Vector{LRPairRecord}`: the LR pair entries
- `source::Symbol`: database origin (currently `:cellchat`)
- `species::String`: `"human"` or `"mouse"`
"""
struct LRPairDB
    records::Vector{LRPairRecord}
    source::Symbol
    species::String
end

"""
    n_lrpairs(db::LRPairDB) -> Int

Return the number of LR pairs in the database.
"""
n_lrpairs(db::LRPairDB) = length(db.records)

"""
    lrpair_names(db::LRPairDB) -> Vector{String}

Return LR pair names in order.
"""
lrpair_names(db::LRPairDB) = [r.name for r in db.records]

"""
    all_ligands(db::LRPairDB) -> Vector{String}

Return all unique ligand gene symbols across all LR pairs.
"""
all_ligands(db::LRPairDB) =
    unique(collect(Iterators.flatten(r.ligands for r in db.records)))

"""
    all_receptors(db::LRPairDB) -> Vector{String}

Return all unique receptor gene symbols across all LR pairs.
"""
all_receptors(db::LRPairDB) =
    unique(collect(Iterators.flatten(r.receptors for r in db.records)))

"""
    to_dataframe(db::LRPairDB) -> DataFrame

Convert to a DataFrame with columns: `lrpair_name`, `ligands`, `receptors`, `pathway`.
"""
function to_dataframe(db::LRPairDB)
    return DataFrame(
        lrpair_name = [r.name for r in db.records],
        ligands = [r.ligands for r in db.records],
        receptors = [r.receptors for r in db.records],
        pathway = [r.pathway for r in db.records],
    )
end
