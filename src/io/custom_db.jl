"""
    load_custom_lrpair_db(path; ligand_col=:ligand, receptor_col=:receptor, kwargs...)

Load a custom ligand-receptor pair database from a CSV or Excel file.

Column names are mapped via keyword arguments. Multi-subunit complexes should be
represented as comma-separated gene symbols within a single cell.

# Arguments
- `path::String`: path to the CSV (`.csv`) or Excel (`.xlsx`, `.xls`) file
- `ligand_col::Union{Symbol,String}`: column name for ligand gene symbols (default `:ligand`)
- `receptor_col::Union{Symbol,String}`: column name for receptor gene symbols (default `:receptor`)
- `lrpair_name_col::Union{Symbol,String,Nothing}`: column for LR pair names;
  if `nothing` (default), names are auto-generated as `"ligand(s)—receptor(s)"`
- `pathway_col::Union{Symbol,String,Nothing}`: column for pathway names;
  if `nothing` (default), pathway is set to `nothing`
- `species::String`: species label (default `"human"`, unrestricted)
- `source::Symbol`: database origin label (default `:custom`)
- `extend::Union{LRPairDB,Nothing}`: if provided, merge with this existing DB
  (custom entries win on duplicate LR pair names)

# Returns
- [`LRPairDB`](@ref)
"""
function load_custom_lrpair_db(
    path::String;
    ligand_col::Union{Symbol,String} = :ligand,
    receptor_col::Union{Symbol,String} = :receptor,
    lrpair_name_col::Union{Symbol,String,Nothing} = nothing,
    pathway_col::Union{Symbol,String,Nothing} = nothing,
    species::String = "human",
    source::Symbol = :custom,
    extend::Union{LRPairDB,Nothing} = nothing,
)
    isfile(path) || throw(ArgumentError("File not found: $path"))
    isempty(species) && throw(ArgumentError("species must be a non-empty string"))

    df = _read_custom_db_file(path)

    # Convert column names to Symbol
    lcol = Symbol(ligand_col)
    rcol = Symbol(receptor_col)
    ncol = lrpair_name_col === nothing ? nothing : Symbol(lrpair_name_col)
    pcol = pathway_col === nothing ? nothing : Symbol(pathway_col)

    # Validate required columns
    colnames = Symbol.(names(df))
    if !(lcol in colnames)
        throw(
            ArgumentError(
                "Ligand column :$lcol not found. Available columns: $(join(colnames, ", "))",
            ),
        )
    end
    if !(rcol in colnames)
        throw(
            ArgumentError(
                "Receptor column :$rcol not found. Available columns: $(join(colnames, ", "))",
            ),
        )
    end
    if ncol !== nothing && !(ncol in colnames)
        throw(
            ArgumentError(
                "LR pair name column :$ncol not found. Available columns: $(join(colnames, ", "))",
            ),
        )
    end
    if pcol !== nothing && !(pcol in colnames)
        throw(
            ArgumentError(
                "Pathway column :$pcol not found. Available columns: $(join(colnames, ", "))",
            ),
        )
    end

    records = LRPairRecord[]
    for (i, row) in enumerate(eachrow(df))
        ligand_str = ismissing(row[lcol]) ? "" : String(strip(string(row[lcol])))
        receptor_str = ismissing(row[rcol]) ? "" : String(strip(string(row[rcol])))

        ligands = _split_genes(ligand_str)
        receptors = _split_genes(receptor_str)

        if isempty(ligands) || isempty(receptors)
            @warn "Skipping row $i: empty ligand or receptor after parsing"
            continue
        end

        # LR pair name
        if ncol !== nothing && !ismissing(row[ncol])
            name = String(strip(string(row[ncol])))
        else
            name = join(ligands, "_") * "—" * join(receptors, "_")
        end

        # Pathway
        pathway = if pcol !== nothing && !ismissing(row[pcol])
            String(strip(string(row[pcol])))
        else
            nothing
        end

        push!(records, LRPairRecord(name, ligands, receptors, pathway, Dict{String,Any}()))
    end

    db = LRPairDB(records, source, species)

    if extend !== nothing
        return merge_lrpair_dbs(extend, db)
    end

    return db
end

# Internal: read CSV or Excel file into a DataFrame
function _read_custom_db_file(path::String)
    ext = lowercase(splitext(path)[2])
    if ext in (".xlsx", ".xls")
        if !isdefined(Main, :XLSX)
            try
                @eval Main using XLSX
            catch
                throw(
                    ArgumentError(
                        "Reading Excel files requires XLSX.jl. " *
                        "Install with: using Pkg; Pkg.add(\"XLSX\")",
                    ),
                )
            end
        end
        return DataFrame(Main.XLSX.readtable(path, 1))
    else
        return CSV.read(path, DataFrame)
    end
end

"""
    merge_lrpair_dbs(dbs...; source=:merged, species=nothing)

Merge two or more [`LRPairDB`](@ref)s into a single database.

Duplicates (by LR pair name) are resolved by keeping the entry from the last
database in argument order. An `@info` message is logged for each replaced duplicate.

# Arguments
- `dbs::LRPairDB...`: two or more databases to merge
- `source::Symbol`: source label for the merged DB (default `:merged`)
- `species::Union{String,Nothing}`: species label; if `nothing` (default), inferred
  from the input DBs (must all agree, otherwise an error is thrown)

# Returns
- [`LRPairDB`](@ref) with deduplicated records
"""
function merge_lrpair_dbs(
    dbs::LRPairDB...;
    source::Symbol = :merged,
    species::Union{String,Nothing} = nothing,
)
    length(dbs) >= 2 || throw(ArgumentError("At least two LRPairDBs required for merging"))

    # Resolve species
    if species === nothing
        all_species = unique([db.species for db in dbs])
        if length(all_species) != 1
            throw(
                ArgumentError(
                    "Species mismatch across databases: $(join(all_species, ", ")). " *
                    "Provide an explicit `species` keyword to override.",
                ),
            )
        end
        species = all_species[1]
    end

    # Merge with last-wins dedup
    seen = Dict{String,LRPairRecord}()
    order = String[]
    for db in dbs
        for rec in db.records
            if haskey(seen, rec.name)
                @info "Replacing duplicate LR pair: $(rec.name)"
            else
                push!(order, rec.name)
            end
            seen[rec.name] = rec
        end
    end

    records = [seen[name] for name in order]
    return LRPairDB(records, source, species)
end
