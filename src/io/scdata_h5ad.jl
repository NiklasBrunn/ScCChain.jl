using PythonCall

function _get_h5ad_handle(scdata::scData)
    scdata.backend == :h5ad ||
        throw(ArgumentError("Unsupported scData backend: $(scdata.backend)"))
    return scdata.handle
end

function _to_dense_matrix(x; slot::String)
    obj = pyhasattr(x, "toarray") ? x.toarray() : x
    arr = pyconvert(Array, obj)
    ndims(arr) == 2 || throw(ArgumentError("`$slot` must be a 2D matrix-like object"))
    return arr
end

function _to_mapping_dict(mapping)
    out = Dict{String,Any}()
    for key_py in mapping.keys()
        key = pyconvert(String, key_py)
        value_py = mapping.__getitem__(key_py)
        value = try
            pyconvert(Any, value_py)
        catch
            value_py
        end
        out[key] = value
    end
    return out
end

function _mapping_keys(mapping)
    return [pyconvert(String, key_py) for key_py in mapping.keys()]
end

function _get_matrix_slot(handle, slot_name::String, key::String)
    mapping = getproperty(handle, Symbol(slot_name))
    has_key = pyconvert(Bool, mapping.__contains__(key))
    has_key || throw(ArgumentError("AnnData `$slot_name` is missing key `$key`"))
    return _to_dense_matrix(mapping.__getitem__(key); slot = "$slot_name[$key]")
end

function _pandas_to_dataframe(df_py)
    col_names = [pyconvert(String, c) for c in df_py.columns.tolist()]
    n_rows = pyconvert(Int, df_py.shape[0])
    data = Dict{Symbol,Vector{Any}}()
    index_vals = pyconvert(Vector{Any}, df_py.index.tolist())
    length(index_vals) == n_rows ||
        throw(ArgumentError("Pandas DataFrame index length does not match row count"))
    data[Symbol("index")] = index_vals

    for col_name in col_names
        values = pyconvert(Vector{Any}, df_py.__getitem__(col_name).tolist())
        length(values) == n_rows ||
            throw(ArgumentError("Pandas DataFrame column `$col_name` length mismatch"))
        data[Symbol(col_name)] = values
    end

    return DataFrame(data)
end

function _py_is_none(x)
    none_type = pybuiltins.type(getproperty(pybuiltins, Symbol("None")))
    return pyconvert(Bool, pybuiltins.isinstance(x, none_type))
end

"""
    load_scdata(path; format=:h5ad, strict=true,
                 subset_highly_variable=false,
                 highly_variable_column="highly_variable",
                 kwargs...)

Load spatial transcriptomics data into a [`scData`](@ref).

# Arguments
- `path::String`: path to the spatial data file
- `format::Symbol`: currently supports `:h5ad`
- `strict::Bool`: if `true`, validates required AnnData slots including spatial coordinates
- `subset_highly_variable::Bool`: if `true`, subset features to rows where
  `var[highly_variable_column]` is truthy before constructing the handle
- `highly_variable_column::String`: feature metadata column used when
  `subset_highly_variable=true`
- `kwargs...`: optional loader options (currently supports `spatial_key`)

# Returns
- [`scData`](@ref)
"""
function load_scdata(
    path::String;
    format::Symbol = :h5ad,
    strict::Bool = true,
    subset_highly_variable::Bool = false,
    highly_variable_column::String = "highly_variable",
    kwargs...,
)
    format == :h5ad || throw(ArgumentError("Unsupported format: $format"))
    isfile(path) || throw(ArgumentError("scData file not found: $path"))

    ad = pyimport("anndata")
    handle = ad.read_h5ad(path)
    if subset_highly_variable
        handle = _subset_h5ad_highly_variable(handle; column = highly_variable_column)
    end

    spatial_key = String(get(kwargs, :spatial_key, "spatial"))
    schema = Dict{String,Any}(
        "spatial_key" => spatial_key,
        "subset_highly_variable" => subset_highly_variable,
        "highly_variable_column" => highly_variable_column,
    )
    _validate_h5ad_schema(handle; strict = strict, spatial_key = spatial_key)

    return scData(:h5ad, path, handle, schema)
end

function _subset_h5ad_highly_variable(handle; column::String)
    has_column = pyconvert(Bool, handle.var.__contains__(column))
    has_column ||
        throw(ArgumentError("AnnData `var` is missing highly variable column `$column`"))

    mask_py = handle.var.__getitem__(column).tolist()
    mask_values = pyconvert(Vector{Any}, mask_py)
    mask = Bool[Bool(v) for v in mask_values]
    length(mask) == pyconvert(Int, handle.n_vars) || throw(
        ArgumentError("AnnData highly variable mask length does not match `var` rows"),
    )

    subset = handle.copy()
    subset._inplace_subset_var(mask)
    return subset
end

function _validate_h5ad_schema(handle; strict::Bool = true, spatial_key::String = "spatial")
    pyhasattr(handle, "X") || throw(ArgumentError("AnnData object is missing `X`"))
    pyhasattr(handle, "obs") || throw(ArgumentError("AnnData object is missing `obs`"))
    pyhasattr(handle, "var") || throw(ArgumentError("AnnData object is missing `var`"))

    n_obs = pyconvert(Int, handle.n_obs)
    n_var = pyconvert(Int, handle.n_vars)
    x = _to_dense_matrix(handle.X; slot = "X")
    size(x, 1) == n_obs || throw(ArgumentError("AnnData `X` rows must match `obs` rows"))
    size(x, 2) == n_var || throw(ArgumentError("AnnData `X` columns must match `var` rows"))

    if strict
        pyhasattr(handle, "obsm") ||
            throw(ArgumentError("AnnData object is missing `obsm`"))
        has_spatial = pyconvert(Bool, handle.obsm.__contains__(spatial_key))
        has_spatial || throw(ArgumentError("AnnData `obsm` is missing key `$spatial_key`"))
        coords = _to_dense_matrix(
            handle.obsm.__getitem__(spatial_key);
            slot = "obsm[$spatial_key]",
        )
        size(coords, 1) == n_obs ||
            throw(ArgumentError("AnnData `obsm[$spatial_key]` rows must match `obs` rows"))
        size(coords, 2) >= 2 || throw(
            ArgumentError("AnnData `obsm[$spatial_key]` must have at least 2 columns"),
        )
        all(isfinite, coords) ||
            throw(ArgumentError("AnnData `obsm[$spatial_key]` contains non-finite values"))
    end

    return nothing
end

"""
    expression_matrix(scdata::scData; layer=nothing)

Return the expression matrix from `scdata`.

# Arguments
- `scdata::scData`: spatial data handle loaded with [`load_scdata`](@ref)
- `layer`: optional layer key; defaults to `nothing` (use `.X`)

# Returns
- `AbstractMatrix`
"""
function expression_matrix(scdata::scData; layer = nothing)
    handle = _get_h5ad_handle(scdata)
    if isnothing(layer)
        return _to_dense_matrix(handle.X; slot = "X")
    end

    return layer(scdata, String(layer))
end

"""
    spatial_coords(scdata::scData; key="spatial")

Return spatial coordinates matrix from `.obsm[key]`.

# Arguments
- `scdata::scData`: spatial data handle loaded with [`load_scdata`](@ref)
- `key::String`: key in AnnData `.obsm` containing coordinates

# Returns
- `AbstractMatrix`
"""
function spatial_coords(scdata::scData; key::String = "spatial")
    return obsm(scdata, key)
end

"""
    obsm(scdata::scData, key::String)

Return a matrix stored in `.obsm[key]`.

# Arguments
- `scdata::scData`: spatial data handle loaded with [`load_scdata`](@ref)
- `key::String`: key in AnnData `.obsm`

# Returns
- `AbstractMatrix`
"""
function obsm(scdata::scData, key::String)
    handle = _get_h5ad_handle(scdata)
    return _get_matrix_slot(handle, "obsm", key)
end

"""
    varm(scdata::scData, key::String)

Return a matrix stored in `.varm[key]`.

# Arguments
- `scdata::scData`: spatial data handle loaded with [`load_scdata`](@ref)
- `key::String`: key in AnnData `.varm`

# Returns
- `AbstractMatrix`
"""
function varm(scdata::scData, key::String)
    handle = _get_h5ad_handle(scdata)
    return _get_matrix_slot(handle, "varm", key)
end

"""
    obsp(scdata::scData, key::String)

Return a matrix stored in `.obsp[key]`.

# Arguments
- `scdata::scData`: spatial data handle loaded with [`load_scdata`](@ref)
- `key::String`: key in AnnData `.obsp`

# Returns
- `AbstractMatrix`
"""
function obsp(scdata::scData, key::String)
    handle = _get_h5ad_handle(scdata)
    return _get_matrix_slot(handle, "obsp", key)
end

"""
    varp(scdata::scData, key::String)

Return a matrix stored in `.varp[key]`.

# Arguments
- `scdata::scData`: spatial data handle loaded with [`load_scdata`](@ref)
- `key::String`: key in AnnData `.varp`

# Returns
- `AbstractMatrix`
"""
function varp(scdata::scData, key::String)
    handle = _get_h5ad_handle(scdata)
    return _get_matrix_slot(handle, "varp", key)
end

"""
    layers(scdata::scData)

Return available layer keys from `.layers`.

# Arguments
- `scdata::scData`: spatial data handle loaded with [`load_scdata`](@ref)

# Returns
- `Vector{String}`
"""
function layers(scdata::scData)
    handle = _get_h5ad_handle(scdata)
    return _mapping_keys(handle.layers)
end

"""
    layer(scdata::scData, key::String)

Return a matrix stored in `.layers[key]`.

# Arguments
- `scdata::scData`: spatial data handle loaded with [`load_scdata`](@ref)
- `key::String`: key in AnnData `.layers`

# Returns
- `AbstractMatrix`
"""
function layer(scdata::scData, key::String)
    handle = _get_h5ad_handle(scdata)
    return _get_matrix_slot(handle, "layers", key)
end

"""
    raw_expression_matrix(scdata::scData)

Return raw expression matrix from `.raw.X`.

# Arguments
- `scdata::scData`: spatial data handle loaded with [`load_scdata`](@ref)

# Returns
- `AbstractMatrix`
"""
function raw_expression_matrix(scdata::scData)
    handle = _get_h5ad_handle(scdata)
    raw = handle.raw
    _py_is_none(raw) && throw(ArgumentError("AnnData `raw` is missing"))
    return _to_dense_matrix(raw.X; slot = "raw.X")
end

function _resolve_cell_annotation_column(obs::DataFrame, column)
    obs_names = Symbol.(names(obs))

    if column == :auto
        preferred = [:Annotation, :annotation, :Cluster, :cluster, :cell_type, :celltype]
        for candidate in preferred
            if candidate in obs_names
                return candidate
            end
        end

        fallback = filter(name -> name != :index, obs_names)
        isempty(fallback) &&
            throw(ArgumentError("Could not infer a cell annotation column from `obs`"))
        return first(fallback)
    end

    resolved = Symbol(column)
    resolved in obs_names ||
        throw(ArgumentError("AnnData `obs` is missing annotation column `$resolved`"))
    return resolved
end

"""
    cell_annotation(scdata::scData; column=:auto)

Return cell-annotation labels from `scdata.obs`.

# Arguments
- `scdata::scData`: spatial data handle loaded with [`load_scdata`](@ref)
- `column`: annotation column name or `:auto` to use a preferred default

# Returns
- `Vector{String}`
"""
function cell_annotation(scdata::scData; column = :auto)
    obs = obs_table(scdata)
    annotation_col = _resolve_cell_annotation_column(obs, column)
    values = obs[!, annotation_col]
    any(ismissing, values) &&
        throw(ArgumentError("Annotation column `$annotation_col` contains missing values"))
    return String.(values)
end

"""
    obs_table(scdata::scData)

Return observation metadata table from `.obs`.

# Arguments
- `scdata::scData`: spatial data handle loaded with [`load_scdata`](@ref)

# Returns
- `DataFrame`
"""
function obs_table(scdata::scData)
    handle = _get_h5ad_handle(scdata)
    return _pandas_to_dataframe(handle.obs)
end

"""
    var_table(scdata::scData)

Return feature metadata table from `.var`.

# Arguments
- `scdata::scData`: spatial data handle loaded with [`load_scdata`](@ref)

# Returns
- `DataFrame`
"""
function var_table(scdata::scData)
    handle = _get_h5ad_handle(scdata)
    return _pandas_to_dataframe(handle.var)
end

"""
    uns(scdata::scData)

Return unstructured metadata from `.uns` as a Julia `Dict`.

# Arguments
- `scdata::scData`: spatial data handle loaded with [`load_scdata`](@ref)

# Returns
- `Dict{String,Any}`
"""
function uns(scdata::scData)
    handle = _get_h5ad_handle(scdata)
    return _to_mapping_dict(handle.uns)
end

"""
    uns(scdata::scData, key::String)

Return a single value from `.uns[key]`.

# Arguments
- `scdata::scData`: spatial data handle loaded with [`load_scdata`](@ref)
- `key::String`: key in AnnData `.uns`

# Returns
- value stored at `key`
"""
function uns(scdata::scData, key::String)
    handle = _get_h5ad_handle(scdata)
    has_key = pyconvert(Bool, handle.uns.__contains__(key))
    has_key || throw(ArgumentError("AnnData `uns` is missing key `$key`"))
    value_py = handle.uns.__getitem__(key)
    return try
        pyconvert(Any, value_py)
    catch
        value_py
    end
end
