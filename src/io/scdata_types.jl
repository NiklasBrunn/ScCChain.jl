"""
    scData

Container for spatial data loaded from external backends (currently AnnData `.h5ad`).

# Fields
- `backend::Symbol`: backend identifier (e.g. `:h5ad`)
- `path::String`: source file path
- `handle::Any`: backend-native handle object
- `schema::Dict{String,Any}`: resolved schema keys and metadata
"""
struct scData
    backend::Symbol
    path::String
    handle::Any
    schema::Dict{String,Any}
end
