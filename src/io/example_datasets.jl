using SHA
using Tar
using CodecZlib

function _default_example_manifest_path()
    return normpath(joinpath(@__DIR__, "..", "..", "data", "examples", "manifest.json"))
end

function _manifest_entry(manifest::AbstractDict{String,Any}, name::String)
    haskey(manifest, name) || throw(ArgumentError("Unknown example dataset `$name`"))
    entry = manifest[name]
    entry isa AbstractDict ||
        throw(ArgumentError("Invalid manifest entry for dataset `$name`"))
    return entry
end

function _dataset_version(entry::AbstractDict{String,Any}, version)
    if isnothing(version)
        haskey(entry, "version") ||
            throw(ArgumentError("Manifest entry is missing required `version`"))
        return String(entry["version"])
    end
    return String(version)
end

function _sha256_file(path::String)
    open(path, "r") do io
        return bytes2hex(sha256(io))
    end
end

function _verify_sha256!(path::String, expected_hex::String)
    actual = lowercase(_sha256_file(path))
    expected = lowercase(strip(expected_hex))
    actual == expected || throw(
        ArgumentError("SHA-256 mismatch for `$path`: expected `$expected`, got `$actual`"),
    )
    return nothing
end

function _extract_tar_gz!(archive_path::String, out_dir::String)
    mkpath(out_dir)
    open(archive_path, "r") do io
        gz = GzipDecompressorStream(io)
        try
            Tar.extract(gz, out_dir)
        finally
            close(gz)
        end
    end
    return out_dir
end

"""
    load_example_dataset_manifest(; manifest_path=nothing)

Load the example dataset manifest JSON.

# Arguments
- `manifest_path`: optional path to a manifest JSON file. Defaults to `data/examples/manifest.json`.

# Returns
- `Dict{String,Any}` with dataset metadata entries.
"""
function load_example_dataset_manifest(; manifest_path = nothing)
    path =
        isnothing(manifest_path) ? _default_example_manifest_path() : String(manifest_path)
    isfile(path) || throw(ArgumentError("Example dataset manifest not found: $path"))
    manifest = JSON.parsefile(path)
    manifest isa AbstractDict{String,Any} ||
        throw(ArgumentError("Example dataset manifest must be a JSON object"))
    return manifest
end

"""
    example_dataset_path(name; version=nothing, dest="data/examples", manifest_path=nothing)

Return the local cache path for an example dataset.

# Arguments
- `name::String`: dataset name in the manifest (e.g. `"visium"`, `"xenium"`).
- `version`: optional version override. Defaults to the manifest version.
- `dest::String`: base cache directory.
- `manifest_path`: optional manifest path override.

# Returns
- `String` absolute path to the dataset cache directory.
"""
function example_dataset_path(
    name::String;
    version = nothing,
    dest::String = "data/examples",
    manifest_path = nothing,
)
    manifest = load_example_dataset_manifest(; manifest_path = manifest_path)
    entry = _manifest_entry(manifest, name)
    ver = _dataset_version(entry, version)
    return normpath(joinpath(abspath(dest), name, ver))
end

"""
    resolve_example_dataset(name; data_dir=nothing)

Resolve the local path for an example dataset.

Checks in order:
1. `data_dir` keyword argument — if provided, returns it directly
2. `SCCCHAIN_DATA_DIR` environment variable — looks for `<name>/` subdirectory

# Arguments
- `name::String`: dataset name (e.g. `"visium"`, `"xenium"`).
- `data_dir::Union{String,Nothing}`: explicit local directory containing the dataset files.

# Returns
- `String` absolute path to the dataset directory.
"""
function resolve_example_dataset(
    name::String;
    data_dir::Union{String,Nothing} = nothing,
)
    # 1. Explicit data_dir
    if !isnothing(data_dir)
        isdir(data_dir) || throw(ArgumentError("data_dir does not exist: $data_dir"))
        return abspath(data_dir)
    end

    # 2. Environment variable
    env_dir = get(ENV, "SCCCHAIN_DATA_DIR", nothing)
    if !isnothing(env_dir)
        candidate = joinpath(env_dir, name)
        if isdir(candidate)
            return abspath(candidate)
        end
        # If env var points directly to the dataset (no subdirectory)
        if isdir(env_dir)
            return abspath(env_dir)
        end
        throw(
            ArgumentError(
                "SCCCHAIN_DATA_DIR is set to '$env_dir' but neither '$candidate' nor '$env_dir' is a valid directory",
            ),
        )
    end

    throw(
        ArgumentError(
            "No data directory found for dataset '$name'. Provide `data_dir` or set the `SCCCHAIN_DATA_DIR` environment variable.",
        ),
    )
end
