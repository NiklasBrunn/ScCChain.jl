using PythonCall

function _load_preprocessing_backend()
    importlib_util = pyimport("importlib.util")
    backend_path = joinpath(@__DIR__, "python", "preprocess_10x.py")
    isfile(backend_path) ||
        throw(ArgumentError("Preprocessing backend not found: $backend_path"))

    spec = importlib_util.spec_from_file_location("scinchain_preprocess_10x", backend_path)
    _py_is_none(spec) &&
        throw(ArgumentError("Could not load Python preprocessing module spec"))
    module_py = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module_py)
    return module_py
end

function _default_preprocess_outdir(indir::AbstractString, modality::Symbol)
    suffix = modality == :visium ? "scanpy_out_visium" : "scanpy_out_xenium"
    return joinpath(indir, suffix)
end

"""
    preprocess_scdata(indir; modality, outdir=nothing, kwargs...)

Preprocess a 10x dataset directory and return a Julia-loadable spatial data handle.

# Arguments
- `indir::AbstractString`: path to the input dataset directory
- `modality::Symbol`: dataset modality (`:visium` or `:xenium`)
- `outdir`: optional output directory
- `normalize_and_log1p::Bool`: normalize to 10k counts and log1p-transform (default `true`)
- `min_genes_per_cell::Int`: minimum detected genes to keep a cell (default `1`)
- `min_cells_per_gene::Int`: minimum cells expressing a gene to keep it (default `1`)
- `max_pct_mt`: maximum mitochondrial percentage for cell filtering (default `nothing`)
- `max_counts_per_cell`: maximum total counts for cell filtering (default `nothing`)
- `run_hvg::Bool`: run highly-variable gene selection (default `false`)
- `hvg_n_top_genes::Int`: number of top HVGs to select (default `2000`)
- `hvg_layer`: layer to use for HVG computation (default `nothing`)
- `hvg_flavor::String`: scanpy HVG flavor (default `"seurat"`)
- `hvg_subset::Bool`: subset the output to HVGs only (default `false`)
- `excel_annotation_path`: path to an Excel file with per-barcode cell type annotations
  (Visium only, default `nothing`). When provided, `Annotation` and `Cluster` columns
  are merged into `obs` before filtering.
- `excel_sheet::String`: sheet name in the Excel file (default `"Visium"`)
- `excel_barcode_col::String`: barcode column name (default `"Barcode"`)
- `excel_annotation_col::String`: annotation column name (default `"Annotation"`)
- `excel_cluster_col::String`: cluster column name (default `"Cluster"`)

When `run_hvg=true`, preprocessing can either annotate `var.highly_variable` while
keeping all genes (`hvg_subset=false`) or write an HVG-only output (`hvg_subset=true`).
If `hvg_layer` is requested, that layer is materialized from the current counts matrix
before any optional normalization/log1p transform so layer-based Scanpy HVG flavors can
still read pre-transform counts.

# Returns
- `Tuple{scData,String}` with loaded handle and written `.h5ad` path
"""
function preprocess_scdata(
    indir::AbstractString;
    modality::Symbol,
    outdir = nothing,
    kwargs...,
)
    isdir(indir) || throw(ArgumentError("Input dataset directory not found: $indir"))
    modality in (:visium, :xenium) ||
        throw(ArgumentError("Unsupported modality: $modality. Expected :visium or :xenium"))

    backend = _load_preprocessing_backend()
    outdir_str =
        isnothing(outdir) ? _default_preprocess_outdir(indir, modality) : String(outdir)
    normalize_and_log1p = Bool(get(kwargs, :normalize_and_log1p, true))
    min_genes_per_cell = Int(get(kwargs, :min_genes_per_cell, 1))
    min_cells_per_gene = Int(get(kwargs, :min_cells_per_gene, 1))
    max_pct_mt = get(kwargs, :max_pct_mt, nothing)
    max_counts_per_cell = get(kwargs, :max_counts_per_cell, nothing)
    run_hvg = Bool(get(kwargs, :run_hvg, false))
    hvg_n_top_genes = Int(get(kwargs, :hvg_n_top_genes, 2000))
    hvg_layer = get(kwargs, :hvg_layer, nothing)
    hvg_flavor = String(get(kwargs, :hvg_flavor, "seurat"))
    hvg_subset = Bool(get(kwargs, :hvg_subset, false))
    excel_annotation_path = get(kwargs, :excel_annotation_path, nothing)
    excel_sheet = String(get(kwargs, :excel_sheet, "Visium"))
    excel_barcode_col = String(get(kwargs, :excel_barcode_col, "Barcode"))
    excel_annotation_col = String(get(kwargs, :excel_annotation_col, "Annotation"))
    excel_cluster_col = String(get(kwargs, :excel_cluster_col, "Cluster"))

    if run_hvg && hvg_n_top_genes <= 0
        throw(ArgumentError("`hvg_n_top_genes` must be positive when `run_hvg=true`"))
    end

    backend_kwargs = (
        normalize_and_log1p = normalize_and_log1p,
        min_genes_per_cell = min_genes_per_cell,
        min_cells_per_gene = min_cells_per_gene,
        max_pct_mt = max_pct_mt,
        max_counts_per_cell = max_counts_per_cell,
    )
    if run_hvg
        backend_kwargs = (
            backend_kwargs...,
            run_hvg = run_hvg,
            hvg_n_top_genes = hvg_n_top_genes,
            hvg_layer = isnothing(hvg_layer) ? nothing : String(hvg_layer),
            hvg_flavor = hvg_flavor,
            hvg_subset = hvg_subset,
        )
    end
    if !isnothing(excel_annotation_path)
        backend_kwargs = (
            backend_kwargs...,
            excel_annotation_path = String(excel_annotation_path),
            excel_sheet = excel_sheet,
            excel_barcode_col = excel_barcode_col,
            excel_annotation_col = excel_annotation_col,
            excel_cluster_col = excel_cluster_col,
        )
    end

    try
        outpath_py = backend.preprocess_10x(
            String(indir),
            outdir_str,
            String(modality);
            backend_kwargs...,
        )
        outpath = pyconvert(String, outpath_py)
        scdata = load_scdata(outpath; format = :h5ad, strict = true)
        return scdata, outpath
    catch err
        throw(
            ArgumentError(
                "Preprocessing failed for modality `$modality`: $(sprint(showerror, err))",
            ),
        )
    end
end
