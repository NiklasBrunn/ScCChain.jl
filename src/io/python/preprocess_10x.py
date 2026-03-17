from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

# Disable Arrow-backed string inference (pandas ≥ 2.1 / default in 3.0)
# to ensure all string columns use plain object arrays that h5py can serialize.
try:
    pd.options.future.infer_string = False
except (AttributeError, KeyError):
    pass

ad.settings.allow_write_nullable_strings = True


def _mt_mask(var_names: pd.Index) -> np.ndarray:
    """Return a boolean mask selecting mitochondrial genes (MT- or MT_ prefix)."""
    names = np.asarray(var_names).astype(str)
    up = np.char.upper(names)
    return np.char.startswith(up, "MT-") | np.char.startswith(up, "MT_")


def _qc_obs_metrics(adata: ad.AnnData) -> None:
    """Compute total_counts, n_genes_by_counts, and pct_counts_mt in adata.obs."""
    mt_mask = _mt_mask(adata.var_names)
    x = adata.X
    if issparse(x):
        total_counts = np.ravel(x.sum(axis=1))
        n_genes = np.ravel((x > 0).sum(axis=1))
        mt_counts = np.ravel(x[:, mt_mask].sum(axis=1))
    else:
        total_counts = x.sum(axis=1)
        n_genes = (x > 0).sum(axis=1)
        mt_counts = x[:, mt_mask].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_mt = np.where(total_counts > 0, (mt_counts / total_counts) * 100.0, 0.0)
    adata.obs["total_counts"] = total_counts
    adata.obs["n_genes_by_counts"] = n_genes
    adata.obs["pct_counts_mt"] = pct_mt


def _copy_matrix(x):
    """Deep-copy a matrix, handling both sparse and dense formats."""
    return x.copy() if hasattr(x, "copy") else np.array(x, copy=True)


def _materialize_hvg_layer(adata: ad.AnnData, hvg_layer: Optional[str]) -> Optional[str]:
    """Ensure the requested layer exists in adata, copying from X if needed.

    Returns the layer name or None if hvg_layer is None.
    """
    if hvg_layer is None:
        return None

    layer_name = str(hvg_layer)
    if not layer_name:
        raise ValueError("`hvg_layer` must be a non-empty string when provided.")
    if layer_name not in adata.layers:
        try:
            adata.layers[layer_name] = _copy_matrix(adata.X)
        except Exception as exc:
            raise ValueError(f"Could not materialize requested HVG layer '{layer_name}'.") from exc
    return layer_name


def _apply_hvg_selection(
    adata: ad.AnnData,
    *,
    run_hvg: bool = False,
    hvg_n_top_genes: int = 2000,
    hvg_layer: Optional[str] = None,
    hvg_flavor: str = "seurat",
    hvg_subset: bool = False,
) -> ad.AnnData:
    """Run highly-variable gene selection via scanpy and optionally subset.

    Annotates var['highly_variable']. If hvg_subset is True, the returned
    AnnData only contains the selected genes.
    """
    if not run_hvg:
        return adata

    if int(hvg_n_top_genes) <= 0:
        raise ValueError("`hvg_n_top_genes` must be positive when `run_hvg=True`.")

    effective_flavor = str(hvg_flavor)
    # `seurat_v3` can crash inside skmisc loess on tiny synthetic fixtures.
    if effective_flavor == "seurat_v3" and adata.n_obs < 10:
        effective_flavor = "seurat"

    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=int(hvg_n_top_genes),
            flavor=effective_flavor,
            layer=hvg_layer,
            subset=False,
        )
    except Exception as exc:
        raise ValueError(f"HVG selection failed: {exc}") from exc

    if "highly_variable" not in adata.var.columns:
        raise ValueError("HVG selection did not annotate `var['highly_variable']`.")

    hvg_mask = np.asarray(adata.var["highly_variable"]).astype(bool)
    target_genes = min(int(hvg_n_top_genes), adata.n_vars)
    if int(hvg_mask.sum()) == 0:
        raise ValueError("HVG selection did not retain any genes.")
    if int(hvg_mask.sum()) != target_genes:
        score_cols = ("variances_norm", "dispersions_norm", "variances", "dispersions")
        score_col = next((col for col in score_cols if col in adata.var.columns), None)
        if score_col is None:
            raise ValueError("HVG selection did not produce a ranking column for top-gene selection.")
        scores = np.asarray(adata.var[score_col], dtype=float)
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        top_idx = np.argsort(scores)[::-1][:target_genes]
        hvg_mask = np.zeros(adata.n_vars, dtype=bool)
        hvg_mask[top_idx] = True
        adata.var["highly_variable"] = hvg_mask

    if hvg_subset:
        adata = adata[:, hvg_mask].copy()
        adata.var["highly_variable"] = True

    return adata


def _load_visium_counts(indir: Path) -> ad.AnnData:
    """Load Visium count matrix from H5 or MTX directory, filtering to gene expression."""
    adata: Optional[ad.AnnData] = None
    h5_candidates = [
        "filtered_feature_bc_matrix.h5",
        "CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5",
        "raw_feature_bc_matrix.h5",
        "CytAssist_FFPE_Human_Breast_Cancer_raw_feature_bc_matrix.h5",
    ]
    for count_file in h5_candidates:
        path = indir / count_file
        if not path.exists():
            continue
        try:
            adata = sc.read_visium(indir, count_file=count_file)
            break
        except Exception:
            continue

    if adata is None:
        mtx_dirs = [
            "filtered_feature_bc_matrix",
            "raw_feature_bc_matrix",
            "CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix",
            "CytAssist_FFPE_Human_Breast_Cancer_raw_feature_bc_matrix",
        ]
        for dirname in mtx_dirs:
            mtx_dir = indir / dirname
            if (mtx_dir / "matrix.mtx.gz").exists() or (mtx_dir / "matrix.mtx").exists():
                adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", make_unique=True)
                break

    if adata is None:
        raise FileNotFoundError(
            "Could not load Visium counts. Expected read_visium H5 or 10x MTX directories."
        )

    adata.var_names_make_unique()
    if "feature_types" in adata.var.columns:
        mask_ge = adata.var["feature_types"].astype(str).str.lower().eq("gene expression")
        if mask_ge.any():
            adata = adata[:, mask_ge].copy()
    return adata


def _attach_visium_spatial(adata: ad.AnnData, indir: Path) -> ad.AnnData:
    """Attach spatial coordinates to Visium AnnData from obsm or tissue_positions file."""
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=float)
        if coords.shape[0] != adata.n_obs:
            raise ValueError("AnnData obsm['spatial'] row count does not match observations")
        adata.obsm["spatial"] = coords[:, :2]
        return adata

    spatial_dir = indir / "spatial"
    tpos = None
    for candidate in ("tissue_positions.csv", "tissue_positions_list.csv", "tissue_positions.tsv"):
        path = spatial_dir / candidate
        if not path.exists():
            continue
        if candidate.endswith(".tsv"):
            tpos = pd.read_csv(path, sep="\t", header=None if "list" in candidate else 0)
        else:
            tpos = pd.read_csv(path, header=None if "list" in candidate else 0)
        break

    if tpos is None:
        raise FileNotFoundError(
            "No spatial coordinates found. Expected obsm['spatial'] or spatial/tissue_positions*.csv|tsv."
        )

    lower_cols = [str(c).lower() for c in tpos.columns]
    if "barcode" in lower_cols:
        lc = {str(c).lower(): c for c in tpos.columns}
        bcol = lc.get("barcode")
        xcol = lc.get("pxl_col_in_fullres")
        ycol = lc.get("pxl_row_in_fullres")
        if bcol is None or xcol is None or ycol is None:
            raise ValueError("tissue_positions.csv missing barcode/pxl_col_in_fullres/pxl_row_in_fullres")
        pos = tpos[[bcol, xcol, ycol]].rename(columns={bcol: "barcode", xcol: "x", ycol: "y"})
    else:
        if tpos.shape[1] < 6:
            raise ValueError("tissue_positions_list.csv must contain six columns")
        tpos = tpos.iloc[:, :6].copy()
        tpos.columns = ["barcode", "in_tissue", "array_row", "array_col", "x", "y"]
        pos = tpos[["barcode", "x", "y"]]

    pos["barcode"] = pos["barcode"].astype(str)
    aligned = pos.set_index("barcode").reindex(adata.obs_names.astype(str))
    coords = aligned[["x", "y"]].to_numpy(dtype=float)
    if np.isnan(coords).any():
        raise ValueError("Missing spatial coordinates after aligning tissue_positions to expression barcodes")

    adata.obsm["spatial"] = coords
    return adata


def _get_visium_scalefactors(adata: ad.AnnData, indir: Path) -> dict:
    """Read Visium scale factors from adata.uns (sc.read_visium) or scalefactors_json.json."""
    sf: dict = {}
    try:
        if "spatial" in adata.uns and isinstance(adata.uns["spatial"], dict):
            lib_id = next(iter(adata.uns["spatial"]))
            raw = adata.uns["spatial"][lib_id].get("scalefactors", {}) or {}
            sf = {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    except (StopIteration, TypeError, ValueError):
        pass

    if not sf:
        sf_path = indir / "spatial" / "scalefactors_json.json"
        if sf_path.exists():
            try:
                with open(sf_path) as fh:
                    raw = json.load(fh)
                sf = {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}
            except Exception:
                pass
    return sf


def _convert_visium_px_to_um(
    adata: ad.AnnData, indir: Path, *, spot_diameter_um: float = 55.0
) -> ad.AnnData:
    """Convert Visium pixel coordinates in obsm['spatial'] to micrometers.

    Uses scale factors from adata.uns (populated by sc.read_visium) or from
    the spatial/scalefactors_json.json file. Auto-detects whether coordinates
    are in hi-res or full-resolution pixel space via a nearest-neighbor
    heuristic (Visium NN spacing ≈ 100 µm).
    """
    if "spatial" not in adata.obsm:
        return adata

    coords_px = np.asarray(adata.obsm["spatial"], dtype=float)
    sf = _get_visium_scalefactors(adata, indir)

    # Compute full-resolution µm/px
    mpp = sf.get("microns_per_pixel")
    sdf = sf.get("spot_diameter_fullres")
    if mpp is not None and np.isfinite(mpp):
        um_px_full = float(mpp)
    elif sdf is not None and np.isfinite(sdf) and sdf > 0:
        um_px_full = spot_diameter_um / float(sdf)
    else:
        return adata  # cannot convert without scale factors

    # Compute hi-res µm/px candidate
    ths = sf.get("tissue_hires_scalef")
    um_px_hi = None
    if ths is not None and np.isfinite(ths) and float(ths) > 0:
        um_px_hi = um_px_full / float(ths)

    # Pick the right conversion using median NN distance heuristic:
    # Visium spots have ~100 µm center-to-center spacing.
    if um_px_hi is not None and coords_px.shape[0] >= 3:
        from scipy.spatial import cKDTree

        dists = cKDTree(coords_px).query(coords_px, k=2)[0]
        nn_px = float(np.nanmedian(dists[:, 1]))
        if nn_px > 0:
            err_full = abs(nn_px * um_px_full - 100.0)
            err_hi = abs(nn_px * um_px_hi - 100.0)
            um_per_px = um_px_full if err_full < err_hi else um_px_hi
        else:
            um_per_px = um_px_full
    else:
        um_per_px = um_px_full

    adata.obsm["spatial"] = coords_px * um_per_px
    return adata


def _to_minimal_scdata(adata: ad.AnnData) -> ad.AnnData:
    """Strip AnnData to minimal scCChain-compatible form: X, obs, var, and obsm['spatial']."""
    if "spatial" not in adata.obsm:
        raise ValueError("Expected obsm['spatial'] in preprocessed AnnData")
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    if coords.ndim != 2 or coords.shape[0] != adata.n_obs or coords.shape[1] < 2:
        raise ValueError("Invalid spatial coordinates shape; expected n_obs x 2+")

    out = ad.AnnData(
        X=adata.X.copy() if hasattr(adata.X, "copy") else adata.X,
        obs=_sanitize_dataframe_for_h5ad(adata.obs),
        var=_sanitize_dataframe_for_h5ad(adata.var),
        obsm={"spatial": coords[:, :2]},
    )
    out.var_names_make_unique()
    return out


def _sanitize_dataframe_for_h5ad(df: pd.DataFrame) -> pd.DataFrame:
    """Convert non-numeric columns to plain object dtype for h5ad compatibility.

    Newer pandas versions back string columns with PyArrow (ArrowStringArray),
    which h5py cannot serialize. This function coerces every non-numeric column
    to numpy object arrays that anndata/h5py can write.
    """
    index_obj = pd.Index(np.asarray(df.index.astype(str), dtype=object), dtype=object)
    out = pd.DataFrame(index=index_obj)
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and not isinstance(
            series.dtype, pd.CategoricalDtype
        ):
            out[col] = np.asarray(series)
        else:
            out[col] = pd.Series(
                np.asarray(series.astype(str), dtype=object),
                index=out.index,
                dtype=object,
            )
    return out


def _merge_excel_annotations(
    adata: ad.AnnData,
    excel_path: Path,
    *,
    sheet_name: str = "Visium",
    barcode_col: str = "Barcode",
    annotation_col: str = "Annotation",
    cluster_col: str = "Cluster",
) -> ad.AnnData:
    """Merge per-barcode annotation labels from an Excel file into adata.obs.

    Handles numeric barcode IDs with potential 0/1-based offset mismatch
    (common in Xenium datasets where the Excel uses 0-based IDs but the
    expression matrix uses 1-based IDs).
    """
    try:
        import openpyxl  # noqa: F401 — verify reader is available
    except ImportError:
        raise ImportError(
            "openpyxl is required to read Excel annotation files. "
            "Install it with: pip install openpyxl"
        )

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    cols_lower = {c.lower(): c for c in df.columns}
    bcol = cols_lower.get(barcode_col.lower())
    acol = cols_lower.get(annotation_col.lower())
    ccol = cols_lower.get(cluster_col.lower())

    if bcol is None:
        raise ValueError(
            f"Excel sheet '{sheet_name}' missing barcode column '{barcode_col}'. "
            f"Available columns: {list(df.columns)}"
        )
    if acol is None and ccol is None:
        raise ValueError(
            f"Excel sheet '{sheet_name}' missing both '{annotation_col}' and "
            f"'{cluster_col}' columns."
        )

    keep_cols = [bcol] + [c for c in [ccol, acol] if c is not None]
    df = df[keep_cols].copy()

    # Robust string conversion for numeric barcode columns (e.g. 12.0 → "12")
    s = df[bcol]
    if pd.api.types.is_numeric_dtype(s):
        ids = pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)
    else:
        ids = s.astype(str).str.replace(r"\.0$", "", regex=True)
    df[bcol] = ids
    df = df.dropna(subset=[bcol]).drop_duplicates(subset=[bcol]).set_index(bcol)

    # Try direct join; if zero matches and IDs look numeric, try +1 offset
    adata_ids = adata.obs_names.astype(str)
    matched = adata_ids.isin(df.index).sum()
    if (
        matched == 0
        and adata_ids.str.isnumeric().all()
        and pd.Index(df.index).str.isnumeric().all()
    ):
        plus = (pd.Index(df.index).astype(int) + 1).astype(str)
        df = df.copy()
        df.index = plus

    adata.obs = adata.obs.join(df, how="left")

    for col in [ccol, acol]:
        if col is not None and col in adata.obs.columns:
            adata.obs[col] = adata.obs[col].astype("category")

    return adata


def _preprocess_visium(
    indir: Path,
    outdir: Path,
    *,
    normalize_and_log1p: bool = True,
    min_genes_per_cell: int = 1,
    min_cells_per_gene: int = 1,
    max_pct_mt: Optional[float] = None,
    max_counts_per_cell: Optional[int] = None,
    run_hvg: bool = False,
    hvg_n_top_genes: int = 2000,
    hvg_layer: Optional[str] = None,
    hvg_flavor: str = "seurat",
    hvg_subset: bool = False,
    excel_annotation_path: Optional[str] = None,
    excel_sheet: str = "Visium",
    excel_barcode_col: str = "Barcode",
    excel_annotation_col: str = "Annotation",
    excel_cluster_col: str = "Cluster",
    spot_diameter_um: float = 55.0,
) -> str:
    """Preprocess a raw 10x Visium directory into a minimal h5ad file.

    Loads counts, attaches spatial coordinates (converted to micrometers),
    filters cells/genes, optionally merges per-barcode annotation labels from
    an Excel file, normalizes, runs HVG selection, and writes
    visium_preprocessed.h5ad to outdir. Returns the output file path.
    """
    adata = _load_visium_counts(indir)
    adata = _attach_visium_spatial(adata, indir)
    adata = _convert_visium_px_to_um(adata, indir, spot_diameter_um=spot_diameter_um)

    if excel_annotation_path is not None:
        xlsx = Path(excel_annotation_path)
        if not xlsx.exists():
            raise FileNotFoundError(
                f"Excel annotation file not found: {xlsx}"
            )
        adata = _merge_excel_annotations(
            adata,
            xlsx,
            sheet_name=excel_sheet,
            barcode_col=excel_barcode_col,
            annotation_col=excel_annotation_col,
            cluster_col=excel_cluster_col,
        )

    sc.pp.filter_genes(adata, min_cells=max(0, int(min_cells_per_gene)))
    sc.pp.filter_cells(adata, min_genes=max(0, int(min_genes_per_cell)))

    if max_pct_mt is not None:
        _qc_obs_metrics(adata)
        adata = adata[adata.obs["pct_counts_mt"] <= float(max_pct_mt)].copy()
    if max_counts_per_cell is not None:
        _qc_obs_metrics(adata)
        adata = adata[adata.obs["total_counts"] <= int(max_counts_per_cell)].copy()

    # For seurat_v3, save pre-normalization counts so HVG uses raw count data
    if run_hvg and hvg_layer is None and str(hvg_flavor) == "seurat_v3":
        adata.layers["counts"] = _copy_matrix(adata.X)
        hvg_layer = "counts"

    hvg_layer_name = _materialize_hvg_layer(adata, hvg_layer) if run_hvg else None
    if normalize_and_log1p:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    adata = _apply_hvg_selection(
        adata,
        run_hvg=run_hvg,
        hvg_n_top_genes=hvg_n_top_genes,
        hvg_layer=hvg_layer_name,
        hvg_flavor=hvg_flavor,
        hvg_subset=hvg_subset,
    )

    minimal = _to_minimal_scdata(adata)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "visium_preprocessed.h5ad"
    minimal.write_h5ad(outpath)
    return str(outpath)


def _load_xenium_counts(indir: Path) -> ad.AnnData:
    """Load Xenium count matrix from H5 or MTX directory, filtering to gene expression."""
    h5_path = indir / "cell_feature_matrix.h5"
    mtx_dir = indir / "cell_feature_matrix"
    if h5_path.exists():
        adata = sc.read_10x_h5(h5_path)
    elif (mtx_dir / "matrix.mtx.gz").exists() or (mtx_dir / "matrix.mtx").exists():
        adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", make_unique=True)
    else:
        raise FileNotFoundError(
            "Could not find Xenium counts. Expected cell_feature_matrix.h5 or cell_feature_matrix/matrix.mtx[.gz]."
        )

    adata.var_names_make_unique()
    if "feature_types" in adata.var.columns:
        mask_ge = adata.var["feature_types"].astype(str).str.lower().eq("gene expression")
        if mask_ge.any():
            adata = adata[:, mask_ge].copy()
    return adata


def _load_xenium_cells_metadata(indir: Path) -> pd.DataFrame:
    """Load Xenium cell metadata from parquet or CSV, using the best available ID column as index."""
    cells_df = None
    for fname in ("cells.parquet", "cells.csv.gz", "cells.csv"):
        fpath = indir / fname
        if not fpath.exists():
            continue
        if fname.endswith(".parquet"):
            cells_df = pd.read_parquet(fpath)
        else:
            cells_df = pd.read_csv(fpath)
        break

    if cells_df is None:
        raise FileNotFoundError("Could not find cells metadata: cells.parquet or cells.csv[.gz].")

    id_cols_priority = ["cell_id", "barcode", "CellID", "cellId", "cell", "id"]
    use_col = next((c for c in id_cols_priority if c in cells_df.columns), None)
    if use_col is not None:
        cells_df = cells_df.set_index(use_col)
    cells_df.index = cells_df.index.astype(str)
    if cells_df.index.has_duplicates:
        cells_df = cells_df[~cells_df.index.duplicated(keep="first")].copy()
    return cells_df


def _attach_xenium_spatial(adata: ad.AnnData, cells_df: pd.DataFrame) -> ad.AnnData:
    """Attach spatial coordinates to Xenium AnnData from cells metadata centroid columns."""
    adata.obs_names = adata.obs_names.astype(str)
    common = adata.obs_names.intersection(cells_df.index)

    if len(common) == 0 and adata.obs_names.str.isnumeric().all() and pd.Index(cells_df.index).str.isnumeric().all():
        idx_plus = (pd.Index(cells_df.index).astype(int) + 1).astype(str)
        common_plus = adata.obs_names.intersection(idx_plus)
        if len(common_plus) > 0:
            cells_df = cells_df.copy()
            cells_df.index = idx_plus
            common = common_plus

    if len(common) == 0:
        raise ValueError(
            "No overlap between expression barcodes and cells metadata IDs. "
            f"Expression example: {list(adata.obs_names[:5])}; "
            f"Metadata example: {list(cells_df.index[:5])}."
        )

    adata = adata[common].copy()
    adata.obs = adata.obs.join(cells_df.reindex(common), how="left")

    x_cols = [c for c in adata.obs.columns if str(c).lower().startswith("x_centroid")]
    y_cols = [c for c in adata.obs.columns if str(c).lower().startswith("y_centroid")]
    if not x_cols or not y_cols:
        x_cols = [
            c
            for c in adata.obs.columns
            if str(c).lower() in {"x", "x_um", "x_micron", "x_global_px", "x_px", "x_pixel"}
        ]
        y_cols = [
            c
            for c in adata.obs.columns
            if str(c).lower() in {"y", "y_um", "y_micron", "y_global_px", "y_px", "y_pixel"}
        ]

    if not x_cols or not y_cols:
        raise ValueError("Could not infer Xenium spatial columns from cells metadata.")

    colx, coly = x_cols[0], y_cols[0]
    coords = np.c_[adata.obs[colx].astype(float).to_numpy(), adata.obs[coly].astype(float).to_numpy()]
    if np.isnan(coords).any() or not np.isfinite(coords).all():
        raise ValueError("Xenium spatial coordinates contain non-finite values.")
    adata.obsm["spatial"] = coords
    return adata


def _preprocess_xenium(
    indir: Path,
    outdir: Path,
    *,
    normalize_and_log1p: bool = True,
    min_genes_per_cell: int = 1,
    min_cells_per_gene: int = 1,
    max_pct_mt: Optional[float] = None,
    max_counts_per_cell: Optional[int] = None,
    run_hvg: bool = False,
    hvg_n_top_genes: int = 2000,
    hvg_layer: Optional[str] = None,
    hvg_flavor: str = "seurat",
    hvg_subset: bool = False,
    excel_annotation_path: Optional[str] = None,
    excel_sheet: str = "Xenium R1 Fig1-5 (supervised)",
    excel_barcode_col: str = "Barcode",
    excel_annotation_col: str = "Annotation",
    excel_cluster_col: str = "Cluster",
) -> str:
    """Preprocess a raw 10x Xenium directory into a minimal h5ad file.

    Loads counts, attaches spatial coordinates from cells metadata, optionally
    merges per-barcode annotation labels from an Excel file, filters
    cells/genes, optionally normalizes, runs HVG selection, and writes
    xenium_preprocessed.h5ad to outdir. Returns the output file path.
    """
    adata = _load_xenium_counts(indir)
    cells_df = _load_xenium_cells_metadata(indir)
    adata = _attach_xenium_spatial(adata, cells_df)

    if excel_annotation_path is not None:
        xlsx = Path(excel_annotation_path)
        if not xlsx.exists():
            raise FileNotFoundError(
                f"Excel annotation file not found: {xlsx}"
            )
        adata = _merge_excel_annotations(
            adata,
            xlsx,
            sheet_name=excel_sheet,
            barcode_col=excel_barcode_col,
            annotation_col=excel_annotation_col,
            cluster_col=excel_cluster_col,
        )

    sc.pp.filter_genes(adata, min_cells=max(0, int(min_cells_per_gene)))
    sc.pp.filter_cells(adata, min_genes=max(0, int(min_genes_per_cell)))

    if max_pct_mt is not None:
        _qc_obs_metrics(adata)
        adata = adata[adata.obs["pct_counts_mt"] <= float(max_pct_mt)].copy()
    if max_counts_per_cell is not None:
        _qc_obs_metrics(adata)
        adata = adata[adata.obs["total_counts"] <= int(max_counts_per_cell)].copy()

    # For seurat_v3, save pre-normalization counts so HVG uses raw count data
    if run_hvg and hvg_layer is None and str(hvg_flavor) == "seurat_v3":
        adata.layers["counts"] = _copy_matrix(adata.X)
        hvg_layer = "counts"

    hvg_layer_name = _materialize_hvg_layer(adata, hvg_layer) if run_hvg else None
    if normalize_and_log1p:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    adata = _apply_hvg_selection(
        adata,
        run_hvg=run_hvg,
        hvg_n_top_genes=hvg_n_top_genes,
        hvg_layer=hvg_layer_name,
        hvg_flavor=hvg_flavor,
        hvg_subset=hvg_subset,
    )

    minimal = _to_minimal_scdata(adata)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "xenium_preprocessed.h5ad"
    minimal.write_h5ad(outpath)
    return str(outpath)


def preprocess_10x(
    indir: str,
    outdir: str,
    modality: str,
    *,
    normalize_and_log1p: bool = True,
    min_genes_per_cell: int = 1,
    min_cells_per_gene: int = 1,
    max_pct_mt: Optional[float] = None,
    max_counts_per_cell: Optional[int] = None,
    run_hvg: bool = False,
    hvg_n_top_genes: int = 2000,
    hvg_layer: Optional[str] = None,
    hvg_flavor: str = "seurat",
    hvg_subset: bool = False,
    excel_annotation_path: Optional[str] = None,
    excel_sheet: str = "Visium",
    excel_barcode_col: str = "Barcode",
    excel_annotation_col: str = "Annotation",
    excel_cluster_col: str = "Cluster",
    spot_diameter_um: float = 55.0,
) -> str:
    """Preprocess a raw 10x spatial transcriptomics directory into a minimal h5ad file.

    Entry point called from Julia via PythonCall. Dispatches to the Visium or
    Xenium pipeline based on ``modality``.

    Parameters
    ----------
    indir : str
        Path to the raw 10x output directory.
    outdir : str
        Path to write the preprocessed h5ad file.
    modality : str
        Either "visium" or "xenium".
    normalize_and_log1p : bool
        Whether to normalize to 10k counts and log1p-transform.
    min_genes_per_cell : int
        Minimum detected genes to keep a cell.
    min_cells_per_gene : int
        Minimum cells expressing a gene to keep it.
    max_pct_mt : float or None
        Maximum mitochondrial percentage for cell filtering.
    max_counts_per_cell : int or None
        Maximum total counts for cell filtering.
    run_hvg : bool
        Whether to run highly-variable gene selection.
    hvg_n_top_genes : int
        Number of top HVGs to select.
    hvg_layer : str or None
        Layer to use for HVG computation (materialized from X if absent).
        When None and hvg_flavor is "seurat_v3", a "counts" layer is
        automatically created from pre-normalization data.
    hvg_flavor : str
        Scanpy HVG flavor ("seurat", "seurat_v3", "cell_ranger").
    hvg_subset : bool
        Whether to subset the AnnData to HVGs only.
    excel_annotation_path : str or None
        Path to an Excel file with per-barcode cell type annotations (Visium only).
    excel_sheet : str
        Sheet name in the Excel file.
    excel_barcode_col : str
        Barcode column name in the Excel file.
    excel_annotation_col : str
        Annotation column name in the Excel file.
    excel_cluster_col : str
        Cluster column name in the Excel file.
    spot_diameter_um : float
        Visium spot diameter in micrometers (default 55.0), used together with
        scale factors to convert pixel coordinates to micrometers. Ignored for
        Xenium (coordinates are already in micrometers).

    Returns
    -------
    str
        Path to the written h5ad file.
    """
    root = Path(indir)
    out = Path(outdir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Input dataset directory not found: {root}")

    mode = str(modality).lower()
    if mode == "visium":
        return _preprocess_visium(
            root,
            out,
            normalize_and_log1p=normalize_and_log1p,
            min_genes_per_cell=min_genes_per_cell,
            min_cells_per_gene=min_cells_per_gene,
            max_pct_mt=max_pct_mt,
            max_counts_per_cell=max_counts_per_cell,
            run_hvg=run_hvg,
            hvg_n_top_genes=hvg_n_top_genes,
            hvg_layer=hvg_layer,
            hvg_flavor=hvg_flavor,
            hvg_subset=hvg_subset,
            excel_annotation_path=excel_annotation_path,
            excel_sheet=excel_sheet,
            excel_barcode_col=excel_barcode_col,
            excel_annotation_col=excel_annotation_col,
            excel_cluster_col=excel_cluster_col,
            spot_diameter_um=spot_diameter_um,
        )
    if mode == "xenium":
        return _preprocess_xenium(
            root,
            out,
            normalize_and_log1p=normalize_and_log1p,
            min_genes_per_cell=min_genes_per_cell,
            min_cells_per_gene=min_cells_per_gene,
            max_pct_mt=max_pct_mt,
            max_counts_per_cell=max_counts_per_cell,
            run_hvg=run_hvg,
            hvg_n_top_genes=hvg_n_top_genes,
            hvg_layer=hvg_layer,
            hvg_flavor=hvg_flavor,
            hvg_subset=hvg_subset,
            excel_annotation_path=excel_annotation_path,
            excel_sheet=excel_sheet,
            excel_barcode_col=excel_barcode_col,
            excel_annotation_col=excel_annotation_col,
            excel_cluster_col=excel_cluster_col,
        )
    raise ValueError(f"Unsupported modality: {modality}")
