using ScCChain
using Test
using PythonCall
using DataFrames

function make_test_h5ad(; include_highly_variable::Bool = true)
    tmpdir = mktempdir()
    path = joinpath(tmpdir, "toy.h5ad")
    globals = pybuiltins.dict()
    globals["path"] = path
    globals["include_highly_variable"] = include_highly_variable
    pyexec(
        """
        import anndata as ad
        import numpy as np
        import pandas as pd
        ad.settings.allow_write_nullable_strings = True
        pd.options.mode.string_storage = "python"

        X = np.array([
            [1.0, 0.0, 2.0, 0.5],
            [0.0, 1.0, 0.0, 3.0],
            [2.0, 2.0, 1.0, 0.0],
        ], dtype=float)
        obs = pd.DataFrame({
            "cell_type": ["A", "B", "C"],
            "Annotation": ["A", "B", "C"],
        }, index=["c0", "c1", "c2"]).astype(object)
        var = pd.DataFrame({"gene_symbol": ["g0", "g1", "g2", "g3"]}, index=["g0", "g1", "g2", "g3"]).astype(object)
        if include_highly_variable:
            var["highly_variable"] = [True, False, True, False]
        obs.index = obs.index.astype(str)
        var.index = var.index.astype(str)
        obsm = {"spatial": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=float)}
        layers = {"counts": (X * 10.0)}
        varm = {"gene_scores": np.array([[0.1, 0.0], [0.0, 0.2], [0.3, 0.1], [0.2, 0.4]], dtype=float)}
        obsp = {"connectivities": np.eye(3, dtype=float)}
        varp = {"feature_graph": np.eye(4, dtype=float)}
        uns = {"dataset_name": "toy"}
        adata = ad.AnnData(
            X=X,
            obs=obs,
            var=var,
            obsm=obsm,
            layers=layers,
            varm=varm,
            obsp=obsp,
            varp=varp,
            uns=uns
        )
        raw_var = pd.DataFrame({"gene_symbol": ["g0", "g1", "g2", "g3"]}, index=["g0", "g1", "g2", "g3"]).astype(object)
        raw_var.index = raw_var.index.astype(str)
        adata.raw = ad.AnnData(X=(X * 100.0), var=raw_var)
        adata.raw.var.index = adata.raw.var.index.astype(str)
        adata.write_h5ad(path)
        """,
        globals,
        globals,
    )
    return path
end

create_minimal_h5ad_fixture() = make_test_h5ad()

function create_nonfinite_spatial_h5ad_fixture()
    tmpdir = mktempdir()
    path = joinpath(tmpdir, "nonfinite_spatial.h5ad")
    globals = pybuiltins.dict()
    globals["path"] = path
    pyexec(
        """
        import anndata as ad
        import numpy as np
        import pandas as pd
        ad.settings.allow_write_nullable_strings = True
        pd.options.mode.string_storage = "python"

        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        obs = pd.DataFrame(index=["c0", "c1"]).astype(object)
        var = pd.DataFrame(index=["g0", "g1"]).astype(object)
        obs.index = obs.index.astype(str)
        var.index = var.index.astype(str)
        obsm = {"spatial": np.array([[0.0, np.nan], [1.0, 1.0]], dtype=float)}
        ad.AnnData(X=X, obs=obs, var=var, obsm=obsm).write_h5ad(path)
        """,
        globals,
        globals,
    )
    return path
end

@testset "scData construction" begin
    handle = scData(:h5ad, "/tmp/a.h5ad", nothing, Dict{String,Any}())
    @test handle.backend == :h5ad
    @test handle.path == "/tmp/a.h5ad"
end

@testset "load_scdata h5ad" begin
    path = create_minimal_h5ad_fixture()
    sd = load_scdata(path; format = :h5ad)
    @test sd isa scData
    @test sd.backend == :h5ad
end

@testset "load_scdata highly-variable subsetting" begin
    path = make_test_h5ad(; include_highly_variable = true)

    sd_full = load_scdata(path; format = :h5ad)
    sd_hvg = load_scdata(path; format = :h5ad, subset_highly_variable = true)

    @test size(expression_matrix(sd_hvg), 2) < size(expression_matrix(sd_full), 2)
    @test all(Bool.(var_table(sd_hvg).highly_variable))
    @test size(spatial_coords(sd_hvg)) == size(spatial_coords(sd_full))
end

@testset "load_scdata highly-variable subsetting requires column" begin
    path = make_test_h5ad(; include_highly_variable = false)
    @test_throws ArgumentError load_scdata(
        path;
        format = :h5ad,
        subset_highly_variable = true,
    )
end

@testset "scData accessors" begin
    path = create_minimal_h5ad_fixture()
    sd = load_scdata(path; format = :h5ad)
    x = expression_matrix(sd)
    @test x isa AbstractMatrix
    @test size(x) == (3, 4)

    coords = spatial_coords(sd)
    @test coords isa AbstractMatrix
    @test size(coords) == (3, 2)
    @test coords[3, 1] == 2.0

    metadata = uns(sd)
    @test metadata isa Dict
    @test metadata["dataset_name"] == "toy"

    @test uns(sd, "dataset_name") == "toy"

    sm = obsm(sd, "spatial")
    @test sm isa AbstractMatrix
    @test size(sm) == (3, 2)
    @test_throws ArgumentError obsm(sd, "does_not_exist")

    rm = raw_expression_matrix(sd)
    @test rm isa AbstractMatrix
    @test size(rm) == (3, 4)
    @test rm[1, 1] == 100.0

    obs = obs_table(sd)
    @test obs isa DataFrame
    @test nrow(obs) == 3
    @test "cell_type" in string.(names(obs))

    var = var_table(sd)
    @test var isa DataFrame
    @test nrow(var) == 4
    @test "gene_symbol" in string.(names(var))

    varm_mat = varm(sd, "gene_scores")
    @test varm_mat isa AbstractMatrix
    @test size(varm_mat) == (4, 2)

    obsp_mat = obsp(sd, "connectivities")
    @test obsp_mat isa AbstractMatrix
    @test size(obsp_mat) == (3, 3)

    varp_mat = varp(sd, "feature_graph")
    @test varp_mat isa AbstractMatrix
    @test size(varp_mat) == (4, 4)

    keys = layers(sd)
    @test keys isa Vector{String}
    @test "counts" in keys

    counts = layer(sd, "counts")
    @test counts isa AbstractMatrix
    @test size(counts) == (3, 4)
    @test counts[1, 1] == 10.0
end

@testset "cell_annotation helper" begin
    path = create_minimal_h5ad_fixture()
    sd = load_scdata(path; format = :h5ad)

    labels_auto = cell_annotation(sd; column = :auto)
    labels_named = cell_annotation(sd; column = "Annotation")

    @test labels_auto == String.(obs_table(sd).Annotation)
    @test labels_named == labels_auto
end

@testset "load_scdata strict validation" begin
    path = create_nonfinite_spatial_h5ad_fixture()
    @test_throws ArgumentError load_scdata(path; format = :h5ad, strict = true)
    @test load_scdata(path; format = :h5ad, strict = false) isa scData
end

@testset "build_cell_graph scData method" begin
    @test hasmethod(build_cell_graph, Tuple{scData,LRPairDB})
end

@testset "build_cell_graph scData integration" begin
    path = create_minimal_h5ad_fixture()
    sd = load_scdata(path; format = :h5ad)
    db = LRPairDB(
        [LRPairRecord("g0—g1", ["g0"], ["g1"], nothing, Dict{String,Any}())],
        :cellchat,
        "human",
    )

    g = build_cell_graph(sd, db; n_neighbors = 2)
    @test g isa CellGraph
    @test size(g.similarity_layer, 1) == size(expression_matrix(sd), 1)
end

@test_throws ArgumentError load_scdata("missing.h5ad"; format = :h5ad)
