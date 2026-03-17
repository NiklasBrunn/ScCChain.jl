using ScCChain
using LinearAlgebra
using SparseArrays
using Test
using PythonCall

function create_minimal_h5ad_fixture()
    tmpdir = mktempdir()
    path = joinpath(tmpdir, "toy_graph.h5ad")
    globals = pybuiltins.dict()
    globals["path"] = path
    pyexec(
        """
        import anndata as ad
        import numpy as np
        import pandas as pd
        ad.settings.allow_write_nullable_strings = True
        pd.options.mode.string_storage = "python"

        X = np.array([
            [1.0, 0.5, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 3.0],
        ], dtype=float)
        obs = pd.DataFrame(index=["c0", "c1", "c2"]).astype(object)
        var = pd.DataFrame({"gene_symbol": ["g0", "g1", "g2"]}, index=["g0", "g1", "g2"]).astype(object)
        obs.index = obs.index.astype(str)
        var.index = var.index.astype(str)
        obsm = {"spatial": np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=float)}
        ad.AnnData(X=X, obs=obs, var=var, obsm=obsm).write_h5ad(path)
        """,
        globals,
        globals,
    )
    return path
end

function _toy_db()
    recs = [
        LRPairRecord("L1—R1", ["G1"], ["G2"], "P1", Dict{String,Any}()),
        LRPairRecord("L3—R4", ["G3"], ["G4"], "P2", Dict{String,Any}()),
        LRPairRecord("MISS—R9", ["MISSING"], ["G2"], "P3", Dict{String,Any}()),
    ]
    return LRPairDB(recs, :cellchat, "human")
end

function _toy_multigene_db()
    recs = [LRPairRecord("L12—R34", ["G1", "G2"], ["G3", "G4"], "Pmix", Dict{String,Any}())]
    return LRPairDB(recs, :cellchat, "human")
end

@testset "Graph module" begin
    @test hasfield(CellGraph, :similarity_layer)
    @test hasfield(CellGraph, :communication_layers)
    @test hasfield(CellGraph, :kept_communication_indices)
    @test hasfield(CellGraph, :dropped_communication_indices)
    @test hasfield(CellGraph, :dropped_communication_reasons)

    expr = Float64[
        1 2 0 1
        0 1 3 0
        2 0 1 1
        1 1 1 2
    ]
    coords = Float64[
        0 0
        1 0
        0 1
        1 1
    ]
    db = _toy_db()
    gene_names = ["G1", "G2", "G3", "G4"]

    @test_throws ArgumentError build_cell_graph(expr, coords, db)
    graph = build_cell_graph(expr, coords, db; gene_names = gene_names)

    @test graph isa CellGraph
    @test graph.similarity_layer isa SparseMatrixCSC
    @test length(graph.communication_layers) == 2

    @test_throws ArgumentError build_cell_graph(expr, coords, db; gene_names = ["G1"])
    @test_throws ArgumentError build_cell_graph(
        expr,
        coords,
        db;
        gene_names = gene_names,
        n_neighbors = 50,
        radius = 300.0,
    )

    g_knn = build_cell_graph(expr, coords, db; gene_names = gene_names, n_neighbors = 2)
    @test nnz(g_knn.similarity_layer) > 0

    g_radius = build_cell_graph(expr, coords, db; gene_names = gene_names, radius = 1.5)
    @test nnz(g_radius.similarity_layer) > 0

    g_sim = build_cell_graph(expr, coords, db; gene_names = gene_names, n_neighbors = 2)
    @test g_sim.similarity_layer == transpose(g_sim.similarity_layer)

    g_no_self = build_cell_graph(
        expr,
        coords,
        db;
        gene_names = gene_names,
        n_neighbors = 2,
        allow_self_loops_similarity = false,
    )
    @test all(iszero, diag(g_no_self.similarity_layer))

    g_inter = build_cell_graph(expr, coords, db; gene_names = gene_names, n_neighbors = 2)
    @test length(g_inter.communication_layers) == 2
    @test g_inter.dropped_communication_indices == [3]
    @test any(contains("missing"), lowercase.(g_inter.dropped_communication_reasons))

    g_no_self_comm = build_cell_graph(
        expr,
        coords,
        db;
        gene_names = gene_names,
        n_neighbors = 2,
        allow_self_loops_communication = false,
    )
    @test all(all(iszero, diag(layer)) for layer in g_no_self_comm.communication_layers)

    mode_db = _toy_multigene_db()
    dist14_sq = sum(abs2, coords[1, :] .- coords[4, :])
    distance_weight = exp(-dist14_sq)

    expected_geom = sqrt(sqrt(expr[1, 1] * expr[1, 2]) * sqrt(expr[4, 3] * expr[4, 4]))
    expected_prod = (expr[1, 1] * expr[1, 2]) * (expr[4, 3] * expr[4, 4])
    expected_min = min(min(expr[1, 1], expr[1, 2]), min(expr[4, 3], expr[4, 4]))
    expected_geom_weighted = expected_geom * distance_weight
    expected_cos_14 =
        dot(expr[1, :], expr[4, :]) / (norm(expr[1, :]) * norm(expr[4, :]) + eps(Float64))

    g_default =
        build_cell_graph(expr, coords, mode_db; gene_names = gene_names, radius = 2.0)
    g_geom = build_cell_graph(
        expr,
        coords,
        mode_db;
        gene_names = gene_names,
        radius = 2.0,
        communication_score_mode = :geometric_mean,
    )
    g_prod = build_cell_graph(
        expr,
        coords,
        mode_db;
        gene_names = gene_names,
        radius = 2.0,
        communication_score_mode = :product,
    )
    g_min = build_cell_graph(
        expr,
        coords,
        mode_db;
        gene_names = gene_names,
        radius = 2.0,
        communication_score_mode = :minimum,
    )
    g_alpha_one = build_cell_graph(
        expr,
        coords,
        mode_db;
        gene_names = gene_names,
        radius = 2.0,
        alpha = 1.0,
    )

    @test g_default.communication_layers[1][1, 4] ≈ expected_geom atol = 1.0f-6
    @test g_geom.communication_layers[1][1, 4] ≈ expected_geom atol = 1.0f-6
    @test g_prod.communication_layers[1][1, 4] ≈ expected_prod atol = 1.0f-6
    @test g_min.communication_layers[1][1, 4] ≈ expected_min atol = 1.0f-6
    @test g_default.metadata["alpha"] == 0.0
    @test g_alpha_one.metadata["alpha"] == 1.0
    @test g_alpha_one.communication_layers[1][1, 4] ≈ expected_geom_weighted atol = 1.0f-6
    @test g_alpha_one.similarity_layer[1, 4] ≈ expected_cos_14 * distance_weight atol =
        1.0f-6
    g_alpha_zero = build_cell_graph(
        expr,
        coords,
        mode_db;
        gene_names = gene_names,
        radius = 2.0,
        alpha = 0.0,
    )
    @test g_alpha_zero.metadata["alpha"] == 0.0
    @test g_alpha_zero.communication_layers[1][1, 4] ≈ expected_geom atol = 1.0f-6
    @test g_alpha_zero.similarity_layer[1, 4] ≈ expected_cos_14 atol = 1.0f-6

    @test g_geom.metadata["communication_score_mode"] == "geometric_mean"
    @test_throws ArgumentError build_cell_graph(
        expr,
        coords,
        mode_db;
        gene_names = gene_names,
        radius = 2.0,
        communication_score_mode = :not_a_mode,
    )
end

@testset "build_cell_graph uses all duplicate gene-symbol columns" begin
    expr = Float64[
        2 8 1
        4 16 3
    ]
    coords = Float64[
        0 0
        1 0
    ]
    db = LRPairDB(
        [LRPairRecord("G1-R1", ["G1"], ["R1"], "toy", Dict{String,Any}())],
        :cellchat,
        "human",
    )
    gene_names = ["G1", "G1", "R1"]

    g = build_cell_graph(
        expr,
        coords,
        db;
        gene_names = gene_names,
        radius = 2.0,
        alpha = 0.0,
        communication_score_mode = :geometric_mean,
    )

    expected_sender_1 = sqrt(expr[1, 1] * expr[1, 2])
    expected_score_12 = sqrt(expected_sender_1 * expr[2, 3])
    @test g.communication_layers[1][1, 2] ≈ expected_score_12 atol = 1.0f-6
end

@testset "build_cell_graph scData keeps resolvable LR pairs" begin
    path = create_minimal_h5ad_fixture()
    sd = load_scdata(path; format = :h5ad)
    db = LRPairDB(
        [LRPairRecord("g0-g1", ["g0"], ["g1"], "toy", Dict{String,Any}())],
        :cellchat,
        "human",
    )

    g = build_cell_graph(sd, db; n_neighbors = 2)
    @test g isa CellGraph
    @test !isempty(g.kept_communication_indices)
    @test isempty(g.dropped_communication_indices)
end
