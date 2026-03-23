using ScCChain
using DataFrames: DataFrame, nrow
using PythonCall
using SparseArrays
using Test

const _ROOT = normpath(joinpath(@__DIR__, ".."))

function _toy_graph_for_chains()
    sim = sparse(Float32[
        1.0 0.9 0.7 0.0
        0.9 1.0 0.6 0.4
        0.7 0.6 1.0 0.8
        0.0 0.4 0.8 1.0
    ])
    inter1 = sparse(Float32[
        0.0 2.0 0.0 0.0
        0.0 0.0 1.0 0.0
        1.0 0.0 0.0 0.5
        0.0 0.0 0.0 0.0
    ])
    inter2 = sparse(Float32[
        0.0 0.5 0.0 0.0
        0.0 0.0 0.0 1.5
        0.0 0.2 0.0 0.0
        0.0 0.0 0.0 0.0
    ])
    return CellGraph(
        sim,
        [inter1, inter2],
        ["I1", "I2"],
        [1, 2],
        Int[],
        String[],
        Dict{String,Any}(),
    )
end

function _chain_test_scdata()
    tmpdir = mktempdir()
    path = joinpath(tmpdir, "chains_toy.h5ad")
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
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ], dtype=float)
        obs = pd.DataFrame({
            "Annotation": ["A", "B", "C", "D", "E"],
        }, index=["c0", "c1", "c2", "c3", "c4"]).astype(object)
        var = pd.DataFrame({"gene_symbol": ["g0", "g1"]}, index=["g0", "g1"]).astype(object)
        obs.index = obs.index.astype(str)
        var.index = var.index.astype(str)
        obsm = {"spatial": np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]], dtype=float)}
        ad.AnnData(X=X, obs=obs, var=var, obsm=obsm).write_h5ad(path)
        """,
        globals,
        globals,
    )
    return load_scdata(path; format = :h5ad)
end

@testset "Chains API contract" begin
    g = _toy_graph_for_chains()
    out = sample_chains(g; programs = nothing, seed = 7, n_samples = 2, n_steps = 4)
    @test out isa ScCChain.ChainResult
    @test hasproperty(out, :chains)
    @test hasproperty(out, :stacked_matrix)
    @test hasproperty(out, :communication_labels)
    @test hasproperty(out, :metadata)
end

@testset "Chains compatibility signatures" begin
    g = _toy_graph_for_chains()
    programs_stub = (metadata = Dict("sender_index" => Int[], "receiver_index" => Int[]),)
    @test_throws ArgumentError sample_chains(g; programs = programs_stub)

    out = sample_chains(g; programs = nothing, seed = 1, n_samples = 1, n_steps = 2)
    @test haskey(out.metadata, "source_mode")
    @test out.metadata["source_mode"] == :raw_graph
end

@testset "Layer resolution modes" begin
    g = _toy_graph_for_chains()

    out_raw = sample_chains(g; programs = nothing, n_samples = 1, n_steps = 2, seed = 2)
    @test out_raw.metadata["source_mode"] == :raw_graph

    pair_meta = Dict("sender_index" => [1, 2, 3], "receiver_index" => [2, 3, 4])
    probs = Float32[
        0.8 0.1 0.2
        0.2 0.9 0.8
    ]
    programs = (latent_split_softmax = probs, metadata = pair_meta)
    out_prog = sample_chains(g; programs = programs, n_samples = 1, n_steps = 2, seed = 2)
    @test out_prog.metadata["source_mode"] == :program_adjusted
    @test length(out_prog.communication_labels) == length(out_prog.chains)
end

@testset "Compatibility sampling semantics" begin
    g = _toy_graph_for_chains()
    out = sample_chains(
        g;
        programs = nothing,
        q0 = [0.5, 0.5],
        n_samples = 3,
        n_steps = 4,
        q = 0.5,
        cells_interest = nothing,
        ntop_nbrs = nothing,
        communication_threshold = nothing,
        remove_noninteracting_paths = true,
        seed = 42,
    )

    @test !isempty(out.chains)
    @test all(length(c) >= 2 for c in out.chains)
    @test length(out.chains) == size(out.stacked_matrix, 1)
    @test length(out.chains) == length(out.communication_labels)
end

@testset "Chains edge cases and metadata" begin
    g = _toy_graph_for_chains()

    @test_throws ArgumentError sample_chains(g; programs = (metadata = Dict(),), seed = 1)
    @test_throws ArgumentError sample_chains(
        g;
        programs = nothing,
        cells_interest = [0],
        seed = 1,
    )

    out = sample_chains(
        g;
        programs = nothing,
        cells_interest = [1, 2],
        n_samples = 1,
        n_steps = 2,
        seed = 1,
    )
    @test haskey(out.metadata, "skipped_layers")
    @test haskey(out.metadata, "per_layer_counts")
    @test out.metadata["seed"] == 1
end

@testset "Chains sampling keeps sparse graph operations" begin
    n = 2500
    sim = spzeros(Float32, n, n)
    sim[1, 2] = 0.9f0
    sim[2, 1] = 0.9f0
    sim[2, 3] = 0.7f0
    sim[3, 2] = 0.7f0

    inter = spzeros(Float32, n, n)
    inter[1, 2] = 1.2f0
    inter[2, 3] = 0.8f0

    g = CellGraph(sim, [inter], ["I1"], [1], Int[], String[], Dict{String,Any}())

    alloc = @allocated sample_chains(
        g;
        programs = nothing,
        start_from = "receiver",
        n_samples = 1,
        n_steps = 2,
        cells_interest = [2],
        remove_noninteracting_paths = false,
        seed = 13,
    )

    @test alloc < 80_000_000
end



@testset "Chain metadata construction" begin
    chains = [[2, 5, 9], [3, 4]]
    labels = ["I1", "I2"]
    cell_types = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

    md = construct_chain_metadata(chains, labels; cell_annotation = cell_types)

    @test nrow(md) == 2
    @test md.receiver_cell_ids == [9, 4]
    @test md.first_sender_cell_ids == [2, 3]
    @test md.penultimate_sender_cell_ids == [5, 3]
    @test md.chain_lengths == [3, 2]
    @test md.receiver_cell_types == ["I", "D"]
end

@testset "Chain metadata attention enrichment" begin
    chains = [[1, 2, 3], [4, 5]]
    md = construct_chain_metadata(chains, ["I1", "I2"])

    # heads x sender_positions x chains
    attention = Array{Float32}(undef, 2, 2, 2)
    attention[:, :, 1] = Float32[0.1 0.9; 0.3 0.7]
    attention[:, :, 2] = Float32[0.8 0.2; 0.6 0.4]

    add_max_attention_to_chain_metadata!(md, attention, chains)
    @test md.max_attention_sender_cell_ids[1] == 2
    @test md.max_attention_sender_cell_ids[2] == 4
    @test md.max_attention_scores[1] ≈ 0.8 atol=1e-6
    @test md.max_attention_scores[2] ≈ 0.7 atol=1e-6
end

@testset "Chain metadata error enrichment" begin
    expr = Float32[1 0; 0 1; 1 1; 2 0]
    chains = [[1, 3], [2, 4]]
    md = construct_chain_metadata(chains, ["I1", "I2"])

    predictions = Float32[1.0 1.5; 1.0 0.5] # genes x chains

    add_chain_model_errors_to_metadata!(
        md,
        chains,
        expr,
        predictions;
        mode = :mse,
        error_pcts = [50],
    )
    @test hasproperty(md, :chain_wise_errors)
    @test md.chain_wise_errors ≈ [0.0, 0.25]

    add_chain_model_errors_to_metadata!(
        md,
        chains,
        expr,
        predictions;
        mode = :adj_mse,
        error_pcts = [50],
    )
    @test md.chain_wise_errors[1] ≈ 0.0 atol=1e-8
    @test md.chain_wise_errors[2] ≈ 1.0 atol=1e-8
    @test all(isfinite, md.chain_wise_errors)
end

@testset "Chain metadata distance and similarity enrichment" begin
    chains = [[1, 2, 3], [2, 3, 4]]
    md = construct_chain_metadata(chains, ["I1", "I2"])

    coords = Float32[0 0; 1 0; 2 0; 2 1]
    expr = Float32[1 0; 1 0; 0 1; 0 1]

    add_distances_to_chain_metadata!(md, coords)
    add_similarities_to_chain_metadata!(md, expr)

    @test hasproperty(md, :first_sender_receiver_distance)
    @test hasproperty(md, :first_sender_receiver_similarity)
end

@testset "Chain metadata subsetting" begin
    md = DataFrame(
        chain_wise_errors = [0.1, 0.2, 0.3],
        receiver_cell_types = ["A", "B", "A"],
        first_sender_cell_types = ["X", "Y", "X"],
    )
    md[!, Symbol("50")] = [true, false, true]

    sub = subset_chain_metadata!(md, 50; receiver_type = "A", sender_type = "X")
    @test nrow(sub) == 2
end

@testset "construct_chain_metadata scdata overload matches vector overload" begin
    chains = [[2, 5, 4], [1, 3]]
    labels = ["I1", "I2"]
    cell_types = ["A", "B", "C", "D", "E"]
    chain_result = ScCChain.ChainResult(
        chains = chains,
        stacked_matrix = Matrix{Vector{Int}}(undef, 0, 2),
        communication_labels = labels,
        metadata = Dict{String,Any}(),
    )
    sd = _chain_test_scdata()

    md_a = construct_chain_metadata(chains, labels; cell_annotation = cell_types)
    md_b =
        construct_chain_metadata(chain_result; scdata = sd, annotation_col = "Annotation")

    @test md_a == md_b
end

@testset "metadata enrichment overloads match signatures" begin
    chains = [[1, 2, 3], [4, 5]]
    labels = ["I1", "I2"]
    cell_types = ["A", "B", "C", "D", "E"]
    expr = Float32[1 0; 1 0; 0 1; 0 1; 0.5 0.5]
    coords = Float32[0 0; 1 0; 2 0; 2 1; 1 1]
    attention = Array{Float32}(undef, 2, 2, 2)
    attention[:, :, 1] = Float32[0.1 0.9; 0.3 0.7]
    attention[:, :, 2] = Float32[0.8 0.2; 0.6 0.4]
    predictions = Float32[1.0 1.5; 1.0 0.5]

    sd = _chain_test_scdata()
    chain_result = ScCChain.ChainResult(
        chains = chains,
        stacked_matrix = Matrix{Vector{Int}}(undef, 0, 2),
        communication_labels = labels,
        metadata = Dict{String,Any}(),
    )
    model_result = ScCChain.ModelResult(
        predictions = predictions,
        attention_per_head = attention,
        latent = zeros(Float32, 2, 2),
    )

    md_a = construct_chain_metadata(chains, labels; cell_annotation = cell_types)
    md_b =
        construct_chain_metadata(chain_result; scdata = sd, annotation_col = "Annotation")

    add_max_attention_to_chain_metadata!(
        md_a,
        attention,
        chains;
        cell_annotation = cell_types,
    )
    add_max_attention_to_chain_metadata!(
        md_b,
        model_result,
        chain_result;
        scdata = sd,
        annotation_col = "Annotation",
    )

    add_chain_model_errors_to_metadata!(
        md_a,
        chains,
        expr,
        predictions;
        mode = :adj_mse,
        error_pcts = [50],
    )
    add_chain_model_errors_to_metadata!(
        md_b,
        chain_result,
        model_result,
        sd;
        mode = :adj_mse,
        error_pcts = [50],
    )

    add_distances_to_chain_metadata!(md_a, coords)
    add_distances_to_chain_metadata!(md_b, sd)

    add_similarities_to_chain_metadata!(md_a, expr)
    add_similarities_to_chain_metadata!(md_b, sd)

    @test md_a == md_b
end
