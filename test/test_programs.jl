using ScCChain
using Test
using SparseArrays
using DataFrames
using Zygote
using Statistics

const _ROOT = normpath(joinpath(@__DIR__, ".."))

function _toy_graph()
    sim = sparse(Float32[
        1 0.8 0
        0.8 1 0.5
        0 0.5 1
    ])
    l1 = sparse(Float32[
        0 1 0
        0 0 2
        1 0 0
    ])
    l2 = sparse(Float32[
        0 0.2 0
        0.4 0 0
        0 0.1 0
    ])
    return CellGraph(
        sim,
        [l1, l2],
        ["L1-R1", "L2-R2"],
        [1, 2],
        Int[],
        String[],
        Dict{String,Any}(),
    )
end

@testset "Programs exports" begin
    @test isdefined(ScCChain, :discover_programs)
    @test isdefined(ScCChain, :discover_programs_bae)
end

@testset "Split transforms" begin
    W = Float32[-2 3; 4 -5; 0 1]
    Wsplit = ScCChain.Programs.split_loadings_nonnegative(W)
    @test size(Wsplit) == (3, 4)
    @test all(Wsplit .>= 0)
    @test W[:, 1] ≈ Wsplit[:, 1] .- Wsplit[:, 2]
    @test W[:, 2] ≈ Wsplit[:, 3] .- Wsplit[:, 4]

    Z = Float32[1 -1 0.5; -0.2 0.3 0.0]
    Zs = ScCChain.Programs.split_softmax_latent(Z)
    @test all(Zs .>= 0)
    @test all(abs.(vec(sum(Zs, dims = 1)) .- 1.0f0) .< 1.0f-5)
    g = Zygote.gradient(z -> sum(abs2, ScCChain.Programs.split_softmax_latent(z)), Z)[1]
    @test size(g) == size(Z)
    @test all(isfinite, g)
end

@testset "Pair matrix extraction" begin
    g = _toy_graph()
    M = ScCChain.Programs.build_pair_feature_matrix(g)
    @test size(M.X, 2) == length(g.communication_layers)
    @test length(M.sender_index) == size(M.X, 1)
    @test length(M.receiver_index) == size(M.X, 1)
    @test M.communication_names == g.communication_names
    @test all(isfinite, M.X)
end

@testset "Select basis utility" begin
    M = Float32[
        0.95 0.05
        0.90 0.10
        0.10 0.90
        0.05 0.95
    ]
    t = ScCChain.Programs._select_basis(
        M;
        nbins = 64,
        j_threshold = 0.2,
        min_support = 1,
        return_stats = true,
    )
    @test Set(t.keep) == Set([1, 2])
    @test isempty(setdiff([1, 2], t.keep))
end

@testset "BAE boosting kernels" begin
    X = Float32[1 0 2; 0 1 1; 1 1 0; 0 0 1]
    hp = ScCChain.Programs.BAEHyperparameters(
        zdim = 2,
        max_iter = 2,
        batchsize = 4,
        M = 1,
        ϵ = 0.01f0,
    )
    model = ScCChain.Programs.init_bae(size(X, 2), hp; seed = 7)
    ScCChain.Programs.compL2Boost!(
        model,
        1,
        X,
        Float32[0.2, -0.1, 0.3, -0.2],
        vec(sum(X .^ 2, dims = 1)),
    )
    @test any(model.encoder_weights[:, 1] .!= 0)
end

@testset "BAE train outputs" begin
    g = _toy_graph()
    adv = discover_programs_bae(
        g;
        n_zdims = 2,
        seed = 3,
        n_runs = 1,
        max_iter = 3,
        batchsize = 8,
        M = 1,
        ϵ = 0.01f0,
    )
    @test adv isa ScCChain.ProgramResult
    @test size(adv.encoder_weights, 2) == 2
    @test size(adv.latent, 2) == 2
    @test all(isfinite, adv.encoder_weights)
    @test all(isfinite, adv.latent)
    @test haskey(adv.metadata, "sparsity")
end

@testset "BAE preprocessing standardization semantics" begin
    X = Float32[
        1 2
        3 4
        5 6
    ]
    Xst = ScCChain.Programs._standardize_cols(X)
    expected = (X .- mean(X, dims = 1)) ./ std(X, dims = 1, corrected = true)
    @test Xst ≈ expected atol=1.0f-6
end

@testset "BAE preprocessing fails on zero-variance features" begin
    Xbad = Float32[
        1 5
        1 7
        1 9
    ]
    @test_throws ArgumentError ScCChain.Programs._standardize_cols(Xbad)
end

@testset "BAE clustering artifacts without basis filtering" begin
    g = _toy_graph()
    out = discover_programs_bae(g; n_zdims = 2, seed = 2, max_iter = 3, batchsize = 8)
    @test out isa ScCChain.ProgramResult
    @test hasproperty(out, :cluster_probs)
    @test hasproperty(out, :cluster_labels)
    @test size(out.cluster_probs, 1) == 4
    @test size(out.cluster_probs, 2) == length(out.cluster_labels)
    @test all(1 .<= out.cluster_labels .<= size(out.cluster_probs, 1))
end

@testset "BAE basis selection artifacts" begin
    g = _toy_graph()
    out = discover_programs_bae(
        g;
        n_zdims = 2,
        seed = 4,
        n_runs = 1,
        max_iter = 3,
        batchsize = 8,
        basis_selection = true,
        basis_nbins = 128,
        basis_j_threshold = 0.0,
        basis_min_support = 1,
    )

    @test out isa ScCChain.ProgramResult
    @test hasproperty(out, :cluster_probs)
    @test hasproperty(out, :cluster_labels)
    @test hasproperty(out, :basis_selection)
    @test hasproperty(out, :cp_mapping)
    @test size(out.cluster_probs, 2) == length(out.cluster_labels)
    @test sort(unique(out.cluster_labels)) == collect(1:maximum(out.cluster_labels))
end

@testset "BAE basis filtering semantics" begin
    g = _toy_graph()
    out = discover_programs_bae(
        g;
        n_zdims = 2,
        seed = 9,
        max_iter = 3,
        batchsize = 8,
        basis_selection = true,
        basis_nbins = 64,
        basis_j_threshold = 0.0,
        basis_min_support = 1,
    )
    @test out.basis_selection.enabled == true
    @test !isempty(out.basis_selection.keep)
    @test size(out.encoder_weights, 2) == 2  # raw encoder weights retain all zdim columns
    @test length(out.cluster_labels) == size(out.cluster_probs, 2)
end

@testset "discover_programs contract" begin
    g = _toy_graph()
    out = discover_programs(g; n_zdims = 2, seed = 5, n_runs = 1, max_iter = 3)
    @test out isa ScCChain.ProgramResult
    @test size(out.encoder_weights, 2) == 2
    @test size(out.loadings_split_nonnegative, 2) == 4
    @test all(out.loadings_split_nonnegative .>= 0)
    @test size(out.latent, 2) == 2
    @test all(out.latent_split_softmax .>= 0)
end

@testset "Programs BAE API" begin
    g = _toy_graph()
    out = discover_programs(g; n_zdims = 2, seed = 11, max_iter = 2, batchsize = 4)

    @test out isa ScCChain.ProgramResult
    @test hasproperty(out, :encoder_weights)
    @test hasproperty(out, :loadings_split_nonnegative)
    @test hasproperty(out, :latent)
    @test hasproperty(out, :latent_split_softmax)
    @test hasproperty(out, :metadata)
end

@testset "top_features_per_program" begin
    g = _toy_graph()
    out = discover_programs(g; n_zdims = 2, seed = 12, max_iter = 2, batchsize = 4)
    tf = top_features_per_program(out)
    @test tf isa Dict{String,DataFrame}
    @test !isempty(tf)
    for (_, df) in tf
        @test "Features" in names(df)
        @test "Scores" in names(df)
        @test "normScores" in names(df)
    end
end

@testset "Programs to communication layers" begin
    ip = Float32[
        0.8 0.2
        0.1 0.9
        0.7 0.3
    ]
    pair_meta = (sender_index = [1, 1, 2], receiver_index = [1, 2, 2])
    layers = programs_to_communication_layers(
        ip,
        2,
        pair_meta;
        CP_cutoff = true,
        self_comm = false,
        materialize_dense = false,
        cutoff = 1.0f-5,
    )
    @test length(layers) == 2
    @test size(layers[1]) == (2, 2)
    @test size(layers[2]) == (2, 2)
    @test layers[1][1, 1] == 0
    @test layers[2][1, 1] == 0
end

@testset "Communication-layer conversion options" begin
    ip = sparse(Float32[
        0.4 0.6
        0.7 0.3
        0.0 1.0
    ])
    pair_meta = (sender_index = [1, 1, 2], receiver_index = [1, 2, 2])
    layers = programs_to_communication_layers(
        ip,
        2,
        pair_meta;
        CP_cutoff = false,
        self_comm = true,
        cutoff = 0.0f0,
    )
    @test length(layers) == 2
    @test nnz(layers[1]) > 0
    @test nnz(layers[2]) > 0
end

@testset "Communication layers keep raw program magnitudes" begin
    ip = Float32[
        2.0 0.5
        1.0 3.0
    ]
    pair_meta = (sender_index = [1, 2], receiver_index = [2, 1])
    layers = programs_to_communication_layers(
        ip,
        2,
        pair_meta;
        CP_cutoff = false,
        self_comm = true,
        cutoff = 0.0f0,
    )
    @test layers[1][1, 2] == 2.0f0
    @test layers[1][2, 1] == 1.0f0
    @test layers[2][1, 2] == 0.5f0
    @test layers[2][2, 1] == 3.0f0
end
