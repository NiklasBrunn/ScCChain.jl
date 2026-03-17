using ScCChain
using Test
using PythonCall
using Flux
using Random

function create_minimal_h5ad_fixture_for_model()
    tmpdir = mktempdir()
    path = joinpath(tmpdir, "toy_model.h5ad")
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
            [1.0, 0.0, 2.0, 0.5],
            [0.0, 1.0, 0.0, 3.0],
            [2.0, 2.0, 1.0, 0.0],
        ], dtype=float)
        obs = pd.DataFrame({"cell_type": ["A", "B", "C"]}, index=["c0", "c1", "c2"]).astype(object)
        var = pd.DataFrame({"gene_symbol": ["g0", "g1", "g2", "g3"]}, index=["g0", "g1", "g2", "g3"]).astype(object)
        obs.index = obs.index.astype(str)
        var.index = var.index.astype(str)

        obsm = {"spatial": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=float)}
        ad.AnnData(X=X, obs=obs, var=var, obsm=obsm).write_h5ad(path)
        """,
        globals,
        globals,
    )
    return path
end

@testset "Model API contract" begin
    chains = [[1, 2], [2, 3]]
    @test_throws Exception train_model(chains, nothing)
end

@testset "Model batch builder semantics" begin
    chains = [[1, 2, 3]]
    expr = Float32[
        10 0
        20 1
        30 2
    ]

    Xb, Yb, mask, idx = ScCChain.Model._build_batch(
        expr,
        chains,
        [1];
        receiver_first = true,
        padding = true,
    )
    @test Yb[:, 1] == Float32[30, 2]
    @test Xb[:, 1, 1] == Float32[30, 2]
    @test Xb[:, 2, 1] == Float32[10, 0]
    @test Xb[:, 3, 1] == Float32[20, 1]
    @test idx == [1]
    @test size(mask) == (2, 1)
end

@testset "Receiver-query encoder forward shapes" begin
    m = ScCChain.Model._init_receiver_query_model(4, 3; n_heads = 2, qk_dim = 2, v_dim = 2)
    x = rand(Float32, 4, 5, 7)
    mask = zeros(Float32, 4, 7)
    yhat, attn = ScCChain.Model._forward_with_attention(m, x; attn_mask = mask)

    @test size(yhat) == (4, 7)
    @test size(attn, 1) == 2
    @test size(attn, 2) == 4
    @test size(attn, 3) == 7
end

@testset "train_model matrix overload" begin
    chains = [[1, 2, 3], [2, 3, 4], [1, 3, 4], [4, 2, 1]]
    expr = rand(Float32, 4, 6)
    trained = train_model(chains, nothing, expr; n_epochs = 3, batch_size = 2, seed = 7)
    @test hasproperty(trained, :is_trained)
    @test trained.is_trained
end

@testset "predict matrix overload" begin
    chains = [[1, 2, 3], [2, 3, 4]]
    expr = rand(Float32, 4, 5)
    m = train_model(chains, nothing, expr; n_epochs = 2, batch_size = 2, seed = 1)
    out = predict(m, chains, expr)
    @test out isa ScCChain.ModelResult
    @test hasproperty(out, :predictions)
    @test hasproperty(out, :attention_per_head)
    @test hasproperty(out, :latent)
    @test size(out.predictions, 1) == size(expr, 2)
    @test size(out.predictions, 2) == length(chains)
end

@testset "scData overloads" begin
    path = create_minimal_h5ad_fixture_for_model()
    sd = load_scdata(path; format = :h5ad)
    chains = [[1, 2], [2, 3]]

    m = train_model(chains, nothing, sd; n_epochs = 2, batch_size = 2, seed = 5)
    out = predict(m, chains, sd)
    @test size(out.predictions, 2) == length(chains)
end

@testset "Model input guards" begin
    expr = rand(Float32, 3, 4)
    @test_throws ArgumentError train_model(Vector{Vector{Int}}(), nothing, expr)
    @test_throws ArgumentError train_model([[1]], nothing, expr)
    @test_throws ArgumentError train_model([[1, 9]], nothing, expr)
    expr_bad = copy(expr)
    expr_bad[1, 1] = NaN32
    @test_throws ArgumentError train_model([[1, 2]], nothing, expr_bad)
end


@testset "Attention dropout parity with compatibility path" begin
    defaults = ScCChain.Model._default_training_parameters()
    @test haskey(defaults, "p_dropout")

    m = ScCChain.Model._init_receiver_query_model(
        4,
        3;
        n_heads = 2,
        qk_dim = 2,
        v_dim = 2,
        p_dropout = 0.5f0,
    )
    @test hasproperty(m.encoder, :attn_drop)
    @test m.encoder.attn_drop.p ≈ 0.5f0

    x = rand(Float32, 4, 5, 8)
    mask = zeros(Float32, 4, 8)

    Flux.trainmode!(m.encoder.attn_drop)
    z1 = m.encoder(x; attn_mask = mask)
    z2 = m.encoder(x; attn_mask = mask)
    @test sum(abs.(z1 .- z2)) > 0.0f0

    Flux.testmode!(m.encoder.attn_drop)
    z3 = m.encoder(x; attn_mask = mask)
    z4 = m.encoder(x; attn_mask = mask)
    @test z3 ≈ z4
end

@testset "Predict latent consistency under dropout" begin
    Random.seed!(17)
    chains = [[1, 2, 3], [2, 3, 4], [1, 4, 2], [4, 3, 1]]
    expr = rand(Float32, 4, 6)
    m = train_model(
        chains,
        nothing,
        expr;
        n_epochs = 2,
        batch_size = 2,
        seed = 17,
        p_dropout = 0.5f0,
    )

    Flux.trainmode!(m.encoder.attn_drop)
    out = predict(m, chains, expr)
    @test out.predictions ≈ m.decoder(out.latent)
end
