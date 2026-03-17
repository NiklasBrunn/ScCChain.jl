using ScCChain
using Test
using LinearAlgebra
using Statistics

@testset "pca" begin
    @testset "basic PCA" begin
        X = Float64[
            1.0 2.0 3.0
            4.0 5.0 6.0
            7.0 8.0 9.0
            10.0 11.0 12.0
        ]
        result = pca(X; k = 2, standardize = true)
        @test size(result) == (4, 2)
        @test all(isfinite, result)
    end

    @testset "k clamped to min(n_obs, n_feat)" begin
        X = randn(3, 5)
        result = pca(X; k = 10)
        @test size(result, 2) == 3  # clamped to n_obs
    end

    @testset "single feature column" begin
        X = reshape(Float64[1.0, 2.0, 3.0, 4.0], 4, 1)
        result = pca(X; k = 1, standardize = true)
        @test size(result) == (4, 1)
        @test all(isfinite, result)
    end

    @testset "zero-variance column handled" begin
        X = Float64[
            1.0 5.0
            1.0 6.0
            1.0 7.0
        ]
        result = pca(X; k = 1, standardize = true)
        @test size(result) == (3, 1)
        @test all(isfinite, result)
    end

    @testset "promote_float64=false preserves Float32" begin
        X = Float32[1 2; 3 4; 5 6]
        result = pca(X; k = 2, promote_float64 = false)
        @test eltype(result) == Float32
    end

    @testset "standardize=false skips scaling" begin
        X = Float64[1 2; 3 4; 5 6; 7 8]
        result = pca(X; k = 2, standardize = false)
        @test size(result) == (4, 2)
        @test all(isfinite, result)
    end

    @testset "pca(scData) filters zero-sum columns" begin
        using PythonCall
        tmpdir = mktempdir()
        path = joinpath(tmpdir, "pca_test.h5ad")
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
                [1.0, 0.0, 2.0],
                [3.0, 0.0, 4.0],
                [5.0, 0.0, 6.0],
            ], dtype=float)
            obs = pd.DataFrame(index=["c0", "c1", "c2"]).astype(object)
            var = pd.DataFrame(index=["g0", "g1", "g2"]).astype(object)
            obs.index = obs.index.astype(str)
            var.index = var.index.astype(str)
            obsm = {"spatial": np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)}
            ad.AnnData(X=X, obs=obs, var=var, obsm=obsm).write_h5ad(path)
            """,
            globals,
            globals,
        )
        sd = load_scdata(path; format = :h5ad)
        result = pca(sd; k = 2)
        # 3 genes, 1 is all-zero → 2 non-zero columns → k clamped to 2
        @test size(result, 1) == 3
        @test size(result, 2) <= 2
        @test all(isfinite, result)
    end
end
