using ScCChain
using Test
using Statistics

@testset "Eval Module" begin

    @testset "gini_impurity" begin
        # Uniform distribution: 1 - 1/K
        @test gini_impurity(["A", "B", "C", "D"]) ≈ 1 - 1 / 4

        # Single class: 0
        @test gini_impurity(["A", "A", "A"]) == 0.0

        # Two classes, balanced
        @test gini_impurity([1, 2, 1, 2]) ≈ 0.5

        # Empty input
        @test gini_impurity(Int[]) == 0.0

        # Weighted: all weight on one class
        @test gini_impurity(["A", "B"]; weights = [1.0, 0.0]) == 0.0

        # Weighted: equal weight
        @test gini_impurity(["A", "B"]; weights = [3.0, 3.0]) ≈ 0.5

        # Negative weight throws
        @test_throws ArgumentError gini_impurity([1, 2]; weights = [1.0, -1.0])

        # Mismatched length throws
        @test_throws ArgumentError gini_impurity([1, 2, 3]; weights = [1.0, 2.0])
    end

    @testset "gini_impurity_normalized" begin
        # Uniform over 4 classes: normalized = 1.0
        @test gini_impurity_normalized(["A", "B", "C", "D"]) ≈ 1.0

        # Single class: 0
        @test gini_impurity_normalized(["A", "A"]) == 0.0

        # Empty: 0
        @test gini_impurity_normalized(String[]) == 0.0

        # total_K normalization ceiling
        labels = repeat(["A", "B"], 50)
        gn = gini_impurity_normalized(labels; total_K = 10)
        @test 0 < gn < 1.0  # less than 1 because only 2/10 classes used

        # total_K < observed classes throws
        @test_throws ArgumentError gini_impurity_normalized(["A", "B", "C"]; total_K = 2)
    end

    @testset "downstream_gene_activity_score" begin
        # 5 cells, 4 genes
        expr = Float64[
            1.0 0.0 2.0 0.5
            0.5 1.0 0.0 1.0
            0.0 0.5 1.0 0.0
            2.0 1.5 0.5 0.0
            1.0 1.0 1.0 1.0
        ]
        genenames = ["GeneA", "GeneB", "GeneC", "GeneD"]
        # 3 chains: receiver is last element
        chains = [[1, 2, 5], [3, 4, 1], [2, 3, 4]]

        # Score downstream activity of GeneA and GeneC in receivers
        scores = downstream_gene_activity_score(expr, chains, genenames, ["GeneA", "GeneC"])
        # Receiver cell 5: mean(1.0, 1.0) = 1.0
        # Receiver cell 1: mean(1.0, 2.0) = 1.5
        # Receiver cell 4: mean(2.0, 0.5) = 1.25
        @test length(scores) == 3
        @test scores[1] ≈ 1.0
        @test scores[2] ≈ 1.5
        @test scores[3] ≈ 1.25

        # Subset with chain_inds
        scores_sub = downstream_gene_activity_score(
            expr,
            chains,
            genenames,
            ["GeneA"];
            chain_inds = [1, 3],
        )
        @test length(scores_sub) == 2
        @test scores_sub[1] ≈ 1.0   # receiver 5: GeneA = 1.0
        @test scores_sub[2] ≈ 2.0   # receiver 4: GeneA = 2.0

        # No matching genes throws
        @test_throws ArgumentError downstream_gene_activity_score(
            expr,
            chains,
            genenames,
            ["NONEXISTENT"],
        )
    end

    @testset "get_avg_expression" begin
        expr = Float64[
            1.0 0.0 2.0
            0.5 1.0 0.0
            0.0 0.5 1.0
        ]
        genenames = ["GeneA", "GeneB", "GeneC"]

        avg, matched, qs = get_avg_expression(expr, genenames, ["GeneA", "GeneC"])
        @test length(avg) == 3
        @test avg[1] ≈ 1.5   # mean(1.0, 2.0)
        @test avg[2] ≈ 0.25  # mean(0.5, 0.0)
        @test avg[3] ≈ 0.5   # mean(0.0, 1.0)
        @test Set(matched) == Set(["GeneA", "GeneC"])
        @test haskey(qs, :q25)
        @test haskey(qs, :q50)
        @test haskey(qs, :q75)

        # No matching genes: returns empty
        avg_e, matched_e, qs_e = get_avg_expression(expr, genenames, ["NONE"])
        @test isempty(avg_e)
        @test isempty(matched_e)
        @test isnan(qs_e.q25)
    end

end # Eval Module
