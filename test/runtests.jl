using ScCChain
using Test

@testset "ScCChain.jl" begin

    @testset "Package loads" begin
        # Smoke test: all submodules must load without error
        @test isdefined(ScCChain, :build_cell_graph)
        @test isdefined(ScCChain, :discover_programs)
        @test isdefined(ScCChain, :sample_chains)
        @test isdefined(ScCChain, :train_model)
        @test isdefined(ScCChain, :predict)
        @test isdefined(ScCChain, :gini_impurity)
        @test isdefined(ScCChain, :load_lrpair_db)
        @test isdefined(ScCChain, :load_custom_lrpair_db)
        @test isdefined(ScCChain, :merge_lrpair_dbs)
        @test isdefined(ScCChain, :plot_spatial)
    end

    include("test_io.jl")
    include("test_scdata_io.jl")
    include("test_example_datasets.jl")
    include("test_preprocessing.jl")
    include("test_graph.jl")
    include("test_programs.jl")
    include("test_chains.jl")
    include("test_model.jl")
    include("test_eval.jl")
    include("test_plotting.jl")
    include("test_utils.jl")
    include("test_custom_db.jl")

end
