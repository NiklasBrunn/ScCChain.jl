using ScCChain
using Test

@testset "resolve_example_dataset with explicit data_dir" begin
    tmp = mktempdir()
    @test resolve_example_dataset("toy"; data_dir = tmp) == abspath(tmp)
    @test_throws ArgumentError resolve_example_dataset(
        "toy";
        data_dir = joinpath(tmp, "nonexistent"),
    )
end

@testset "resolve_example_dataset with SCCCHAIN_DATA_DIR" begin
    tmp = mktempdir()
    dataset_dir = joinpath(tmp, "toy")
    mkpath(dataset_dir)

    withenv("SCCCHAIN_DATA_DIR" => tmp) do
        @test resolve_example_dataset("toy") == abspath(dataset_dir)
    end

    withenv("SCCCHAIN_DATA_DIR" => nothing) do
        @test_throws ArgumentError resolve_example_dataset("toy")
    end
end
