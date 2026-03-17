using ScCChain
using Test
using DataFrames
using CSV

@testset "Custom LR Pair DB" begin

    @testset "load_custom_lrpair_db — default columns" begin
        csv = """ligand,receptor
TGFB1,TGFBR1
WNT5A,FZD4
"BMP2, BMP7","BMPR1A, BMPR1B"
"""
        tmpdir = mktempdir()
        tmpfile = joinpath(tmpdir, "custom.csv")
        write(tmpfile, csv)

        db = load_custom_lrpair_db(tmpfile)

        @test db isa LRPairDB
        @test db.source == :custom
        @test db.species == "human"
        @test n_lrpairs(db) == 3

        # Single-gene LR pair
        @test db.records[1].ligands == ["TGFB1"]
        @test db.records[1].receptors == ["TGFBR1"]

        # Auto-generated name
        @test db.records[1].name == "TGFB1—TGFBR1"

        # Multi-subunit (comma-separated)
        @test db.records[3].ligands == ["BMP2", "BMP7"]
        @test db.records[3].receptors == ["BMPR1A", "BMPR1B"]
        @test db.records[3].name == "BMP2_BMP7—BMPR1A_BMPR1B"

        # Pathway is nothing when no column
        @test db.records[1].pathway === nothing
    end

    @testset "merge_lrpair_dbs — basic merge" begin
        recs1 = [
            LRPairRecord("L1—R1", ["L1"], ["R1"], "P1", Dict{String,Any}()),
            LRPairRecord("L2—R2", ["L2"], ["R2"], "P2", Dict{String,Any}()),
        ]
        recs2 = [
            LRPairRecord("L3—R3", ["L3"], ["R3"], "P3", Dict{String,Any}()),
            LRPairRecord("L4—R4", ["L4"], ["R4"], nothing, Dict{String,Any}()),
        ]
        db1 = LRPairDB(recs1, :cellchat, "human")
        db2 = LRPairDB(recs2, :custom, "human")

        merged = merge_lrpair_dbs(db1, db2)
        @test merged.source == :merged
        @test merged.species == "human"
        @test n_lrpairs(merged) == 4
        @test lrpair_names(merged) == ["L1—R1", "L2—R2", "L3—R3", "L4—R4"]
    end

    @testset "merge_lrpair_dbs — duplicate handling (last wins)" begin
        recs1 = [
            LRPairRecord("L1—R1", ["L1"], ["R1"], "P1", Dict{String,Any}()),
            LRPairRecord("SHARED—NAME", ["OLD_L"], ["OLD_R"], "OLD_P", Dict{String,Any}()),
        ]
        recs2 = [
            LRPairRecord("SHARED—NAME", ["NEW_L"], ["NEW_R"], "NEW_P", Dict{String,Any}()),
            LRPairRecord("L3—R3", ["L3"], ["R3"], "P3", Dict{String,Any}()),
        ]
        db1 = LRPairDB(recs1, :cellchat, "human")
        db2 = LRPairDB(recs2, :custom, "human")

        merged = merge_lrpair_dbs(db1, db2)
        @test n_lrpairs(merged) == 3

        # Last DB wins: SHARED—NAME should have NEW_L
        shared = filter(r -> r.name == "SHARED—NAME", merged.records)
        @test length(shared) == 1
        @test shared[1].ligands == ["NEW_L"]
    end

    @testset "merge_lrpair_dbs — species mismatch error" begin
        db1 = LRPairDB(LRPairRecord[], :cellchat, "human")
        db2 = LRPairDB(LRPairRecord[], :custom, "mouse")

        @test_throws ArgumentError merge_lrpair_dbs(db1, db2)

        # With explicit override, no error
        merged = merge_lrpair_dbs(db1, db2; species = "human")
        @test merged.species == "human"
    end

    @testset "merge_lrpair_dbs — custom source" begin
        db1 = LRPairDB(LRPairRecord[], :cellchat, "human")
        db2 = LRPairDB(LRPairRecord[], :custom, "human")

        merged = merge_lrpair_dbs(db1, db2; source = :combined)
        @test merged.source == :combined
    end

    @testset "load_custom_lrpair_db — custom column mapping" begin
        csv = """gene_ligand,gene_receptor,name,pw
TGFB1,TGFBR1,myInteraction,TGFb
WNT5A,FZD4,wntSignal,WNT
"""
        tmpdir = mktempdir()
        tmpfile = joinpath(tmpdir, "mapped.csv")
        write(tmpfile, csv)

        db = load_custom_lrpair_db(
            tmpfile;
            ligand_col = :gene_ligand,
            receptor_col = :gene_receptor,
            lrpair_name_col = :name,
            pathway_col = :pw,
        )

        @test n_lrpairs(db) == 2
        @test db.records[1].name == "myInteraction"
        @test db.records[1].pathway == "TGFb"
        @test db.records[2].name == "wntSignal"
        @test db.records[2].pathway == "WNT"
    end

    @testset "load_custom_lrpair_db — custom species and source" begin
        csv = "ligand,receptor\nTGFB1,TGFBR1\n"
        tmpdir = mktempdir()
        tmpfile = joinpath(tmpdir, "custom.csv")
        write(tmpfile, csv)

        db = load_custom_lrpair_db(tmpfile; species = "zebrafish", source = :my_db)
        @test db.species == "zebrafish"
        @test db.source == :my_db
    end

    @testset "load_custom_lrpair_db — missing required column error" begin
        csv = "lig,rec\nTGFB1,TGFBR1\n"
        tmpdir = mktempdir()
        tmpfile = joinpath(tmpdir, "bad.csv")
        write(tmpfile, csv)

        @test_throws ArgumentError load_custom_lrpair_db(tmpfile)
        @test_throws ArgumentError load_custom_lrpair_db(
            tmpfile;
            ligand_col = :lig,
            receptor_col = :missing_col,
        )
    end

    @testset "load_custom_lrpair_db — missing optional column error" begin
        csv = "ligand,receptor\nTGFB1,TGFBR1\n"
        tmpdir = mktempdir()
        tmpfile = joinpath(tmpdir, "custom.csv")
        write(tmpfile, csv)

        @test_throws ArgumentError load_custom_lrpair_db(
            tmpfile;
            pathway_col = :nonexistent,
        )
        @test_throws ArgumentError load_custom_lrpair_db(tmpfile; lrpair_name_col = :nope)
    end

    @testset "load_custom_lrpair_db — empty rows skipped with warning" begin
        csv = """ligand,receptor
TGFB1,TGFBR1
,FZD4
WNT5A,
BMP2,BMPR1A
"""
        tmpdir = mktempdir()
        tmpfile = joinpath(tmpdir, "gaps.csv")
        write(tmpfile, csv)

        db = @test_logs (:warn,) (:warn,) load_custom_lrpair_db(tmpfile)
        @test n_lrpairs(db) == 2
        @test lrpair_names(db) == ["TGFB1—TGFBR1", "BMP2—BMPR1A"]
    end

    @testset "load_custom_lrpair_db — file not found" begin
        @test_throws ArgumentError load_custom_lrpair_db("/nonexistent/path.csv")
    end

    @testset "load_custom_lrpair_db — empty species error" begin
        csv = "ligand,receptor\nTGFB1,TGFBR1\n"
        tmpdir = mktempdir()
        tmpfile = joinpath(tmpdir, "custom.csv")
        write(tmpfile, csv)

        @test_throws ArgumentError load_custom_lrpair_db(tmpfile; species = "")
    end

    @testset "load_custom_lrpair_db — extend keyword" begin
        existing = LRPairDB(
            [
                LRPairRecord("L1—R1", ["L1"], ["R1"], "P1", Dict{String,Any}()),
                LRPairRecord(
                    "SHARED—NAME",
                    ["OLD"],
                    ["OLD_R"],
                    nothing,
                    Dict{String,Any}(),
                ),
            ],
            :cellchat,
            "human",
        )

        csv3 = "ligand,receptor,iname\nCUSTOM_L,CUSTOM_R,SHARED—NAME\nNEW_L,NEW_R,NEW—INTERACTION\n"
        tmpdir = mktempdir()
        tmpfile2 = joinpath(tmpdir, "extend2.csv")
        write(tmpfile2, csv3)

        db = load_custom_lrpair_db(tmpfile2; lrpair_name_col = :iname, extend = existing)

        @test n_lrpairs(db) == 3  # L1—R1, SHARED—NAME (custom wins), NEW—INTERACTION
        shared = filter(r -> r.name == "SHARED—NAME", db.records)
        @test length(shared) == 1
        @test shared[1].ligands == ["CUSTOM_L"]  # custom wins
    end

    @testset "load_custom_lrpair_db — xlsx without XLSX.jl" begin
        tmpdir = mktempdir()
        tmpfile = joinpath(tmpdir, "data.xlsx")
        write(tmpfile, "fake xlsx content")

        # Should error with install instructions
        err = try
            load_custom_lrpair_db(tmpfile)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("XLSX.jl", err.msg)
    end

    @testset "load_lrpair_db — format=:custom dispatch" begin
        csv = "ligand,receptor\nTGFB1,TGFBR1\n"
        tmpdir = mktempdir()
        tmpfile = joinpath(tmpdir, "custom.csv")
        write(tmpfile, csv)

        db = load_lrpair_db(tmpfile; format = :custom)
        @test db isa LRPairDB
        @test db.source == :custom
        @test n_lrpairs(db) == 1
    end

end # Custom LR Pair DB
