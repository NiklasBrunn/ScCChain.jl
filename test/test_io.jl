using ScCChain
using Test
using DataFrames

@testset "IO Module" begin

    @testset "LRPairRecord construction" begin
        rec = LRPairRecord(
            "TGFB1—TGFBR1_TGFBR2",
            ["TGFB1"],
            ["TGFBR1", "TGFBR2"],
            "TGFb",
            Dict{String,Any}("annotation" => "Secreted Signaling"),
        )
        @test rec.name == "TGFB1—TGFBR1_TGFBR2"
        @test rec.ligands == ["TGFB1"]
        @test rec.receptors == ["TGFBR1", "TGFBR2"]
        @test rec.pathway == "TGFb"
        @test rec.metadata["annotation"] == "Secreted Signaling"
    end

    @testset "LRPairDB construction and accessors" begin
        recs = [
            LRPairRecord("L1—R1", ["L1"], ["R1"], "P1", Dict{String,Any}()),
            LRPairRecord("L2_L3—R2", ["L2", "L3"], ["R2"], "P2", Dict{String,Any}()),
            LRPairRecord("L4—R3_R4", ["L4"], ["R3", "R4"], nothing, Dict{String,Any}()),
        ]
        db = LRPairDB(recs, :cellchat, "human")

        @test n_lrpairs(db) == 3
        @test lrpair_names(db) == ["L1—R1", "L2_L3—R2", "L4—R3_R4"]
        @test Set(all_ligands(db)) == Set(["L1", "L2", "L3", "L4"])
        @test Set(all_receptors(db)) == Set(["R1", "R2", "R3", "R4"])
        @test db.source == :cellchat
        @test db.species == "human"
    end

    @testset "to_dataframe" begin
        recs = [
            LRPairRecord("L1—R1", ["L1"], ["R1"], "P1", Dict{String,Any}()),
            LRPairRecord("L2—R2", ["L2"], ["R2"], nothing, Dict{String,Any}()),
        ]
        db = LRPairDB(recs, :cellchat, "human")
        df = to_dataframe(db)

        @test df isa DataFrame
        @test nrow(df) == 2
        @test names(df) == ["lrpair_name", "ligands", "receptors", "pathway"]
        @test df.lrpair_name == ["L1—R1", "L2—R2"]
        @test df.ligands[1] == ["L1"]
        @test df.pathway[2] === nothing
    end

    @testset "load_cellchat_db" begin
        # Create a minimal CellChat-format CSV in a temp file
        csv_content = """interaction_name,pathway_name,ligand,receptor,agonist,antagonist,co_A_receptor,co_I_receptor,evidence,annotation,interaction_name_2,is_neurotransmitter,ligand_symbol,ligand_family,ligand_location,ligand_keyword,ligand_secreted_type,ligand_transmembrane,receptor_symbol,receptor_family,receptor_location,receptor_keyword,receptor_surfaceome_main,receptor_surfaceome_sub,receptor_adhesome,receptor_secreted_type,receptor_transmembrane,version
TGFB1_TGFBR1_TGFBR2,TGFb,a,b,,,,,,,,,TGFB1,,,,,,"TGFBR1, TGFBR2",,,,,,,,v2
WNT5A_FZD4,WNT,c,d,,,,,,Secreted Signaling,,,,,,,,WNT5A,,,,FZD4,,,,,v2
BMP2_BMPR1A,BMP,e,f,,,,,,ECM-Receptor,,,,,,,,BMP2,,,,BMPR1A,,,,,v2"""

        tmpdir = mktempdir()
        tmpfile = joinpath(tmpdir, "CellChatDB_human_interaction.csv")
        write(tmpfile, csv_content)

        db = load_cellchat_db(tmpfile; species = "human")

        @test db isa LRPairDB
        @test db.source == :cellchat
        @test db.species == "human"
        @test n_lrpairs(db) == 3

        # Check first record: TGFB1 with complex receptor
        rec1 = db.records[1]
        @test rec1.ligands == ["TGFB1"]
        @test rec1.receptors == ["TGFBR1", "TGFBR2"]
        @test rec1.pathway == "TGFb"

        # Check LR pair name construction: ligands_joined—receptors_joined
        @test rec1.name == "TGFB1—TGFBR1_TGFBR2"

        # Check metadata preserved
        @test haskey(rec1.metadata, "annotation")
    end

    @testset "load_cellchat_db with communication_type filter" begin
        csv_content = """interaction_name,pathway_name,ligand,receptor,agonist,antagonist,co_A_receptor,co_I_receptor,evidence,annotation,interaction_name_2,is_neurotransmitter,ligand_symbol,ligand_family,ligand_location,ligand_keyword,ligand_secreted_type,ligand_transmembrane,receptor_symbol,receptor_family,receptor_location,receptor_keyword,receptor_surfaceome_main,receptor_surfaceome_sub,receptor_adhesome,receptor_secreted_type,receptor_transmembrane,version
a,TGFb,a,b,,,,,,Secreted Signaling,,,,,,,,TGFB1,,,,TGFBR1,,,,,v2
b,WNT,c,d,,,,,,Secreted Signaling,,,,,,,,WNT5A,,,,FZD4,,,,,v2
c,BMP,e,f,,,,,,ECM-Receptor,,,,,,,,BMP2,,,,BMPR1A,,,,,v2"""

        tmpdir = mktempdir()
        tmpfile = joinpath(tmpdir, "CellChatDB_human_interaction.csv")
        write(tmpfile, csv_content)

        db = load_cellchat_db(
            tmpfile;
            species = "human",
            communication_type = "Secreted Signaling",
        )
        @test n_lrpairs(db) == 2

        db2 = load_cellchat_db(
            tmpfile;
            species = "human",
            communication_type = ["ECM-Receptor"],
        )
        @test n_lrpairs(db2) == 1
    end

    @testset "load_lrpair_db dispatch" begin
        cellchat_csv = """interaction_name,pathway_name,ligand,receptor,agonist,antagonist,co_A_receptor,co_I_receptor,evidence,annotation,interaction_name_2,is_neurotransmitter,ligand_symbol,ligand_family,ligand_location,ligand_keyword,ligand_secreted_type,ligand_transmembrane,receptor_symbol,receptor_family,receptor_location,receptor_keyword,receptor_surfaceome_main,receptor_surfaceome_sub,receptor_adhesome,receptor_secreted_type,receptor_transmembrane,version
a,TGFb,a,b,,,,,,Secreted Signaling,,,,,,,,TGFB1,,,,TGFBR1,,,,,v2
b,WNT,c,d,,,,,,Secreted Signaling,,,,,,,,WNT5A,,,,FZD4,,,,,v2"""
        tmpdir = mktempdir()
        cellchat_file = joinpath(tmpdir, "CellChatDB_human_interaction.csv")
        write(cellchat_file, cellchat_csv)

        db = load_lrpair_db(cellchat_file; format = :cellchat, species = "human")
        @test db.source == :cellchat
        @test n_lrpairs(db) == 2
    end

    @testset "bundled databases are available" begin
        @test isfile(default_cellchat_path("human"))
        @test isfile(default_cellchat_path("mouse"))

        db_cellchat = load_cellchat_db(; species = "human")

        @test db_cellchat.source == :cellchat
        @test n_lrpairs(db_cellchat) > 1000

        db_auto_cellchat = load_lrpair_db(; format = :cellchat, species = "human")
        @test n_lrpairs(db_auto_cellchat) == n_lrpairs(db_cellchat)
    end

    @testset "load_ppi_database" begin
        # Default bundled path
        ppi = load_ppi_database("human")
        @test ppi isa DataFrame
        @test nrow(ppi) > 1000
        @test names(ppi) == ["source", "target", "experimental_score"]

        ppi_mouse = load_ppi_database("mouse")
        @test ppi_mouse isa DataFrame
        @test nrow(ppi_mouse) > 1000

        # Invalid species
        @test_throws ArgumentError load_ppi_database("fish")

        # Custom path
        tmpdir = mktempdir()
        cp(default_ppi_database_path("human"), joinpath(tmpdir, "human_signaling_ppi.csv"))
        ppi_custom = load_ppi_database("human"; data_path = tmpdir)
        @test nrow(ppi_custom) == nrow(ppi)
    end

    @testset "extract_downstream_genes" begin
        # Build a small synthetic PPI network:
        # A -> B (0.9), A -> C (0.5)
        # B -> D (0.8), B -> E (0.3)
        # C -> F (0.7), C -> D (0.4)
        ppi = DataFrame(
            source = ["A", "A", "B", "B", "C", "C"],
            target = ["B", "C", "D", "E", "F", "D"],
            experimental_score = [0.9, 0.5, 0.8, 0.3, 0.7, 0.4],
        )

        # top_n=2: second-order targets sorted by max score: D(0.8), F(0.7), E(0.3)
        # With include_immediate=true: immediate=[B,C] ∪ top2=[D,F]
        genes = extract_downstream_genes(ppi, "A"; top_n = 2)
        @test "D" in genes
        @test "F" in genes
        @test "B" in genes  # immediate
        @test "C" in genes  # immediate
        @test !("E" in genes)  # not in top 2

        # include_immediate=false
        genes_no_imm =
            extract_downstream_genes(ppi, "A"; top_n = 2, include_immediate = false)
        @test "D" in genes_no_imm
        @test "F" in genes_no_imm
        @test !("B" in genes_no_imm)

        # Vector input
        genes_vec = extract_downstream_genes(ppi, ["A"]; top_n = 10)
        @test length(genes_vec) > 0

        # No targets found
        empty_ppi =
            DataFrame(source = String[], target = String[], experimental_score = Float64[])
        @test extract_downstream_genes(empty_ppi, "X"; top_n = 5) == String[]

        # top_percent
        genes_pct = extract_downstream_genes(ppi, "A"; top_n = nothing, top_percent = 50)
        @test length(genes_pct) > 0
    end

end # IO Module
