using ScCChain
using Test
using CodecZlib

function write_gzip(path::String, content::AbstractString)
    open(path, "w") do io
        gz = GzipCompressorStream(io)
        write(gz, content)
        close(gz)
    end
end

function create_visium_10x_fixture(; rich_hvg::Bool = false)
    root = mktempdir()
    mtx_dir = joinpath(root, "filtered_feature_bc_matrix")
    spatial_dir = joinpath(root, "spatial")
    mkpath(mtx_dir)
    mkpath(spatial_dir)

    matrix_body = if rich_hvg
        """
%%MatrixMarket matrix coordinate integer general
6 8 40
1 1 1
2 1 2
3 1 10
5 1 3
6 1 20
1 2 1
2 2 4
3 2 10
4 2 6
5 2 3
1 3 1
2 3 8
3 3 10
5 3 3
6 3 20
1 4 1
2 4 12
3 4 10
4 4 6
5 4 3
1 5 1
2 5 16
3 5 10
5 5 3
6 5 20
1 6 1
2 6 20
3 6 10
4 6 6
5 6 3
1 7 1
2 7 24
3 7 10
5 7 3
6 7 20
1 8 1
2 8 28
3 8 10
4 8 6
5 8 3
"""
    else
        """
%%MatrixMarket matrix coordinate integer general
3 2 3
1 1 1
3 1 2
2 2 3
"""
    end
    barcodes =
        rich_hvg ?
        "cellA-1\ncellB-1\ncellC-1\ncellD-1\ncellE-1\ncellF-1\ncellG-1\ncellH-1\n" :
        "cellA-1\ncellB-1\n"
    features = if rich_hvg
        "gene_1\tG1\tGene Expression\n" *
        "gene_2\tG2\tGene Expression\n" *
        "gene_3\tG3\tGene Expression\n" *
        "gene_4\tG4\tGene Expression\n" *
        "gene_5\tG5\tGene Expression\n" *
        "gene_6\tG6\tGene Expression\n"
    else
        "gene_1\tG1\tGene Expression\n" *
        "gene_2\tG2\tGene Expression\n" *
        "gene_3\tG3\tGene Expression\n"
    end
    tissue_positions = if rich_hvg
        "barcode,in_tissue,array_row,array_col,pxl_col_in_fullres,pxl_row_in_fullres\n" *
        "cellA-1,1,0,0,10,20\n" *
        "cellB-1,1,0,1,30,40\n" *
        "cellC-1,1,1,0,50,60\n" *
        "cellD-1,1,1,1,70,80\n" *
        "cellE-1,1,2,0,90,100\n" *
        "cellF-1,1,2,1,110,120\n" *
        "cellG-1,1,3,0,130,140\n" *
        "cellH-1,1,3,1,150,160\n"
    else
        "barcode,in_tissue,array_row,array_col,pxl_col_in_fullres,pxl_row_in_fullres\n" *
        "cellA-1,1,0,0,10,20\n" *
        "cellB-1,1,0,1,30,40\n"
    end

    write_gzip(joinpath(mtx_dir, "matrix.mtx.gz"), matrix_body)
    write_gzip(joinpath(mtx_dir, "barcodes.tsv.gz"), barcodes)
    write_gzip(joinpath(mtx_dir, "features.tsv.gz"), features)
    write(joinpath(spatial_dir, "tissue_positions.csv"), tissue_positions)
    return root
end

function create_xenium_10x_fixture(; bad_ids::Bool = false, rich_hvg::Bool = false)
    root = mktempdir()
    mtx_dir = joinpath(root, "cell_feature_matrix")
    mkpath(mtx_dir)

    matrix_body = if rich_hvg
        """
%%MatrixMarket matrix coordinate integer general
6 8 40
1 1 2
2 1 2
3 1 9
5 1 3
6 1 18
1 2 2
2 2 4
3 2 9
4 2 5
5 2 3
1 3 2
2 3 8
3 3 9
5 3 3
6 3 18
1 4 2
2 4 12
3 4 9
4 4 5
5 4 3
1 5 2
2 5 16
3 5 9
5 5 3
6 5 18
1 6 2
2 6 20
3 6 9
4 6 5
5 6 3
1 7 2
2 7 24
3 7 9
5 7 3
6 7 18
1 8 2
2 8 28
3 8 9
4 8 5
5 8 3
"""
    else
        """
%%MatrixMarket matrix coordinate integer general
3 2 3
1 1 2
2 2 4
3 2 1
"""
    end
    barcodes =
        rich_hvg ? "cell1\ncell2\ncell3\ncell4\ncell5\ncell6\ncell7\ncell8\n" :
        "cell1\ncell2\n"
    features = if rich_hvg
        "gene_1\tG1\tGene Expression\n" *
        "gene_2\tG2\tGene Expression\n" *
        "gene_3\tG3\tGene Expression\n" *
        "gene_4\tG4\tGene Expression\n" *
        "gene_5\tG5\tGene Expression\n" *
        "gene_6\tG6\tGene Expression\n"
    else
        "gene_1\tG1\tGene Expression\n" *
        "gene_2\tG2\tGene Expression\n" *
        "gene_3\tG3\tGene Expression\n"
    end
    cells_csv = if bad_ids
        "cell_id,x_centroid,y_centroid\nother1,50,20\nother2,60,25\n"
    elseif rich_hvg
        "cell_id,x_centroid,y_centroid\ncell1,50,20\ncell2,60,25\ncell3,70,30\ncell4,80,35\ncell5,90,40\ncell6,100,45\ncell7,110,50\ncell8,120,55\n"
    else
        "cell_id,x_centroid,y_centroid\ncell1,50,20\ncell2,60,25\n"
    end

    write_gzip(joinpath(mtx_dir, "matrix.mtx.gz"), matrix_body)
    write_gzip(joinpath(mtx_dir, "barcodes.tsv.gz"), barcodes)
    write_gzip(joinpath(mtx_dir, "features.tsv.gz"), features)
    write(joinpath(root, "cells.csv"), cells_csv)
    return root
end

@testset "preprocess_scdata API validation" begin
    @test isdefined(ScCChain, :preprocess_scdata)
    @test_throws ArgumentError preprocess_scdata("missing_dir"; modality = :visium)
    @test_throws ArgumentError preprocess_scdata(pwd(); modality = :invalid)

    indir = create_visium_10x_fixture(; rich_hvg = true)
    @test_throws ArgumentError preprocess_scdata(
        indir;
        modality = :visium,
        run_hvg = true,
        hvg_n_top_genes = 0,
    )
end

@testset "preprocess_scdata visium synthetic" begin
    indir = create_visium_10x_fixture()
    sd, outpath = preprocess_scdata(indir; modality = :visium, normalize_and_log1p = false)
    @test isfile(outpath)
    @test sd isa scData
    @test size(expression_matrix(sd), 1) > 0
    @test size(spatial_coords(sd), 2) == 2
end

@testset "preprocess_scdata xenium synthetic" begin
    indir = create_xenium_10x_fixture()
    sd, outpath = preprocess_scdata(indir; modality = :xenium, normalize_and_log1p = false)
    @test isfile(outpath)
    @test sd isa scData
    @test size(expression_matrix(sd), 1) == size(spatial_coords(sd), 1)
    @test size(spatial_coords(sd), 2) == 2
end

@testset "preprocess_scdata xenium id mismatch fails clearly" begin
    indir = create_xenium_10x_fixture(; bad_ids = true)
    @test_throws ArgumentError preprocess_scdata(
        indir;
        modality = :xenium,
        normalize_and_log1p = false,
    )
end

@testset "preprocess_scdata minimal schema contract" begin
    visium_dir = create_visium_10x_fixture()
    sd_visium, _ =
        preprocess_scdata(visium_dir; modality = :visium, normalize_and_log1p = false)
    @test size(expression_matrix(sd_visium), 2) > 0
    @test_throws ArgumentError layer(sd_visium, "counts")
    @test_throws ArgumentError raw_expression_matrix(sd_visium)

    xenium_dir = create_xenium_10x_fixture()
    sd_xenium, _ =
        preprocess_scdata(xenium_dir; modality = :xenium, normalize_and_log1p = false)
    @test size(expression_matrix(sd_xenium), 2) > 0
    @test_throws ArgumentError layer(sd_xenium, "counts")
    @test_throws ArgumentError raw_expression_matrix(sd_xenium)
end

@testset "preprocess_scdata gene-name contract" begin
    visium_dir = create_visium_10x_fixture()
    sd_visium, _ =
        preprocess_scdata(visium_dir; modality = :visium, normalize_and_log1p = false)
    vvar = var_table(sd_visium)
    vcols = Set(String.(names(vvar)))
    @test ("index" in vcols) || ("gene_symbol" in vcols)

    xenium_dir = create_xenium_10x_fixture()
    sd_xenium, _ =
        preprocess_scdata(xenium_dir; modality = :xenium, normalize_and_log1p = false)
    xvar = var_table(sd_xenium)
    xcols = Set(String.(names(xvar)))
    @test ("index" in xcols) || ("gene_symbol" in xcols)
end

@testset "preprocess_scdata HVG coverage" begin
    @testset "preprocess_scdata HVG annotation without subsetting" begin
        indir = create_visium_10x_fixture(; rich_hvg = true)
        sd, _ = preprocess_scdata(
            indir;
            modality = :visium,
            normalize_and_log1p = true,
            run_hvg = true,
            hvg_n_top_genes = 2,
        )

        vtbl = var_table(sd)
        has_hvg = "highly_variable" in names(vtbl)
        @test has_hvg
        if has_hvg
            @test count(Bool, vtbl.highly_variable) == 2
        end
        @test size(expression_matrix(sd), 2) == size(vtbl, 1)
    end

    @testset "preprocess_scdata HVG subset output" begin
        indir = create_xenium_10x_fixture(; rich_hvg = true)
        sd_full, _ = preprocess_scdata(
            indir;
            modality = :xenium,
            normalize_and_log1p = true,
            run_hvg = true,
            hvg_n_top_genes = 2,
            hvg_subset = false,
        )
        sd_hvg, _ = preprocess_scdata(
            indir;
            modality = :xenium,
            normalize_and_log1p = true,
            run_hvg = true,
            hvg_n_top_genes = 2,
            hvg_subset = true,
        )

        @test size(expression_matrix(sd_hvg), 2) == 2
        @test size(expression_matrix(sd_hvg), 2) < size(expression_matrix(sd_full), 2)
        hvg_names = names(var_table(sd_hvg))
        @test "highly_variable" in hvg_names
        if "highly_variable" in hvg_names
            @test all(Bool.(var_table(sd_hvg).highly_variable))
        end
    end
end

@testset "preprocess_scdata HVG counts layer path" begin
    indir = create_visium_10x_fixture(; rich_hvg = true)
    sd, _ = preprocess_scdata(
        indir;
        modality = :visium,
        normalize_and_log1p = true,
        run_hvg = true,
        hvg_layer = "counts",
        hvg_flavor = "seurat_v3",
        hvg_n_top_genes = 2,
    )

    @test "highly_variable" in names(var_table(sd))
    @test_throws ArgumentError layer(sd, "counts")
end

@testset "preprocess_scdata visium excel annotation merge" begin
    indir = create_visium_10x_fixture()
    # Create a minimal Excel file with annotation labels via PythonCall
    using PythonCall
    openpyxl = pyimport("openpyxl")
    xlsx_path = joinpath(indir, "annotations.xlsx")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Visium"
    ws.append(pylist(["Barcode", "Cluster", "Annotation"]))
    ws.append(pylist(["cellA-1", "1", "TypeA"]))
    ws.append(pylist(["cellB-1", "2", "TypeB"]))
    wb.save(xlsx_path)

    sd, outpath = preprocess_scdata(
        indir;
        modality = :visium,
        normalize_and_log1p = false,
        excel_annotation_path = xlsx_path,
    )
    @test isfile(outpath)
    obs = obs_table(sd)
    @test "Annotation" in names(obs)
    @test "Cluster" in names(obs)
end

@testset "preprocess_scdata excel missing file raises error" begin
    indir = create_visium_10x_fixture()
    @test_throws ArgumentError preprocess_scdata(
        indir;
        modality = :visium,
        normalize_and_log1p = false,
        excel_annotation_path = joinpath(indir, "nonexistent.xlsx"),
    )
end

@testset "preprocess_scdata without HVG keeps schema unchanged" begin
    indir = create_xenium_10x_fixture(; rich_hvg = true)
    sd, _ = preprocess_scdata(indir; modality = :xenium, normalize_and_log1p = true)
    @test !("highly_variable" in names(var_table(sd)))
end

@testset "preprocessing docs artifacts exist" begin
    repo_root = dirname(dirname(pathof(ScCChain)))
    @test isfile(joinpath(repo_root, "examples", "04_preprocessing_visium_xenium.ipynb"))
    @test isfile(joinpath(repo_root, "examples", "17_preprocessing_hvg_selection.ipynb"))
end
