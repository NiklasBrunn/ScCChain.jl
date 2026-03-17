using Plots

@testset "Plotting" begin
    @testset "plot_spatial" begin
        coords = Float64[0 0; 1 0; 0 1; 1 1; 0.5 0.5]
        types = ["A", "B", "A", "B", "C"]
        values = [0.1, 0.5, 0.3, 0.9, 0.7]

        @testset "no annotation" begin
            p = plot_spatial(coords)
            @test p isa Plots.Plot
        end

        @testset "categorical annotation" begin
            p = plot_spatial(coords, types)
            @test p isa Plots.Plot
        end

        @testset "continuous annotation" begin
            p = plot_spatial(coords, values; data_type = :continuous)
            @test p isa Plots.Plot
        end

        @testset "continuous with percentage" begin
            pct = [10.0, 50.0, 30.0, 90.0, 70.0]
            p = plot_spatial(coords, pct; data_type = :continuous, percentage = true)
            @test p isa Plots.Plot
        end

        @testset "custom palette" begin
            p = plot_spatial(
                coords,
                types;
                custompalette = ["#FF0000", "#00FF00", "#0000FF"],
            )
            @test p isa Plots.Plot
        end
    end

    @testset "plot_spatial!" begin
        coords = Float64[0 0; 1 0; 0 1; 1 1; 0.5 0.5]
        types = ["A", "B", "A", "B", "C"]
        values = [0.1, 0.5, 0.3, 0.9, 0.7]

        @testset "overlay no annotation" begin
            base = plot_spatial(coords)
            p = plot_spatial!(base, coords)
            @test p isa Plots.Plot
        end

        @testset "overlay discrete with color_map" begin
            base = plot_spatial(coords)
            cmap = Dict("A" => "#FF0000", "B" => "#00FF00", "C" => "#0000FF")
            p = plot_spatial!(base, coords, types; color_map = cmap)
            @test p isa Plots.Plot
        end

        @testset "overlay continuous" begin
            base = plot_spatial(coords)
            p = plot_spatial!(base, coords, values; base_type = :continuous)
            @test p isa Plots.Plot
        end

        @testset "overlay with alpha_variable" begin
            base = plot_spatial(coords)
            cmap = Dict("A" => "#FF0000", "B" => "#00FF00", "C" => "#0000FF")
            alphas = [0.2, 0.8, 0.4, 1.0, 0.6]
            p = plot_spatial!(
                base,
                coords,
                types;
                color_map = cmap,
                alpha_variable = alphas,
            )
            @test p isa Plots.Plot
        end
    end

    @testset "plot_chains!" begin
        coords = Float64[0 0; 1 0; 2 0; 0 1; 1 1; 2 1; 0 2; 1 2; 2 2; 0.5 0.5]
        types = ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"]

        # Build a synthetic stacked_matrix: 4 chains, each length 3
        # Column 1: path (Vector{Int}), Column 2: layer indices (Vector{Int})
        # Layer index 0 = similarity edge, 1 = communication edge
        stacked = Matrix{Vector{Int}}(undef, 4, 2)
        stacked[1, 1] = [1, 2, 3];
        stacked[1, 2] = [0, 1]
        stacked[2, 1] = [4, 5, 6];
        stacked[2, 2] = [0, 1]
        stacked[3, 1] = [7, 8, 9];
        stacked[3, 2] = [0, 1]
        stacked[4, 1] = [1, 5, 9];
        stacked[4, 2] = [0, 1]

        labels = ["CP1", "CP1", "CP2", "CP2"]
        colormap = Dict("CP1" => "#FF0000", "CP2" => "#0000FF")

        @testset "plain chains" begin
            p = plot_chains!(coords, nothing, stacked)
            @test p isa Plots.Plot
        end

        @testset "chains with communication colors" begin
            p = plot_chains!(
                coords,
                types,
                stacked;
                communication_labels = labels,
                communication_colormap = colormap,
            )
            @test p isa Plots.Plot
        end

        @testset "chains with error weighting" begin
            errors = Float64[0.1, 0.5, 0.3, 0.9]
            p = plot_chains!(coords, types, stacked; error_vec = errors)
            @test p isa Plots.Plot
        end

        @testset "chains with both communication colors and error weighting" begin
            errors = Float64[0.1, 0.5, 0.3, 0.9]
            p = plot_chains!(
                coords,
                types,
                stacked;
                communication_labels = labels,
                communication_colormap = colormap,
                error_vec = errors,
            )
            @test p isa Plots.Plot
        end

        @testset "chains with subsample" begin
            p = plot_chains!(coords, nothing, stacked; subsample = 2)
            @test p isa Plots.Plot
        end

        @testset "chains with cell type highlighting" begin
            ct_colormap = Dict("A" => "#FF0000", "B" => "#00FF00", "C" => "#0000FF")
            p = plot_chains!(
                coords,
                types,
                stacked;
                cell_type_colormap = ct_colormap,
                selected_cell_types = ["A", "B"],
            )
            @test p isa Plots.Plot
        end
    end

    @testset "plot_cell_pairs!" begin
        coords = Float64[0 0; 1 0; 2 0; 0 1; 1 1; 2 1; 0 2; 1 2; 2 2; 0.5 0.5]
        types = ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"]

        chains = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 5, 9]]
        labels = ["CP1", "CP1", "CP2", "CP2"]
        colormap = Dict("CP1" => "#FF0000", "CP2" => "#0000FF")

        @testset "first to last" begin
            p = plot_cell_pairs!(
                coords,
                types,
                chains;
                communication_labels = labels,
                communication_colormap = colormap,
            )
            @test p isa Plots.Plot
        end

        @testset "max attention" begin
            # Synthetic 3D attention: (n_heads=2, n_senders=2, n_chains=4)
            A = rand(Float64, 2, 2, 4)
            p = plot_cell_pairs!(
                coords,
                types,
                chains;
                communication_labels = labels,
                communication_colormap = colormap,
                link_from = :max_attention,
                A = A,
            )
            @test p isa Plots.Plot
        end

        @testset "with error weighting" begin
            errors = Float64[0.1, 0.5, 0.3, 0.9]
            p = plot_cell_pairs!(
                coords,
                types,
                chains;
                communication_labels = labels,
                communication_colormap = colormap,
                error_vec = errors,
            )
            @test p isa Plots.Plot
        end

        @testset "error weighting + max attention" begin
            A = rand(Float64, 2, 2, 4)
            errors = Float64[0.1, 0.5, 0.3, 0.9]
            p = plot_cell_pairs!(
                coords,
                types,
                chains;
                communication_labels = labels,
                communication_colormap = colormap,
                link_from = :max_attention,
                A = A,
                error_vec = errors,
            )
            @test p isa Plots.Plot
        end

        @testset "no communication labels (default red)" begin
            p = plot_cell_pairs!(coords, nothing, chains)
            @test p isa Plots.Plot
        end

        @testset "with subsample" begin
            p = plot_cell_pairs!(
                coords,
                types,
                chains;
                communication_labels = labels,
                communication_colormap = colormap,
                subsample = 2,
            )
            @test p isa Plots.Plot
        end
    end

    @testset "plot_chord" begin
        using DataFrames

        @testset "_build_flow_matrix" begin
            df = DataFrame(
                sender = ["A", "A", "B", "B", "A"],
                receiver = ["B", "B", "A", "C", "C"],
            )
            mat, labels = ScCChain.Plotting._build_flow_matrix(df, :sender, :receiver)
            @test labels == ["A", "B", "C"]
            @test size(mat) == (3, 3)
            # A->B: 2, B->A: 1, B->C: 1, A->C: 1
            @test mat[1, 2] == 2.0  # A->B
            @test mat[2, 1] == 1.0  # B->A
            @test mat[2, 3] == 1.0  # B->C
            @test mat[1, 3] == 1.0  # A->C
            @test mat[1, 1] == 0.0  # A->A (no self)
            @test mat[3, 3] == 0.0  # C->C (no self)
        end

        @testset "arc layout" begin
            mat = [0.0 2.0; 1.0 0.0]
            labels = ["A", "B"]
            starts, ends, out_ends, order, out_deg, in_deg, degree =
                ScCChain.Plotting._compute_arc_layout(mat, labels, :size, 0.0, true)
            @test order == [1, 2]
            @test out_deg == [2.0, 1.0]
            @test in_deg == [1.0, 2.0]
            @test degree == [3.0, 3.0]
            @test ends .- starts ≈ [180.0, 180.0]
            @test out_ends .- starts ≈ [120.0, 60.0]
            total_span = sum(ends .- starts)
            @test total_span ≈ 360.0 atol=1e-10
        end

        @testset "basic chord plot" begin
            df = DataFrame(
                sender = ["A", "A", "B", "B", "C"],
                receiver = ["B", "C", "A", "C", "A"],
            )
            p = plot_chord(df; source_column = :sender, target_column = :receiver)
            @test p isa Plots.Plot
        end

        @testset "chord plot with colormap" begin
            df = DataFrame(sender = ["A", "A", "B"], receiver = ["B", "C", "A"])
            cmap = Dict("A" => "#FF0000", "B" => "#00FF00", "C" => "#0000FF")
            p = plot_chord(
                df;
                source_column = :sender,
                target_column = :receiver,
                cell_type_colormap = cmap,
            )
            @test p isa Plots.Plot
        end

        @testset "chord plot undirected" begin
            df = DataFrame(sender = ["A", "B"], receiver = ["B", "A"])
            p = plot_chord(
                df;
                source_column = :sender,
                target_column = :receiver,
                directed = false,
            )
            @test p isa Plots.Plot
        end

        @testset "chord plot sort none" begin
            df = DataFrame(sender = ["A", "B", "C"], receiver = ["B", "A", "A"])
            p = plot_chord(
                df;
                source_column = :sender,
                target_column = :receiver,
                sort = :none,
            )
            @test p isa Plots.Plot
        end
    end

    @testset "plot_bars" begin
        @testset "string labels" begin
            labels = ["VEGFA-KDR", "CCL", "VEGFA-KDR", "CCL", "CCL", "WNT"]
            p = plot_bars(labels)
            @test p isa Plots.Plot
        end

        @testset "integer labels (program IDs)" begin
            labels = [1, 2, 1, 3, 2, 2, 1, 3]
            p = plot_bars(labels)
            @test p isa Plots.Plot
        end

        @testset "with colormap" begin
            labels = ["CP1", "CP2", "CP1", "CP2", "CP3"]
            cmap = Dict("CP1" => "#FF0000", "CP2" => "#00FF00", "CP3" => "#0000FF")
            p = plot_bars(labels; colormap = cmap)
            @test p isa Plots.Plot
        end

        @testset "with subset_inds" begin
            labels = ["A", "B", "C", "A", "B"]
            p = plot_bars(labels; subset_inds = [1, 2, 4])
            @test p isa Plots.Plot
        end

        @testset "with top_k" begin
            labels = ["A", "A", "A", "B", "B", "C"]
            p = plot_bars(labels; top_k = 2)
            @test p isa Plots.Plot
        end

        @testset "sort_by name" begin
            labels = [3, 1, 2, 1, 3]
            p = plot_bars(labels; sort_by = :name)
            @test p isa Plots.Plot
        end

        @testset "no legend" begin
            labels = ["X", "Y", "X"]
            p = plot_bars(labels; show_legend = false)
            @test p isa Plots.Plot
        end

        @testset "invalid top_k" begin
            @test_throws AssertionError plot_bars(["A"]; top_k = 0)
        end
    end

    @testset "plot_stacked_bars" begin
        @testset "basic stacked bars" begin
            labels = ["CP1", "CP1", "CP2", "CP2", "CP1", "CP2"]
            cell_types = ["A", "B", "A", "A", "B", "B"]
            p = plot_stacked_bars(labels, cell_types)
            @test p isa Plots.Plot
        end

        @testset "integer labels" begin
            labels = [1, 1, 2, 2, 3, 3]
            cell_types = ["Tcell", "Bcell", "Tcell", "Macro", "Bcell", "Macro"]
            p = plot_stacked_bars(labels, cell_types)
            @test p isa Plots.Plot
        end

        @testset "with colormaps" begin
            labels = ["CP1", "CP2", "CP1", "CP2"]
            cell_types = ["A", "B", "A", "B"]
            cmap = Dict("CP1" => "#FF0000", "CP2" => "#0000FF")
            ct_cmap = Dict("A" => "#00FF00", "B" => "#FFFF00")
            p = plot_stacked_bars(
                labels,
                cell_types;
                colormap = cmap,
                cell_type_colormap = ct_cmap,
            )
            @test p isa Plots.Plot
        end

        @testset "with subset_inds" begin
            labels = ["CP1", "CP2", "CP3", "CP1"]
            cell_types = ["A", "B", "A", "B"]
            p = plot_stacked_bars(labels, cell_types; subset_inds = [1, 2, 4])
            @test p isa Plots.Plot
        end

        @testset "annotate counts" begin
            labels = ["CP1", "CP1", "CP2"]
            cell_types = ["A", "B", "A"]
            p = plot_stacked_bars(labels, cell_types; annotate_counts = true)
            @test p isa Plots.Plot
        end

        @testset "mismatched lengths" begin
            @test_throws AssertionError plot_stacked_bars(["A", "B"], ["X"])
        end
    end

    @testset "plot_gini_lines" begin
        @testset "basic line plot" begin
            pcts = [10, 20, 30, 50, 100]
            gini_r = [0.1, 0.2, 0.3, 0.5, 0.8]
            gini_ma = [0.15, 0.25, 0.35, 0.55, 0.85]
            gini_f = [0.05, 0.15, 0.25, 0.45, 0.75]
            p = plot_gini_lines(pcts, gini_r, gini_ma, gini_f)
            @test p isa Plots.Plot
        end

        @testset "custom kwargs" begin
            pcts = [25, 50, 100]
            p = plot_gini_lines(
                pcts,
                [0.3, 0.5, 0.9],
                [0.4, 0.6, 0.95],
                [0.2, 0.4, 0.8];
                title = "Test Gini",
                ylims = (0.0, 1.0),
            )
            @test p isa Plots.Plot
        end

        @testset "mismatched lengths" begin
            @test_throws ArgumentError plot_gini_lines(
                [10, 20],
                [0.1, 0.2],
                [0.3],
                [0.4, 0.5],
            )
        end
    end

    @testset "plot_top_lrpairs" begin
        @testset "basic dot plot" begin
            names = ["VEGFA-KDR", "MDK-NCL", "WNT5A-FZD3"]
            coeffs = [1.0, 0.75, 0.5]
            p = plot_top_lrpairs(names, coeffs)
            @test p isa Plots.Plot
        end

        @testset "custom kwargs" begin
            p = plot_top_lrpairs(
                ["A", "B"],
                [0.9, 0.3];
                title = "Test",
                marker_color = :red,
            )
            @test p isa Plots.Plot
        end

        @testset "mismatched lengths" begin
            @test_throws ArgumentError plot_top_lrpairs(["A", "B"], [0.5])
        end
    end

    @testset "plot_downstream_boxplots" begin
        @testset "basic boxplots" begin
            groups = [rand(20), rand(20), rand(20), rand(20)]
            labels = ["25", "50", "100", "Non receivers"]
            p = plot_downstream_boxplots(groups, labels)
            @test p isa Plots.Plot
        end

        @testset "custom kwargs" begin
            groups = [rand(10), rand(10)]
            p = plot_downstream_boxplots(
                groups,
                ["A", "B"];
                title = "Test",
                box_color = :green,
            )
            @test p isa Plots.Plot
        end

        @testset "mismatched lengths" begin
            @test_throws ArgumentError plot_downstream_boxplots([rand(5)], ["A", "B"])
        end
    end

    @testset "plot_chord_python" begin
        using DataFrames, PythonCall

        has_mpl_chord = try
            pyimport("mpl_chord_diagram")
            true
        catch
            false
        end

        if has_mpl_chord
            df = DataFrame(
                sender_cell_type = ["A", "A", "B", "B", "C"],
                receiver_cell_type = ["B", "C", "A", "C", "A"],
            )
            fig = plot_chord_python(df)
            @test !isnothing(fig)
            plt = pyimport("matplotlib.pyplot")
            plt.close("all")
        else
            @warn "Skipping plot_chord_python test: mpl_chord_diagram not installed"
            @test_broken false  # mark as known skip
        end
    end
end
