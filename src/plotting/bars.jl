# Bar chart visualization functions.

"""
    plot_bars(labels; kwargs...) -> Plot

Plot a colored bar chart counting occurrences of each unique label.

Works with both communication program IDs (integers like `1, 2, 3`) and
ligand-receptor pair names (strings like `"VEGFA-KDR"`, `"CCL"`).

# Arguments
- `labels::AbstractVector`: one label per chain — program IDs or LR pair names.

## Filtering & sorting
- `subset_inds::Union{AbstractVector{Int}, Nothing}=nothing`: indices to include.
- `top_k::Union{Int, Nothing}=nothing`: keep only the `top_k` most frequent groups.
- `sort_by::Symbol=:count`: `:count` (descending) or `:name` (ascending alpha/numeric).

## Appearance
- `colormap::Union{AbstractDict, Nothing}=nothing`: label → hex color. Falls back to
  `COMMUNICATION_COLORS` palette when `nothing`.
- `title::String="Communication chain counts"`: plot title.
- `ylabel::String="Communication chain counts"`: y-axis label.
- `xlabel::String=""`: x-axis label.
- `xrotation::Int=50`: x-tick rotation angle.
- `show_legend::Bool=true`: display a legend.
- `legend_position::Symbol=:outerright`: legend placement.
- `dpi::Integer=300`: plot resolution.
- `plot_size::Tuple{Int,Int}=(1200, 600)`: figure dimensions.

# Returns
A `Plots.Plot` object.
"""
function plot_bars(
    labels::AbstractVector;
    subset_inds::Union{AbstractVector{Int},Nothing} = nothing,
    colormap::Union{AbstractDict,Nothing} = nothing,
    top_k::Union{Int,Nothing} = nothing,
    sort_by::Symbol = :count,
    title::String = "Communication chain counts",
    ylabel::String = "Communication chain counts",
    xlabel::String = "",
    xrotation::Int = 50,
    show_legend::Bool = true,
    legend_position::Symbol = :outerright,
    dpi::Integer = 300,
    plot_size::Tuple{Int,Int} = (1200, 600),
)
    @assert isnothing(top_k) || top_k > 0 "top_k must be a positive integer"

    # Subset
    labs = isnothing(subset_inds) ? labels : labels[subset_inds]

    # Convert to strings for uniform handling
    str_labels = string.(labs)

    # Count occurrences
    df = DataFrame(group = str_labels)
    counts_df = combine(groupby(df, :group), nrow => :count)

    # Optional top-k filtering (before sorting for display)
    if !isnothing(top_k)
        sort!(counts_df, :count; rev = true)
        counts_df = first(counts_df, min(top_k, nrow(counts_df)))
    end

    # Sort for display
    if sort_by == :count
        _sort_bar_counts!(counts_df)
    elseif sort_by == :name
        _sort_bar_names!(counts_df)
    else
        throw(ArgumentError("sort_by must be :count or :name, got :$sort_by"))
    end

    groups = counts_df.group
    values = counts_df.count

    # Resolve colors
    cmap = _resolve_bar_colormap(groups, colormap)
    bar_colors = [get(cmap, g, "#999999") for g in groups]

    # Bar plot
    x = 1:length(groups)
    p = bar(
        x,
        values;
        fillcolor = bar_colors,
        legend = show_legend ? legend_position : false,
        ylabel = ylabel,
        xlabel = xlabel,
        title = title,
        xticks = (x, groups),
        xrotation = xrotation,
        size = plot_size,
        dpi = dpi,
        label = false,
    )

    # Legend entries
    if show_legend
        _add_bar_legend!(p, groups, cmap)
    end

    return p
end

# ---- Internal helpers ----

"""Sort counts_df descending by count, ties ascending by group name/number."""
function _sort_bar_counts!(counts_df::DataFrame)
    maybe_int = [tryparse(Int, g) for g in counts_df.group]
    if all(!isnothing, maybe_int)
        counts_df.sort_key = something.(maybe_int, 0)
        sort!(counts_df, [:count, :sort_key]; rev = [true, false])
        select!(counts_df, Not(:sort_key))
    else
        sort!(counts_df, [:count, :group]; rev = [true, false])
    end
end

"""Sort counts_df ascending by group name (numeric-aware)."""
function _sort_bar_names!(counts_df::DataFrame)
    maybe_int = [tryparse(Int, g) for g in counts_df.group]
    if all(!isnothing, maybe_int)
        counts_df.sort_key = something.(maybe_int, 0)
        sort!(counts_df, :sort_key)
        select!(counts_df, Not(:sort_key))
    else
        sort!(counts_df, :group)
    end
end

"""Build a string→color Dict, auto-generating from COMMUNICATION_COLORS if needed."""
function _resolve_bar_colormap(
    groups::AbstractVector{String},
    colormap::Union{AbstractDict,Nothing},
)
    if !isnothing(colormap)
        return Dict(string(k) => string(v) for (k, v) in colormap)
    end
    palette = COMMUNICATION_COLORS
    unique_groups = unique(groups)
    return Dict(
        g => palette[mod1(i, length(palette))] for (i, g) in enumerate(unique_groups)
    )
end

"""Add invisible scatter series as legend entries for each group."""
function _add_bar_legend!(p, groups::AbstractVector{String}, cmap::Dict{String,String})
    maybe_int = [tryparse(Int, g) for g in unique(groups)]
    ugroups = unique(groups)
    if all(!isnothing, maybe_int)
        perm = sortperm(something.(maybe_int, 0))
    else
        perm = sortperm(ugroups)
    end
    for idx in perm
        g = ugroups[idx]
        plot!(
            p,
            [NaN],
            [NaN];
            label = g,
            color = get(cmap, g, "#999999"),
            seriestype = :shape,
            markershape = :rect,
            markersize = 8,
        )
    end
end

# ---- Stacked bar plot ----

"""
    plot_stacked_bars(labels, cell_types; kwargs...) -> Plot

Two-panel bar visualization: count bars on top (colored by group), stacked
cell type proportions on bottom.

Works with both communication program IDs and ligand-receptor pair names.

# Arguments
- `labels::AbstractVector`: one label per chain — program IDs or LR pair names.
- `cell_types::AbstractVector{String}`: one cell type per chain (e.g., receiver type).

## Filtering
- `subset_inds::Union{AbstractVector{Int}, Nothing}=nothing`: indices to include.

## Appearance
- `colormap::Union{AbstractDict, Nothing}=nothing`: label → hex color for the count bars.
- `cell_type_colormap::Union{AbstractDict{String,String}, Nothing}=nothing`: cell type → hex color.
- `annotate_counts::Bool=false`: annotate total counts above the top bars.
- `title::String="Cell type proportions per communication program"`: overall title.
- `ylabel_top::String="Counts"`: y-axis label for count panel.
- `ylabel_bottom::String="Proportion"`: y-axis label for proportion panel.
- `xlabel::String=""`: x-axis label.
- `xrotation::Int=50`: x-tick rotation angle.
- `dpi::Integer=300`: plot resolution.
- `plot_size::Tuple{Int,Int}=(1200, 800)`: figure dimensions.

# Returns
A `Plots.Plot` object.
"""
function plot_stacked_bars(
    labels::AbstractVector,
    cell_types::AbstractVector{String};
    subset_inds::Union{AbstractVector{Int},Nothing} = nothing,
    colormap::Union{AbstractDict,Nothing} = nothing,
    cell_type_colormap::Union{AbstractDict{String,String},Nothing} = nothing,
    annotate_counts::Bool = false,
    title::String = "Cell type proportions per communication program",
    ylabel_top::String = "Counts",
    ylabel_bottom::String = "Proportion",
    xlabel::String = "",
    xrotation::Int = 50,
    dpi::Integer = 300,
    plot_size::Tuple{Int,Int} = (1200, 800),
)
    @assert length(labels) == length(cell_types) "labels and cell_types must have the same length"

    # Subset
    str_labels = isnothing(subset_inds) ? string.(labels) : string.(labels[subset_inds])
    ct = isnothing(subset_inds) ? cell_types : cell_types[subset_inds]

    # Count per (group, cell_type)
    df = DataFrame(group = str_labels, cell_type = ct)
    counts = combine(groupby(df, [:group, :cell_type]), nrow => :count)
    totals = combine(groupby(counts, :group), :count => sum => :total)
    counts = leftjoin(counts, totals; on = :group)
    counts.proportion = counts.count ./ counts.total

    # Sort groups descending by total count
    sort!(totals, :total; rev = true)
    ordered_groups = totals.group

    # Resolve colormaps
    group_cmap = _resolve_bar_colormap(ordered_groups, colormap)
    all_cell_types = sort(unique(ct))
    ct_cmap = _resolve_cell_type_colormap(all_cell_types, cell_type_colormap)

    n_groups = length(ordered_groups)
    x = 1:n_groups
    group_to_x = Dict(g => i for (i, g) in enumerate(ordered_groups))

    # ---- Top panel: count bars ----
    top_colors = [get(group_cmap, g, "#999999") for g in ordered_groups]
    top_values = [totals[totals.group .== g, :total][1] for g in ordered_groups]

    top_plot = bar(
        x,
        top_values;
        fillcolor = top_colors,
        legend = false,
        ylabel = ylabel_top,
        title = title,
        xticks = false,
        label = false,
    )

    if annotate_counts
        for (i, v) in enumerate(top_values)
            annotate!(top_plot, i, v * 1.02, text(string(v), 8, :center, :bold))
        end
    end

    # ---- Bottom panel: stacked proportions ----
    bottom_plot = _build_stacked_proportion_plot(
        counts,
        ordered_groups,
        all_cell_types,
        ct_cmap,
        group_to_x;
        ylabel = ylabel_bottom,
        xlabel = xlabel,
        xrotation = xrotation,
    )

    # ---- Legend panels ----
    legend_top = _build_legend_panel(ordered_groups, group_cmap)
    legend_bottom = _build_legend_panel(all_cell_types, ct_cmap)

    # ---- Combine 4-panel layout ----
    p = plot(
        top_plot,
        legend_top,
        bottom_plot,
        legend_bottom;
        layout = @layout([grid(2, 2, heights = [0.3, 0.7], widths = [0.75, 0.25])]),
        size = plot_size,
        dpi = dpi,
    )

    return p
end

# ---- Internal helpers for stacked bars ----

"""Build a cell_type → color Dict, auto-generating from palette if needed."""
function _resolve_cell_type_colormap(
    cell_types::AbstractVector{String},
    colormap::Union{AbstractDict{String,String},Nothing},
)
    if !isnothing(colormap)
        return Dict(string(k) => string(v) for (k, v) in colormap)
    end
    palette = select_tableau_palette(length(cell_types))
    return Dict(
        ct => palette[mod1(i, length(palette))] for (i, ct) in enumerate(cell_types)
    )
end

"""Build the stacked proportion bar plot using manual cumulative offsets."""
function _build_stacked_proportion_plot(
    counts::DataFrame,
    ordered_groups::AbstractVector{String},
    all_cell_types::AbstractVector{String},
    ct_cmap::Dict{String,String},
    group_to_x::Dict{String,Int};
    ylabel = "Proportion",
    xlabel = "",
    xrotation = 50,
)
    n_groups = length(ordered_groups)
    x = 1:n_groups

    # Initialize with empty plot
    p = plot(;
        ylabel = ylabel,
        xlabel = xlabel,
        xticks = (x, ordered_groups),
        xrotation = xrotation,
        ylims = (0, 1.05),
        legend = false,
    )

    # Stack bars by cell type
    for ct in all_cell_types
        heights = zeros(Float64, n_groups)
        for row in eachrow(counts)
            if row.cell_type == ct
                gi = group_to_x[row.group]
                heights[gi] = row.proportion
            end
        end
        bar!(
            p,
            x,
            heights;
            fillcolor = get(ct_cmap, ct, "#999999"),
            bar_position = :stack,
            label = false,
        )
    end

    return p
end

"""Build a legend-only plot panel."""
function _build_legend_panel(labels::AbstractVector{String}, cmap::Dict{String,String})
    p = plot(;
        legend = :left,
        framestyle = :none,
        grid = false,
        ticks = nothing,
        xlabel = "",
        ylabel = "",
        title = "",
    )
    for lab in labels
        plot!(
            p,
            [NaN],
            [NaN];
            label = lab,
            color = get(cmap, lab, "#999999"),
            seriestype = :shape,
            markershape = :rect,
            markersize = 8,
        )
    end
    return p
end
