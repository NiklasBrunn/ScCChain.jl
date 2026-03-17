# Evaluation-focused plotting functions for communication program analysis.

using StatsPlots

"""
    plot_gini_lines(pcts, gini_receiver, gini_max_attention, gini_first; kwargs...) -> Plot

Line plot of normalized Gini impurity scores across error-filtered chain subsets.

# Arguments
- `pcts::AbstractVector{<:Real}`: percentage of chains retained at each filtering step.
- `gini_receiver::AbstractVector{<:Real}`: Gini scores for receiver cell types.
- `gini_max_attention::AbstractVector{<:Real}`: Gini scores for max-attention sender cell types.
- `gini_first::AbstractVector{<:Real}`: Gini scores for first sender cell types.

## Appearance
- `title::String="Gini score"`: plot title.
- `xlabel::String="Percentage of chains retained"`: x-axis label.
- `ylabel::String="Gini score"`: y-axis label.
- `ylims::Tuple{Real,Real}=(0.0, 1.0)`: y-axis limits.
- `colors::NTuple{3,Symbol}=(:blue, :red, :orange)`: line colors.
- `labels::NTuple{3,String}=("Receiver Cell Types", "Max Attention Cell Types", "First Cell Types")`:
  legend labels.
- `markers::NTuple{3,Symbol}=(:circle, :rect, :diamond)`: marker shapes.
- `markersize::Int=5`: marker size.
- `linewidth::Real=2.0`: line width.
- `legend_position::Symbol=:topright`: legend placement.
- `dpi::Integer=300`: plot resolution.
- `plot_size::Tuple{Int,Int}=(1200, 800)`: figure dimensions.

# Returns
A `Plots.Plot` object.
"""
function plot_gini_lines(
    pcts::AbstractVector{<:Real},
    gini_receiver::AbstractVector{<:Real},
    gini_max_attention::AbstractVector{<:Real},
    gini_first::AbstractVector{<:Real};
    title::String = "Gini score",
    xlabel::String = "Percentage of chains retained",
    ylabel::String = "Gini score",
    ylims::Tuple{Real,Real} = (0.0, 1.0),
    colors::NTuple{3,Symbol} = (:blue, :red, :orange),
    labels::NTuple{3,String} = (
        "Receiver Cell Types",
        "Max Attention Cell Types",
        "First Cell Types",
    ),
    markers::NTuple{3,Symbol} = (:circle, :rect, :diamond),
    markersize::Int = 5,
    linewidth::Real = 2.0,
    legend_position::Symbol = :topright,
    dpi::Integer = 300,
    plot_size::Tuple{Int,Int} = (1200, 800),
)
    n = length(pcts)
    length(gini_receiver) == n &&
    length(gini_max_attention) == n &&
    length(gini_first) == n ||
        throw(ArgumentError("All Gini vectors must have the same length as pcts"))

    p = plot(
        pcts,
        gini_receiver;
        label = labels[1],
        color = colors[1],
        marker = markers[1],
        ms = markersize,
        lw = linewidth,
        legend = legend_position,
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        ylims = ylims,
        size = plot_size,
        dpi = dpi,
        grid = false,
    )
    plot!(
        p,
        pcts,
        gini_max_attention;
        label = labels[2],
        color = colors[2],
        marker = markers[2],
        ms = markersize,
        lw = linewidth,
    )
    plot!(
        p,
        pcts,
        gini_first;
        label = labels[3],
        color = colors[3],
        marker = markers[3],
        ms = markersize,
        lw = linewidth,
    )

    return p
end

"""
    plot_top_lrpairs(names, coefficients; kwargs...) -> Plot

Dot plot of normalized BAE coefficients for the top LR pairs of a communication program.

# Arguments
- `names::AbstractVector{<:AbstractString}`: LR pair names.
- `coefficients::AbstractVector{<:Real}`: normalized coefficient values.

## Appearance
- `title::String="Top selected LR pairs"`: plot title.
- `ylabel::String="Normalized coefficient"`: y-axis label.
- `xlabel::String=""`: x-axis label.
- `xrotation::Int=50`: x-tick rotation angle.
- `marker_color::Symbol=:brown`: marker fill color.
- `marker_size::Int=8`: marker size.
- `dpi::Integer=300`: plot resolution.
- `plot_size::Tuple{Int,Int}=(1000, 600)`: figure dimensions.

# Returns
A `Plots.Plot` object.
"""
function plot_top_lrpairs(
    names::AbstractVector{<:AbstractString},
    coefficients::AbstractVector{<:Real};
    title::String = "Top selected LR pairs",
    ylabel::String = "Normalized coefficient",
    xlabel::String = "",
    xrotation::Int = 50,
    marker_color::Symbol = :brown,
    marker_size::Int = 8,
    dpi::Integer = 300,
    plot_size::Tuple{Int,Int} = (1000, 600),
)
    length(names) == length(coefficients) ||
        throw(ArgumentError("names and coefficients must have the same length"))

    x = 1:length(names)
    p = scatter(
        x,
        coefficients;
        label = false,
        color = marker_color,
        ms = marker_size,
        ylabel = ylabel,
        xlabel = xlabel,
        title = title,
        xticks = (x, names),
        xrotation = xrotation,
        size = plot_size,
        dpi = dpi,
        grid = false,
    )
    hline!(p, [0.0]; linestyle = :dash, color = :grey, label = false)

    return p
end

"""
    plot_downstream_boxplots(score_groups, group_labels; kwargs...) -> Plot

Box plots comparing downstream gene expression scores across chain retention levels.

# Arguments
- `score_groups::AbstractVector{<:AbstractVector{<:Real}}`: one vector of scores per group.
- `group_labels::AbstractVector{<:AbstractString}`: labels for each group.

## Appearance
- `title::String="Downstream score"`: plot title.
- `ylabel::String="Downstream Gene Scores"`: y-axis label.
- `xlabel::String="Percentage of chains retained"`: x-axis label.
- `box_color::Union{Symbol,String}="#3A86FF"`: box fill color for chain groups.
- `non_receiver_color::Union{Symbol,String}="#EAEAEA"`: box fill color for the last group.
- `show_means::Bool=true`: overlay mean markers as red crosses.
- `dpi::Integer=300`: plot resolution.
- `plot_size::Tuple{Int,Int}=(800, 600)`: figure dimensions.

# Returns
A `Plots.Plot` object.
"""
function plot_downstream_boxplots(
    score_groups::AbstractVector{<:AbstractVector{<:Real}},
    group_labels::AbstractVector{<:AbstractString};
    title::String = "Downstream score",
    ylabel::String = "Downstream Gene Scores",
    xlabel::String = "Percentage of chains retained",
    box_color::Union{Symbol,String} = "#3A86FF",
    non_receiver_color::Union{Symbol,String} = "#EAEAEA",
    show_means::Bool = true,
    dpi::Integer = 300,
    plot_size::Tuple{Int,Int} = (800, 600),
)
    length(score_groups) == length(group_labels) ||
        throw(ArgumentError("score_groups and group_labels must have the same length"))

    n = length(score_groups)
    p = plot(;
        legend = false,
        size = plot_size,
        dpi = dpi,
        title = title,
        ylabel = ylabel,
        xlabel = xlabel,
    )

    for i = 1:n
        c = i == n ? non_receiver_color : box_color
        boxplot!(
            p,
            fill(i, length(score_groups[i])),
            score_groups[i];
            label = false,
            color = c,
            mediancolor = :black,
            whiskercolor = :black,
        )
    end

    if show_means
        means = [mean(g) for g in score_groups]
        scatter!(
            p,
            1:n,
            means;
            markershape = :xcross,
            markercolor = :red,
            markerstrokecolor = :red,
            ms = 5,
            msw = 2.5,
            label = nothing,
        )
    end

    xticks!(p, (1:n, group_labels))

    return p
end
