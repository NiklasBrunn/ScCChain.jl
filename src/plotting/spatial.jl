# Spatial tissue scatter plot functions.

"""
    plot_spatial(cell_locations, [cell_annotation]; kwargs...) -> Plot

Plot spatial transcriptomics data as a tissue scatter plot.

# Arguments
- `cell_locations::AbstractMatrix{<:AbstractFloat}`: N×2 matrix of (x, y) coordinates.
- `cell_annotation`: optional vector of labels (categorical) or numeric values (continuous).
- `data_type::Symbol=:categorical`: `:categorical` or `:continuous`.
- `color_grad::Symbol=:Blues`: color gradient for continuous data.
- `custompalette=nothing`: custom color vector overriding automatic palette selection.
- `legend::Bool=true`: show legend.
- `percentage::Bool=false`: clamp continuous colorbar to 0–100.
- `markersize::Real=DEFAULT_MARKERSIZE`: marker size.
- `alpha::Real=1.0`: marker transparency.
- `grid::Bool=true`: show grid lines.
- `dpi::Integer=300`: plot resolution.
- `plot_size=(600, 600)`: figure dimensions.

# Returns
A `Plots.Plot` object.
"""
function plot_spatial(
    cell_locations::AbstractMatrix{<:AbstractFloat},
    cell_annotation = nothing;
    data_type::Symbol = :categorical,
    color_grad::Symbol = :Blues,
    legend::Bool = true,
    percentage::Bool = false,
    markersize::Real = DEFAULT_MARKERSIZE,
    custompalette = nothing,
    grid::Bool = true,
    dpi::Integer = 300,
    alpha::Real = 1.0,
    plot_size = (600, 600),
)
    common = (
        markerstrokewidth = 0,
        foreground_color_border = :darkgrey,
        gridlinewidth = 0.8,
        foreground_color_axis = :lightgrey,
    )

    if isnothing(cell_annotation)
        return scatter(
            cell_locations[:, 1],
            cell_locations[:, 2];
            color = :grey95,
            legend = false,
            markersize = markersize,
            alpha = alpha,
            size = plot_size,
            grid = grid,
            dpi = dpi,
            common...,
        )
    elseif data_type == :categorical
        ncelltypes = length(unique(cell_annotation))
        tableau =
            isnothing(custompalette) ? select_tableau_palette(ncelltypes) : custompalette
        cell_cat = categorical(cell_annotation)
        palette = tableau[1:length(unique(cell_cat))]
        return scatter(
            cell_locations[:, 1],
            cell_locations[:, 2];
            group = cell_cat,
            palette = palette,
            legend = (legend ? :outertopright : false),
            markersize = markersize,
            alpha = alpha,
            size = plot_size,
            grid = grid,
            dpi = dpi,
            common...,
        )
    elseif data_type == :continuous
        clim_kw = percentage ? (clim = (0, 100),) : NamedTuple()
        return scatter(
            cell_locations[:, 1],
            cell_locations[:, 2];
            color = color_grad,
            zcolor = cell_annotation,
            legend = (legend ? true : false),
            label = "",
            markersize = markersize,
            alpha = alpha,
            size = plot_size,
            grid = grid,
            dpi = dpi,
            clim_kw...,
            common...,
        )
    else
        throw(
            ArgumentError("data_type must be :categorical or :continuous, got :$data_type"),
        )
    end
end

"""
    plot_spatial!(base_plot, cell_locations, [cell_annotation]; kwargs...) -> Plot

Overlay spatial data onto an existing plot.

# Arguments
- `base_plot`: existing `Plots.Plot` to draw on.
- `cell_locations::AbstractMatrix{<:AbstractFloat}`: N×2 coordinates.
- `cell_annotation`: optional labels or numeric values.
- `color_grad::Symbol=:Blues`: gradient for continuous overlay.
- `legend::Bool=false`: show legend.
- `color_map::Dict=Dict()`: explicit type → color string mapping.
- `alpha_variable=nothing`: per-cell transparency vector (auto-normalized to [0,1]).
- `markersize::Real=DEFAULT_MARKERSIZE`: marker size.
- `base_type::Symbol=:discrete`: `:discrete` or `:continuous`.
- `unique_color=:lightgrey`: color when no annotation is given.
- `marker::Symbol=:circle`: marker shape.
- `colorbar::Bool=true`: show colorbar for continuous overlay.
- `custompalette=nothing`: custom color vector for discrete overlay.
- `alpha::Real=1.0`: global transparency.
- `plot_size=(800, 500)`: figure dimensions.

# Returns
The modified `Plots.Plot` object.
"""
function plot_spatial!(
    base_plot,
    cell_locations::AbstractMatrix{<:AbstractFloat},
    cell_annotation = nothing;
    color_grad::Symbol = :Blues,
    legend::Bool = false,
    color_map::Dict = Dict(),
    alpha_variable = nothing,
    markersize::Real = DEFAULT_MARKERSIZE,
    base_type::Symbol = :discrete,
    unique_color = :lightgrey,
    marker::Symbol = :circle,
    colorbar::Bool = true,
    custompalette = nothing,
    alpha::Real = 1.0,
    plot_size = (800, 500),
)
    common = (
        markerstrokewidth = 0,
        foreground_color_border = :darkgrey,
        gridlinewidth = 0.8,
        foreground_color_axis = :lightgrey,
    )

    if isnothing(cell_annotation)
        scatter!(
            base_plot,
            cell_locations[:, 1],
            cell_locations[:, 2];
            color = unique_color,
            legend = false,
            label = "",
            alpha = alpha,
            marker = marker,
            markersize = markersize,
            size = plot_size,
            common...,
        )
    elseif base_type == :continuous
        scatter!(
            base_plot,
            cell_locations[:, 1],
            cell_locations[:, 2];
            color = color_grad,
            zcolor = cell_annotation,
            legend = (legend ? true : false),
            alpha = alpha,
            marker = marker,
            markersize = markersize,
            colorbar = colorbar,
            size = plot_size,
            common...,
        )
    else
        # Discrete overlay with per-cell coloring
        if isempty(color_map)
            unique_cell_types = unique(cell_annotation)
            ncelltypes = length(unique_cell_types)
            tableau =
                isnothing(custompalette) ? select_tableau_palette(ncelltypes) :
                custompalette
            palette = tableau[1:ncelltypes]
            color_lookup = Dict(
                unique_cell_types[i] => palette[i] for i in eachindex(unique_cell_types)
            )
        else
            color_lookup = color_map
        end

        colors = [parse(Colorant, string(color_lookup[type])) for type in cell_annotation]

        if !isnothing(alpha_variable)
            length(alpha_variable) == length(cell_annotation) || throw(
                ArgumentError("alpha_variable must be the same length as cell_annotation"),
            )
            min_a = minimum(alpha_variable)
            max_a = maximum(alpha_variable)
            normalized_alpha =
                max_a != min_a ? (alpha_variable .- min_a) ./ (max_a - min_a) :
                ones(length(alpha_variable))
            scatter!(
                base_plot,
                cell_locations[:, 1],
                cell_locations[:, 2];
                color = colors,
                markeralpha = normalized_alpha,
                legend = (legend ? :outertopright : false),
                alpha = alpha,
                marker = marker,
                markersize = markersize,
                size = plot_size,
                common...,
            )
        else
            scatter!(
                base_plot,
                cell_locations[:, 1],
                cell_locations[:, 2];
                color = colors,
                legend = (legend ? :outertopright : false),
                alpha = alpha,
                marker = marker,
                markersize = markersize,
                size = plot_size,
                common...,
            )
        end
    end

    return base_plot
end
