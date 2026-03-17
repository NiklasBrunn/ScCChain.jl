# Chain overlay visualization functions.

"""
    plot_chains!(cell_locations, cell_annotation, stacked_matrix; kwargs...) -> Plot

Plot communication chains as paths through spatial tissue.

Grey edges show similarity-based connections; colored edges show communication
LR pairs. Enable error weighting by providing `error_vec`; enable per-communication
coloring by providing both `communication_labels` and `communication_colormap`.

# Arguments
- `cell_locations::AbstractMatrix{<:AbstractFloat}`: N×2 spatial coordinates.
- `cell_annotation::Union{Nothing, AbstractVector{String}}`: cell type labels (or nothing).
- `stacked_matrix::AbstractMatrix`: 2-column matrix; col 1 = cell ID paths (`Vector{Int}`),
  col 2 = layer index vectors (0 = similarity, 1 = communication).

## Communication coloring (provide both or neither)
- `communication_labels::Union{Nothing, AbstractVector{String}}=nothing`: one label per chain row.
- `communication_colormap::Union{Nothing, AbstractDict{String,String}}=nothing`: label → hex color.

## Error weighting (provide `error_vec` to activate)
- `error_vec::Union{Nothing, AbstractVector{<:AbstractFloat}}=nothing`: one error per chain.
- `n_weight_bins::Int=10`: number of visual weight bins (≤20 recommended).
- `max_line_width::Real=0.5`: line width for lowest-error chains.
- `min_line_width::Real=0.05`: line width for highest-error chains.
- `alpha_min::Real=0.05`: minimum line transparency.
- `alpha_gamma::Real=1.0`: gamma curve for alpha mapping.
- `line_gamma::Real=1.0`: gamma curve for line width mapping.

## Cell type highlighting
- `cell_type_colormap::Union{Nothing, AbstractDict{String,String}}=nothing`: type → hex color.
- `selected_cell_types::Union{Nothing, AbstractVector{String}}=nothing`: types to highlight
  (others greyed out).

## General display
- `line_width::Real=0.5`: line width when no error weighting.
- `linealpha::Real=1.0`: line alpha when no error weighting.
- `base_plot=nothing`: existing plot (auto-creates scatter if nothing).
- `markersize::Real=DEFAULT_MARKERSIZE`: cell marker size.
- `subsample::Union{Nothing, Int}=nothing`: random chain subsample count.
- `dpi::Integer=300`: plot resolution.
- `plot_size=(1200, 600)`: figure dimensions.

# Returns
A `Plots.Plot` object.
"""
function plot_chains!(
    cell_locations::AbstractMatrix{T},
    cell_annotation::Union{Nothing,AbstractVector{String}},
    stacked_matrix::AbstractMatrix;
    communication_labels::Union{Nothing,AbstractVector{String}} = nothing,
    communication_colormap::Union{Nothing,AbstractDict{String,String}} = nothing,
    error_vec::Union{Nothing,AbstractVector{<:AbstractFloat}} = nothing,
    n_weight_bins::Int = 10,
    max_line_width::Real = 0.5,
    min_line_width::Real = 0.05,
    alpha_min::Real = 0.05,
    alpha_gamma::Real = 1.0,
    line_gamma::Real = 1.0,
    cell_type_colormap::Union{Nothing,AbstractDict{String,String}} = nothing,
    selected_cell_types::Union{Nothing,AbstractVector{String}} = nothing,
    line_width::Real = 0.5,
    linealpha::Real = 1.0,
    base_plot = nothing,
    markersize::Real = DEFAULT_MARKERSIZE,
    subsample::Union{Nothing,Int} = nothing,
    dpi::Integer = 300,
    plot_size = (1200, 600),
) where {T<:AbstractFloat}

    # Validate communication coloring arguments
    if !isnothing(communication_labels) ⊻ !isnothing(communication_colormap)
        throw(
            ArgumentError(
                "communication_colormap must be provided if communication_labels is provided and vice versa",
            ),
        )
    end
    if !isnothing(communication_labels)
        @assert length(communication_labels) == size(stacked_matrix, 1) "communication_labels must equal number of rows in stacked_matrix"
    end

    # Build base plot
    base_plot = _build_base_plot(
        base_plot,
        cell_locations,
        cell_annotation;
        cell_type_colormap = cell_type_colormap,
        selected_cell_types = selected_cell_types,
        markersize = markersize,
        dpi = dpi,
        plot_size = plot_size,
    )
    dark_grey = RGB(0.2, 0.2, 0.2)

    # Select rows (with optional subsample)
    rows = collect(1:size(stacked_matrix, 1))
    if subsample !== nothing
        shuffle!(rows)
        resize!(rows, min(subsample, length(rows)))
    end

    # Dispatch to error-weighted or flat rendering
    if isnothing(error_vec)
        _render_chains_flat!(
            base_plot,
            cell_locations,
            stacked_matrix,
            rows,
            dark_grey,
            T;
            communication_labels = communication_labels,
            communication_colormap = communication_colormap,
            line_width = line_width,
            linealpha = linealpha,
        )
    else
        @assert length(error_vec) >= maximum(rows; init = 0) "error_vec length must cover stacked_matrix rows"
        if n_weight_bins > 20
            @warn "n_weight_bins > 20 can slow down plotting; consider a smaller value."
        end
        _render_chains_error_weighted!(
            base_plot,
            cell_locations,
            stacked_matrix,
            rows,
            error_vec,
            dark_grey,
            T;
            communication_labels = communication_labels,
            communication_colormap = communication_colormap,
            n_weight_bins = n_weight_bins,
            max_line_width = max_line_width,
            min_line_width = min_line_width,
            alpha_min = alpha_min,
            alpha_gamma = alpha_gamma,
            line_gamma = line_gamma,
        )
    end

    return base_plot
end

# ---- Internal: build/reuse the base tissue scatter plot ----

function _build_base_plot(
    base_plot,
    cell_locations::AbstractMatrix{<:AbstractFloat},
    cell_annotation::Union{Nothing,AbstractVector{String}};
    cell_type_colormap = nothing,
    selected_cell_types = nothing,
    markersize = DEFAULT_MARKERSIZE,
    dpi = 300,
    plot_size = (1200, 600),
)
    if !isnothing(base_plot)
        return base_plot
    end

    # Apply cell type highlighting if requested
    color_map_selected =
        _apply_selected_cell_types(cell_annotation, cell_type_colormap, selected_cell_types)

    if isnothing(cell_annotation)
        return plot_spatial(
            cell_locations;
            markersize = markersize,
            dpi = dpi,
            plot_size = plot_size,
        )
    else
        if isnothing(color_map_selected)
            return plot_spatial(
                cell_locations,
                cell_annotation;
                markersize = markersize,
                dpi = dpi,
                plot_size = plot_size,
            )
        else
            ks_sorted = sort(collect(keys(color_map_selected)))
            vals_sorted = getindex.(Ref(color_map_selected), ks_sorted)
            return plot_spatial(
                cell_locations,
                cell_annotation;
                custompalette = vals_sorted,
                markersize = markersize,
                dpi = dpi,
                plot_size = plot_size,
            )
        end
    end
end

function _apply_selected_cell_types(
    cell_annotation,
    cell_type_colormap,
    selected_cell_types,
)
    if isnothing(selected_cell_types)
        return cell_type_colormap
    end
    isnothing(cell_type_colormap) && throw(
        ArgumentError(
            "cell_type_colormap must be provided if selected_cell_types are provided",
        ),
    )
    if !isnothing(cell_annotation)
        cats = unique(cell_annotation)
        @assert all(ct -> ct in cats, selected_cell_types) "selected_cell_types must be a subset of cell_annotation"
    end
    return Dict{String,String}(
        k => (k in selected_cell_types ? v : "#D3D3D3") for (k, v) in cell_type_colormap
    )
end

# ---- Internal: flat rendering (no error weighting) ----

function _render_chains_flat!(
    base_plot,
    cell_locations::AbstractMatrix{T},
    stacked_matrix,
    rows,
    dark_grey,
    ::Type{T};
    communication_labels = nothing,
    communication_colormap = nothing,
    line_width = 0.5,
    linealpha = 1.0,
) where {T}
    use_communication_colors =
        !isnothing(communication_labels) && !isnothing(communication_colormap)
    nanT = T(NaN)

    if use_communication_colors
        # Per-communication-color rendering
        # First pass: count segments
        total_grey = 0
        color_counts = Dict{Any,Int}()
        @inbounds for i in rows
            path = stacked_matrix[i, 1]
            layers_path = stacked_matrix[i, 2]
            comm = findfirst(==(1), layers_path)
            if comm === nothing || length(path) < 2
                continue
            end
            total_grey += max(comm - 1, 0)
            if comm <= length(path) - 1
                c = get(communication_colormap, communication_labels[i], "red")
                color_counts[c] = get(color_counts, c, 0) + 1
            end
        end

        # Allocate grey buffer
        xg = Vector{T}(undef, 3 * total_grey)
        yg = similar(xg)
        gi = 1

        # Allocate per-color buffers
        x_by_color = Dict{Any,Vector{T}}()
        y_by_color = Dict{Any,Vector{T}}()
        idx_by_color = Dict{Any,Int}()
        for (c, cnt) in color_counts
            x_by_color[c] = Vector{T}(undef, 3 * cnt)
            y_by_color[c] = Vector{T}(undef, 3 * cnt)
            idx_by_color[c] = 1
        end

        # Second pass: fill buffers
        @inbounds @views for i in rows
            path = stacked_matrix[i, 1]
            layers_path = stacked_matrix[i, 2]
            comm = findfirst(==(1), layers_path)
            if comm === nothing || length(path) < 2
                continue
            end
            coords = view(cell_locations, path, :)
            for j = 1:(comm-1)
                xg[gi] = coords[j, 1];
                yg[gi] = coords[j, 2];
                gi += 1
                xg[gi] = coords[j+1, 1];
                yg[gi] = coords[j+1, 2];
                gi += 1
                xg[gi] = nanT;
                yg[gi] = nanT;
                gi += 1
            end
            if comm <= size(coords, 1) - 1
                c = get(communication_colormap, communication_labels[i], "red")
                xr = x_by_color[c];
                yr = y_by_color[c];
                ri = idx_by_color[c]
                xr[ri] = coords[comm, 1];
                yr[ri] = coords[comm, 2];
                ri += 1
                xr[ri] = coords[comm+1, 1];
                yr[ri] = coords[comm+1, 2];
                ri += 1
                xr[ri] = nanT;
                yr[ri] = nanT;
                ri += 1
                idx_by_color[c] = ri
            end
        end

        resize!(xg, gi - 1);
        resize!(yg, gi - 1)
        for (c, ri) in idx_by_color
            resize!(x_by_color[c], ri - 1);
            resize!(y_by_color[c], ri - 1)
        end

        # Plot
        plot!(
            base_plot,
            xg,
            yg;
            color = dark_grey,
            line = (:solid, line_width),
            linealpha = linealpha,
            label = false,
        )
        for (c, xv) in x_by_color
            if !isempty(xv)
                plot!(
                    base_plot,
                    xv,
                    y_by_color[c];
                    color = c,
                    line = (:solid, line_width),
                    linealpha = linealpha,
                    label = false,
                )
            end
        end
    else
        # Single-color rendering (red for communication)
        total_grey = 0
        total_red = 0
        @inbounds for i in rows
            path = stacked_matrix[i, 1]
            layers_path = stacked_matrix[i, 2]
            comm = findfirst(==(1), layers_path)
            if comm === nothing || length(path) < 2
                continue
            end
            total_grey += max(comm - 1, 0)
            total_red += (comm <= length(path) - 1) ? 1 : 0
        end

        xg = Vector{T}(undef, 3 * total_grey);
        yg = similar(xg)
        xr = Vector{T}(undef, 3 * total_red);
        yr = similar(xr)
        gi = 1;
        ri = 1

        @inbounds @views for i in rows
            path = stacked_matrix[i, 1]
            layers_path = stacked_matrix[i, 2]
            comm = findfirst(==(1), layers_path)
            if comm === nothing || length(path) < 2
                continue
            end
            coords = view(cell_locations, path, :)
            for j = 1:(comm-1)
                xg[gi] = coords[j, 1];
                yg[gi] = coords[j, 2];
                gi += 1
                xg[gi] = coords[j+1, 1];
                yg[gi] = coords[j+1, 2];
                gi += 1
                xg[gi] = nanT;
                yg[gi] = nanT;
                gi += 1
            end
            if comm <= size(coords, 1) - 1
                xr[ri] = coords[comm, 1];
                yr[ri] = coords[comm, 2];
                ri += 1
                xr[ri] = coords[comm+1, 1];
                yr[ri] = coords[comm+1, 2];
                ri += 1
                xr[ri] = nanT;
                yr[ri] = nanT;
                ri += 1
            end
        end

        resize!(xg, gi - 1);
        resize!(yg, gi - 1)
        resize!(xr, ri - 1);
        resize!(yr, ri - 1)

        plot!(
            base_plot,
            xg,
            yg;
            color = dark_grey,
            line = (:solid, line_width),
            linealpha = linealpha,
            label = false,
        )
        if !isempty(xr)
            plot!(
                base_plot,
                xr,
                yr;
                color = :red,
                line = (:solid, line_width),
                linealpha = linealpha,
                label = false,
            )
        end
    end
end

# ---- Internal: error-weighted rendering ----

function _render_chains_error_weighted!(
    base_plot,
    cell_locations::AbstractMatrix{T},
    stacked_matrix,
    rows,
    error_vec,
    dark_grey,
    ::Type{T};
    communication_labels = nothing,
    communication_colormap = nothing,
    n_weight_bins = 10,
    max_line_width = 0.5,
    min_line_width = 0.05,
    alpha_min = 0.05,
    alpha_gamma = 1.0,
    line_gamma = 1.0,
) where {T}
    use_communication_colors =
        !isnothing(communication_labels) && !isnothing(communication_colormap)
    nbins = max(1, n_weight_bins)
    nanT = T(NaN)

    # Compute per-row inverse-normalized weights and bin assignments
    err = T.(error_vec[rows])
    emax = maximum(err);
    emin = minimum(err);
    rng = emax - emin
    w = rng == 0 ? ones(T, length(err)) : 1 .- (err .- emin) ./ rng
    bin_of(k) = max(1, min(nbins, Int(ceil(w[k] * nbins))))

    # Bin-to-visual mapping helpers
    unit_ramp(b) = nbins > 1 ? (b - 1) / (nbins - 1) : 0.5
    wmin = max(eps(T), T(min_line_width))
    amin = max(eps(T), T(alpha_min))
    width_of_bin(b) = wmin + (max_line_width - wmin) * (unit_ramp(b))^line_gamma
    linealpha_of_bin(b) =
        clamp(amin + (one(T) - amin) * (unit_ramp(b))^alpha_gamma, amin, one(T))

    if use_communication_colors
        # Count segments by (color, bin)
        grey_count_by_bin = zeros(Int, nbins)
        color_count_by_bin = Dict{Tuple{Any,Int},Int}()

        @inbounds for (k, i) in enumerate(rows)
            path = stacked_matrix[i, 1]
            layers_path = stacked_matrix[i, 2]
            comm = findfirst(==(1), layers_path)
            if comm === nothing || length(path) < 2
                continue
            end
            b = bin_of(k)
            grey_count_by_bin[b] += max(comm - 1, 0)
            if comm <= length(path) - 1
                c = get(communication_colormap, communication_labels[i], "red")
                key = (c, b)
                color_count_by_bin[key] = get(color_count_by_bin, key, 0) + 1
            end
        end

        # Allocate per-bin grey buffers
        xg = [Vector{T}(undef, 3 * grey_count_by_bin[b]) for b = 1:nbins]
        yg = [Vector{T}(undef, 3 * grey_count_by_bin[b]) for b = 1:nbins]
        gi = ones(Int, nbins)

        # Allocate per-(color, bin) buffers
        x_by_cb = Dict{Tuple{Any,Int},Vector{T}}()
        y_by_cb = Dict{Tuple{Any,Int},Vector{T}}()
        idx_by_cb = Dict{Tuple{Any,Int},Int}()
        for (key, cnt) in color_count_by_bin
            x_by_cb[key] = Vector{T}(undef, 3 * cnt)
            y_by_cb[key] = Vector{T}(undef, 3 * cnt)
            idx_by_cb[key] = 1
        end

        # Fill buffers
        @inbounds @views for (k, i) in enumerate(rows)
            path = stacked_matrix[i, 1]
            layers_path = stacked_matrix[i, 2]
            comm = findfirst(==(1), layers_path)
            if comm === nothing || length(path) < 2
                continue
            end
            b = bin_of(k)
            coords = view(cell_locations, path, :)
            for j = 1:(comm-1)
                xg[b][gi[b]] = coords[j, 1];
                yg[b][gi[b]] = coords[j, 2];
                gi[b] += 1
                xg[b][gi[b]] = coords[j+1, 1];
                yg[b][gi[b]] = coords[j+1, 2];
                gi[b] += 1
                xg[b][gi[b]] = nanT;
                yg[b][gi[b]] = nanT;
                gi[b] += 1
            end
            if comm <= size(coords, 1) - 1
                c = get(communication_colormap, communication_labels[i], "red")
                key = (c, b)
                xr = x_by_cb[key];
                yr = y_by_cb[key];
                ri = idx_by_cb[key]
                xr[ri] = coords[comm, 1];
                yr[ri] = coords[comm, 2];
                ri += 1
                xr[ri] = coords[comm+1, 1];
                yr[ri] = coords[comm+1, 2];
                ri += 1
                xr[ri] = nanT;
                yr[ri] = nanT;
                ri += 1
                idx_by_cb[key] = ri
            end
        end

        # Trim & render
        for b = 1:nbins
            resize!(xg[b], gi[b] - 1);
            resize!(yg[b], gi[b] - 1)
        end
        for (key, ri) in idx_by_cb
            resize!(x_by_cb[key], ri - 1);
            resize!(y_by_cb[key], ri - 1)
        end

        for b = 1:nbins
            if !isempty(xg[b])
                plot!(
                    base_plot,
                    xg[b],
                    yg[b];
                    color = dark_grey,
                    line = (:solid, width_of_bin(b)),
                    linealpha = linealpha_of_bin(b),
                    label = false,
                )
            end
        end
        for (key, xv) in x_by_cb
            c, b = key
            if !isempty(xv)
                plot!(
                    base_plot,
                    xv,
                    y_by_cb[key];
                    color = c,
                    line = (:solid, width_of_bin(b)),
                    linealpha = linealpha_of_bin(b),
                    label = false,
                )
            end
        end
    else
        # Single-color (red) with error weighting
        grey_count = zeros(Int, nbins)
        red_count = zeros(Int, nbins)

        @inbounds for (k, i) in enumerate(rows)
            path = stacked_matrix[i, 1]
            layers_path = stacked_matrix[i, 2]
            comm = findfirst(==(1), layers_path)
            if comm === nothing || length(path) < 2
                continue
            end
            b = bin_of(k)
            grey_count[b] += max(comm - 1, 0)
            red_count[b] += (comm <= length(path) - 1) ? 1 : 0
        end

        xg = [Vector{T}(undef, 3 * grey_count[b]) for b = 1:nbins]
        yg = [Vector{T}(undef, 3 * grey_count[b]) for b = 1:nbins]
        xr = [Vector{T}(undef, 3 * red_count[b]) for b = 1:nbins]
        yr = [Vector{T}(undef, 3 * red_count[b]) for b = 1:nbins]
        gi = ones(Int, nbins)
        ri = ones(Int, nbins)

        @inbounds @views for (k, i) in enumerate(rows)
            path = stacked_matrix[i, 1]
            layers_path = stacked_matrix[i, 2]
            comm = findfirst(==(1), layers_path)
            if comm === nothing || length(path) < 2
                continue
            end
            b = bin_of(k)
            coords = view(cell_locations, path, :)
            for j = 1:(comm-1)
                xg[b][gi[b]] = coords[j, 1];
                yg[b][gi[b]] = coords[j, 2];
                gi[b] += 1
                xg[b][gi[b]] = coords[j+1, 1];
                yg[b][gi[b]] = coords[j+1, 2];
                gi[b] += 1
                xg[b][gi[b]] = nanT;
                yg[b][gi[b]] = nanT;
                gi[b] += 1
            end
            if comm <= size(coords, 1) - 1
                xr[b][ri[b]] = coords[comm, 1];
                yr[b][ri[b]] = coords[comm, 2];
                ri[b] += 1
                xr[b][ri[b]] = coords[comm+1, 1];
                yr[b][ri[b]] = coords[comm+1, 2];
                ri[b] += 1
                xr[b][ri[b]] = nanT;
                yr[b][ri[b]] = nanT;
                ri[b] += 1
            end
        end

        for b = 1:nbins
            resize!(xg[b], gi[b] - 1);
            resize!(yg[b], gi[b] - 1)
            resize!(xr[b], ri[b] - 1);
            resize!(yr[b], ri[b] - 1)
        end

        for b = 1:nbins
            if !isempty(xg[b])
                plot!(
                    base_plot,
                    xg[b],
                    yg[b];
                    color = dark_grey,
                    line = (:solid, width_of_bin(b)),
                    linealpha = linealpha_of_bin(b),
                    label = false,
                )
            end
        end
        for b = 1:nbins
            if !isempty(xr[b])
                plot!(
                    base_plot,
                    xr[b],
                    yr[b];
                    color = :red,
                    line = (:solid, width_of_bin(b)),
                    linealpha = linealpha_of_bin(b),
                    label = false,
                )
            end
        end
    end
end
