# Cell pair link visualization functions.

"""
    plot_cell_pairs!(cell_locations, cell_annotation, chains; kwargs...) -> Plot

Plot sender → receiver cell pair links through spatial tissue.

# Arguments
- `cell_locations::AbstractMatrix{<:AbstractFloat}`: N×2 spatial coordinates.
- `cell_annotation::Union{Nothing, AbstractVector{String}}`: cell type labels (or nothing).
- `chains::AbstractVector{<:AbstractVector{<:Integer}}`: cell ID paths per chain.

## Link mode
- `link_from::Symbol=:first`: `:first` links first→last cell; `:max_attention` links
  first→most-attended sender (requires `A`).
- `A::Union{Nothing, AbstractArray{<:Real,3}}=nothing`: attention weight array
  (n_heads × n_senders × n_chains). Required when `link_from=:max_attention`.

## Communication coloring (provide both or neither)
- `communication_labels::Union{Nothing, AbstractVector{String}}=nothing`: one label per chain.
- `communication_colormap::Union{Nothing, AbstractDict{String,String}}=nothing`: label → hex color.

## Error weighting (provide `error_vec` to activate)
- `error_vec::Union{Nothing, AbstractVector{<:AbstractFloat}}=nothing`: one error per chain.
- `n_weight_bins::Int=10`: number of visual weight bins.
- `max_line_width::Real=0.5`, `min_line_width::Real=0.05`: line width range.
- `alpha_min::Real=0.05`, `alpha_gamma::Real=1.0`, `line_gamma::Real=1.0`: visual curves.

## Cell type highlighting & display
- `cell_type_colormap`, `selected_cell_types`: same as `plot_chains!`.
- `line_width::Real=0.5`, `linealpha::Real=1.0`: defaults when no error weighting.
- `base_plot=nothing`, `markersize`, `subsample`, `dpi`, `plot_size`.

# Returns
A `Plots.Plot` object.
"""
function plot_cell_pairs!(
    cell_locations::AbstractMatrix{T},
    cell_annotation::Union{Nothing,AbstractVector{String}},
    chains::AbstractVector{<:AbstractVector{<:Integer}};
    link_from::Symbol = :first,
    A::Union{Nothing,AbstractArray{<:Real,3}} = nothing,
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

    @assert link_from in (:first, :max_attention) "link_from must be :first or :max_attention"
    @assert !isnothing(A) || link_from == :first "A must be provided if link_from=:max_attention"

    total_paths = length(chains)
    if !isnothing(communication_labels)
        @assert length(communication_labels) == total_paths "communication_labels must equal number of chains"
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

    default_color = "red"

    # Choose paths (optional subsample)
    idxs = collect(1:total_paths)
    if subsample !== nothing
        shuffle!(idxs)
        resize!(idxs, min(subsample, total_paths))
    end

    # Compute max-attention sender cell IDs if needed
    most_attended_ids = Vector{Int}(undef, total_paths)
    if link_from == :max_attention
        A_mean = reshape(mean(A; dims = 1), size(A, 2), size(A, 3))
        max_positions = mapslices(argmax, A_mean; dims = 1)[:]
        @inbounds for i = 1:total_paths
            p = chains[i]
            if isempty(p)
                most_attended_ids[i] = 0
                continue
            end
            k = max_positions[i]
            most_attended_ids[i] = (1 <= k <= length(p)) ? p[k] : last(p)
        end
    end

    nrows = size(cell_locations, 1)
    nanT = T(NaN)

    # Helper to get target cell for a given chain
    _get_target(i) = link_from == :first ? last(chains[i]) : most_attended_ids[i]

    if isnothing(error_vec)
        # Flat rendering
        color_counts = Dict{Any,Int}()
        @inbounds for i in idxs
            p = chains[i]
            length(p) < 2 && continue
            a = first(p);
            b = _get_target(i)
            (1 <= a <= nrows && 1 <= b <= nrows) || continue
            a == b && continue
            lbl = isnothing(communication_labels) ? nothing : communication_labels[i]
            c =
                isnothing(lbl) ? default_color :
                get(communication_colormap, lbl, default_color)
            color_counts[c] = get(color_counts, c, 0) + 1
        end

        x_by_color = Dict{Any,Vector{T}}()
        y_by_color = Dict{Any,Vector{T}}()
        idx_by_color = Dict{Any,Int}()
        for (c, cnt) in color_counts
            x_by_color[c] = Vector{T}(undef, 3 * cnt)
            y_by_color[c] = Vector{T}(undef, 3 * cnt)
            idx_by_color[c] = 1
        end

        @inbounds for i in idxs
            p = chains[i]
            length(p) < 2 && continue
            a = first(p);
            b = _get_target(i)
            (1 <= a <= nrows && 1 <= b <= nrows) || continue
            a == b && continue
            lbl = isnothing(communication_labels) ? nothing : communication_labels[i]
            c =
                isnothing(lbl) ? default_color :
                get(communication_colormap, lbl, default_color)
            xv = x_by_color[c];
            yv = y_by_color[c];
            ri = idx_by_color[c]
            xv[ri] = cell_locations[a, 1];
            yv[ri] = cell_locations[a, 2];
            ri += 1
            xv[ri] = cell_locations[b, 1];
            yv[ri] = cell_locations[b, 2];
            ri += 1
            xv[ri] = nanT;
            yv[ri] = nanT;
            ri += 1
            idx_by_color[c] = ri
        end

        for (c, ri) in idx_by_color
            resize!(x_by_color[c], ri - 1);
            resize!(y_by_color[c], ri - 1)
            if !isempty(x_by_color[c])
                plot!(
                    base_plot,
                    x_by_color[c],
                    y_by_color[c];
                    color = c,
                    line = (:solid, line_width),
                    linealpha = linealpha,
                    label = false,
                )
            end
        end
    else
        # Error-weighted rendering
        @assert length(error_vec) == total_paths "error_vec must equal number of chains"
        if n_weight_bins > 20
            @warn "n_weight_bins > 20 can slow down plotting."
        end
        nbins = max(1, n_weight_bins)

        err_used = T.(error_vec[idxs])
        emin = minimum(err_used);
        emax = maximum(err_used);
        rng = emax - emin
        w_used = rng == 0 ? ones(T, length(err_used)) : 1 .- (err_used .- emin) ./ rng

        bin_for_path = fill(0, total_paths)
        bin_of_weight(w) = max(1, min(nbins, Int(ceil(w * nbins))))
        @inbounds for (j, i) in enumerate(idxs)
            bin_for_path[i] = bin_of_weight(w_used[j])
        end

        # Count segments by (color, bin)
        color_count_by_bin = Dict{Tuple{Any,Int},Int}()
        @inbounds for i in idxs
            p = chains[i]
            length(p) < 2 && continue
            a = first(p);
            b = _get_target(i)
            (1 <= a <= nrows && 1 <= b <= nrows) || continue
            a == b && continue
            lbl = isnothing(communication_labels) ? nothing : communication_labels[i]
            c =
                isnothing(lbl) ? default_color :
                get(communication_colormap, lbl, default_color)
            key = (c, bin_for_path[i])
            color_count_by_bin[key] = get(color_count_by_bin, key, 0) + 1
        end

        x_by_cb = Dict{Tuple{Any,Int},Vector{T}}()
        y_by_cb = Dict{Tuple{Any,Int},Vector{T}}()
        idx_by_cb = Dict{Tuple{Any,Int},Int}()
        for (key, cnt) in color_count_by_bin
            x_by_cb[key] = Vector{T}(undef, 3 * cnt)
            y_by_cb[key] = Vector{T}(undef, 3 * cnt)
            idx_by_cb[key] = 1
        end

        @inbounds for i in idxs
            p = chains[i]
            length(p) < 2 && continue
            a = first(p);
            b = _get_target(i)
            (1 <= a <= nrows && 1 <= b <= nrows) || continue
            a == b && continue
            lbl = isnothing(communication_labels) ? nothing : communication_labels[i]
            c =
                isnothing(lbl) ? default_color :
                get(communication_colormap, lbl, default_color)
            key = (c, bin_for_path[i])
            xv = x_by_cb[key];
            yv = y_by_cb[key];
            ri = idx_by_cb[key]
            xv[ri] = cell_locations[a, 1];
            yv[ri] = cell_locations[a, 2];
            ri += 1
            xv[ri] = cell_locations[b, 1];
            yv[ri] = cell_locations[b, 2];
            ri += 1
            xv[ri] = nanT;
            yv[ri] = nanT;
            ri += 1
            idx_by_cb[key] = ri
        end

        unit_ramp(b) = nbins > 1 ? (b - 1) / (nbins - 1) : 0.5
        wmin = max(eps(T), T(min_line_width))
        amin = max(eps(T), T(alpha_min))
        width_of_bin(b) = wmin + (max_line_width - wmin) * (unit_ramp(b))^line_gamma
        linealpha_of_bin(b) =
            clamp(amin + (one(T) - amin) * (unit_ramp(b))^alpha_gamma, amin, one(T))

        for (key, xv) in x_by_cb
            c, bbin = key
            ri = idx_by_cb[key]
            resize!(x_by_cb[key], ri - 1);
            resize!(y_by_cb[key], ri - 1)
            if !isempty(xv)
                plot!(
                    base_plot,
                    xv,
                    y_by_cb[key];
                    color = c,
                    line = (:solid, width_of_bin(bbin)),
                    linealpha = linealpha_of_bin(bbin),
                    label = false,
                )
            end
        end
    end

    return base_plot
end
