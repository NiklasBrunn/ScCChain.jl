# Chord diagram visualization functions.
# Approach adapted from mpl_chord_diagram (Python: https://github.com/tfardet/mpl_chord_diagram) for consistent visual output.

using DataFrames

"""
    _build_flow_matrix(df, source_column, target_column) -> (Matrix{Float64}, Vector{String})

Build a cell-type adjacency matrix from chain metadata DataFrame.

Counts rows grouped by (source_column, target_column) and returns a square matrix
of flow counts plus the ordered label vector.
"""
function _build_flow_matrix(
    df::AbstractDataFrame,
    source_column::Symbol,
    target_column::Symbol,
)
    # Collect all unique labels (preserving first-seen order, then sorted for determinism)
    all_labels =
        sort(unique(vcat(string.(df[!, source_column]), string.(df[!, target_column]))))
    label_to_idx = Dict(l => i for (i, l) in enumerate(all_labels))
    n = length(all_labels)
    mat = zeros(Float64, n, n)
    for row in eachrow(df)
        i = label_to_idx[string(row[source_column])]
        j = label_to_idx[string(row[target_column])]
        mat[i, j] += 1.0
    end
    return mat, all_labels
end

"""
    _compute_arc_layout(mat, labels, sort_mode, gap, directed)

Compute start/end angles (in degrees) for each cell-type arc on the unit circle.

For directed diagrams, each arc is split into an outgoing portion (first half)
and an incoming portion (second half).

Returns `(arc_starts, arc_ends, out_ends, order)` where `out_ends[i]` marks the
boundary between outgoing and incoming segments within arc `i`.
"""
function _compute_arc_layout(
    mat::Matrix{Float64},
    labels::Vector{String},
    sort_mode::Symbol,
    gap::Float64,
    directed::Bool,
)
    n = length(labels)
    out_deg = vec(sum(mat; dims = 2))   # row sums: outgoing
    in_deg = vec(sum(mat; dims = 1))    # col sums: incoming
    degree = directed ? (out_deg .+ in_deg) : out_deg

    # Determine arc order
    if sort_mode == :size
        order = sortperm(degree; rev = true)
    else
        order = collect(1:n)
    end

    grand_total = sum(degree)
    if grand_total == 0.0
        grand_total = 1.0
    end

    extent = 360.0  # degrees
    # Angular span per node proportional to degree
    y = degree ./ grand_total .* (extent - gap * n)
    y_out = directed ? (out_deg ./ grand_total .* (extent - gap * n)) : y

    # Compute start angles in order
    arc_starts = zeros(Float64, n)
    arc_ends = zeros(Float64, n)
    out_ends = zeros(Float64, n)

    current = 0.0
    for idx in order
        arc_starts[idx] = current
        out_ends[idx] = current + y_out[idx]
        arc_ends[idx] = current + y[idx]
        current += y[idx] + gap
    end

    return arc_starts, arc_ends, out_ends, order, out_deg, in_deg, degree
end

"""
    _compute_chord_positions(mat, arc_starts, arc_ends, out_ends, out_deg, in_deg, degree,
                             directed, sort_mode)

Compute the angular sub-segment positions for each chord (i → j).

Returns a Dict mapping `(i, j) => (src_start, src_end, dst_start, dst_end)` in degrees.
Chords are stacked within each arc so they fill the full arc proportionally.
"""
function _compute_chord_positions(
    mat::Matrix{Float64},
    arc_starts::Vector{Float64},
    arc_ends::Vector{Float64},
    out_ends::Vector{Float64},
    out_deg::Vector{Float64},
    in_deg::Vector{Float64},
    degree::Vector{Float64},
    directed::Bool,
    sort_mode::Symbol,
    gap::Float64,
)
    n = size(mat, 1)

    # Normalized angular widths for each chord within source arc (outgoing)
    zmat = Vector{Vector{Float64}}(undef, n)
    for i = 1:n
        d = directed ? out_deg[i] : degree[i]
        if d > 0
            zmat[i] = (mat[i, :] ./ d) .* (out_ends[i] - arc_starts[i])
        else
            zmat[i] = zeros(n)
        end
    end

    # Normalized angular widths for incoming (transpose of mat)
    zin_mat = if directed
        [
            begin
                d = in_deg[i]
                if d > 0
                    (mat[:, i] ./ d) .* (arc_ends[i] - out_ends[i])
                else
                    zeros(n)
                end
            end for i = 1:n
        ]
    else
        zmat
    end

    # Sort order within each arc (by size = smallest first)
    mat_ids = if sort_mode == :size
        [sortperm(zmat[i]) for i = 1:n]
    else
        [collect(1:n) for _ = 1:n]
    end

    # Compute chord positions
    pos = Dict{Tuple{Int,Int},NTuple{4,Float64}}()

    for i = 1:n
        z = zmat[i]
        z0 = arc_starts[i]

        for j in mat_ids[i]
            # Source position: stack outgoing chords
            src_start = z0
            src_end = z0 + z[j]

            # Destination position: find where chord j→i lands in arc j's incoming
            zj = zin_mat[j]
            startj = directed ? out_ends[j] : arc_starts[j]
            jids = mat_ids[j]

            if directed
                jids = reverse(jids)
            end

            # Find how far along arc j we need to go before reaching chord from i
            stop_idx = findfirst(==(i), jids)
            if stop_idx === nothing
                z0 += z[j]
                continue
            end

            startji = startj + sum(zj[jids[k]] for k = 1:(stop_idx-1); init = 0.0)
            endji = startji + zj[jids[stop_idx]]

            pos[(i, j)] = (src_start, src_end, startji, endji)
            z0 += z[j]
        end
    end

    return pos
end

"""
    _draw_arc!(p, θ_start, θ_end, inner_r, outer_r, color; n_points=50)

Draw a filled arc segment on plot `p`. Angles are in degrees.
"""
function _draw_arc!(
    p,
    θ_start::Float64,
    θ_end::Float64,
    inner_r::Float64,
    outer_r::Float64,
    color;
    n_points::Int = 50,
)
    θs = range(deg2rad(θ_start), deg2rad(θ_end); length = n_points)
    xs = vcat([outer_r * cos(θ) for θ in θs], [inner_r * cos(θ) for θ in reverse(θs)])
    ys = vcat([outer_r * sin(θ) for θ in θs], [inner_r * sin(θ) for θ in reverse(θs)])
    plot!(
        p,
        Shape(xs, ys);
        fillcolor = color,
        linecolor = color,
        linewidth = 0.5,
        label = false,
    )
end

"""
    _draw_chord!(p, start1, end1, start2, end2, radius, chordwidth, color;
                 alpha, directed, gap, n_points)

Draw a chord between two arc segments using cubic Bezier curves with control
points at `rchord = radius * (1 - chordwidth)`, matching the mpl_chord_diagram style.

For directed chords, the destination is an arrow tip. Angles are in degrees.
"""
function _draw_chord!(
    p,
    start1::Float64,
    end1::Float64,
    start2::Float64,
    end2::Float64,
    radius::Float64,
    chordwidth::Float64,
    color;
    alpha::Float64 = 0.65,
    directed::Bool = true,
    gap::Float64 = 0.03,
    n_points::Int = 60,
)
    # Convert to radians
    s1 = deg2rad(start1)
    e1 = deg2rad(end1)
    s2 = deg2rad(start2)
    e2 = deg2rad(end2)

    # Ensure correct ordering
    if s1 > e1
        s1, e1 = e1, s1
    end

    # Control point radius: chord dips inward by chordwidth fraction
    rchord = radius * (1.0 - chordwidth)

    # Cubic Bezier: P0 -> P1 -> P2 -> P3
    function bezier_cubic(p0, p1, p2, p3, t)
        c0 = (1 - t)^3
        c1 = 3 * (1 - t)^2 * t
        c2 = 3 * (1 - t) * t^2
        c3 = t^3
        return (
            c0 * p0[1] + c1 * p1[1] + c2 * p2[1] + c3 * p3[1],
            c0 * p0[2] + c1 * p1[2] + c2 * p2[2] + c3 * p3[2],
        )
    end

    polar2xy(r, θ) = (r * cos(θ), r * sin(θ))

    ts = range(0.0, 1.0; length = n_points)

    xs = Float64[]
    ys = Float64[]

    # 1. Source outer arc (from s1 to e1 along the circle)
    arc_θs = range(s1, e1; length = max(3, n_points ÷ 3))
    for θ in arc_θs
        px, py = polar2xy(radius, θ)
        push!(xs, px)
        push!(ys, py)
    end

    # 2. Connecting curve: source end → destination start
    #    Cubic Bezier with control points at rchord
    p0 = polar2xy(radius, e1)
    p3_ctrl = if directed
        # For directed: destination is an arrow, first going to start2
        asize = max(deg2rad(gap), 0.02)
        polar2xy(radius - asize, s2)
    else
        polar2xy(radius, s2)
    end
    cp1 = polar2xy(rchord, e1)
    cp2 = polar2xy(rchord, s2)
    for t in ts
        bx, by = bezier_cubic(p0, cp1, cp2, p3_ctrl, t)
        push!(xs, bx)
        push!(ys, by)
    end

    # 3. Destination: arrow tip (directed) or arc (undirected)
    if directed
        if s2 > e2
            s2, e2 = e2, s2
        end
        tip = 0.5 * (s2 + e2)
        asize = max(deg2rad(gap), 0.02)
        # Arrow: start2 → tip → end2
        tipx, tipy = polar2xy(radius, tip)
        push!(xs, tipx)
        push!(ys, tipy)
        endx, endy = polar2xy(radius - asize, e2)
        push!(xs, endx)
        push!(ys, endy)
    else
        # Destination arc
        if s2 > e2
            s2, e2 = e2, s2
        end
        dst_θs = range(s2, e2; length = max(3, n_points ÷ 3))
        for θ in dst_θs
            px, py = polar2xy(radius, θ)
            push!(xs, px)
            push!(ys, py)
        end
    end

    # 4. Return curve: destination end → source start
    p0_ret = if directed
        asize = max(deg2rad(gap), 0.02)
        polar2xy(radius - asize, e2)
    else
        polar2xy(radius, e2)
    end
    p3_ret = polar2xy(radius, s1)
    cp1_ret = polar2xy(rchord, e2)
    cp2_ret = polar2xy(rchord, s1)
    for t in ts
        bx, by = bezier_cubic(p0_ret, cp1_ret, cp2_ret, p3_ret, t)
        push!(xs, bx)
        push!(ys, by)
    end

    plot!(
        p,
        Shape(xs, ys);
        fillcolor = color,
        fillalpha = alpha,
        linecolor = color,
        linealpha = alpha * 0.3,
        linewidth = 0.2,
        label = false,
    )
end

"""
    _draw_self_chord!(p, start, stop, radius, chordwidth, color; alpha, n_points)

Draw a self-chord (loop) for node i → i.
"""
function _draw_self_chord!(
    p,
    start_deg::Float64,
    end_deg::Float64,
    radius::Float64,
    chordwidth::Float64,
    color;
    alpha::Float64 = 0.65,
    n_points::Int = 60,
)
    s = deg2rad(start_deg)
    e = deg2rad(end_deg)
    if s > e
        s, e = e, s
    end

    rchord = radius * (1.0 - chordwidth)

    polar2xy(r, θ) = (r * cos(θ), r * sin(θ))

    function bezier_cubic(p0, p1, p2, p3, t)
        c0 = (1 - t)^3
        c1 = 3 * (1 - t)^2 * t
        c2 = 3 * (1 - t) * t^2
        c3 = t^3
        return (
            c0 * p0[1] + c1 * p1[1] + c2 * p2[1] + c3 * p3[1],
            c0 * p0[2] + c1 * p1[2] + c2 * p2[2] + c3 * p3[2],
        )
    end

    xs = Float64[]
    ys = Float64[]

    # Outer arc
    arc_θs = range(s, e; length = max(3, n_points ÷ 3))
    for θ in arc_θs
        px, py = polar2xy(radius, θ)
        push!(xs, px)
        push!(ys, py)
    end

    # Return via inner (rchord) — cubic Bezier
    ts = range(0.0, 1.0; length = n_points)
    p0 = polar2xy(radius, e)
    p3 = polar2xy(radius, s)
    cp1 = polar2xy(rchord, e)
    cp2 = polar2xy(rchord, s)
    for t in ts
        bx, by = bezier_cubic(p0, cp1, cp2, p3, t)
        push!(xs, bx)
        push!(ys, by)
    end

    plot!(
        p,
        Shape(xs, ys);
        fillcolor = color,
        fillalpha = alpha,
        linecolor = color,
        linealpha = alpha * 0.3,
        linewidth = 0.2,
        label = false,
    )
end

"""
    _draw_labels!(p, labels, starts, ends, radius, fontsize, label_orientation, label_color)

Draw cell type labels around the outside of the chord diagram.

`label_orientation` can be `:radial` (orthogonal to arcs, pointing outward),
`:tangential` (along the arcs), or `:none` (no labels). Angles in degrees.
"""
function _draw_labels!(
    p,
    labels::Vector{String},
    starts::Vector{Float64},
    ends::Vector{Float64},
    radius::Float64,
    fontsize::Float64,
    label_orientation::Symbol,
    label_color,
)
    label_orientation == :none && return

    for (i, label) in enumerate(labels)
        mid_deg = (starts[i] + ends[i]) / 2.0
        mid_θ = deg2rad(mid_deg)
        lx = radius * cos(mid_θ)
        ly = radius * sin(mid_θ)

        if label_orientation == :radial
            rotation = mid_deg
            on_left = mid_deg > 90.0 && mid_deg < 270.0
            if on_left
                rotation += 180.0
            end
            halign = on_left ? :right : :left
        else  # :tangential
            rotation = mid_deg - 90.0
            on_left = mid_deg > 90.0 && mid_deg < 270.0
            if on_left
                rotation += 180.0
            end
            halign = :center
        end

        annotate!(
            p,
            lx,
            ly,
            text(label, round(Int, fontsize), halign, label_color; rotation = rotation),
        )
    end
end

"""
    plot_chord(df; kwargs...) -> Plot

Create a chord diagram showing directed communication flows between cell types.

Reads source and target cell types from a chain metadata DataFrame and draws a
circular chord diagram with arc sizes proportional to total flow. Chord rendering
follows the mpl chord diagram (python: https://github.com/tfardet/mpl_chord_diagram) approach: chords fill their proportional arc segments
and use cubic Bezier curves with control points at `radius * (1 - chordwidth)`.

# Arguments
- `df::AbstractDataFrame`: chain metadata with source/target cell type columns.

## Column selection
- `source_column::Symbol=:sender_cell_type`: column for source cell types.
- `target_column::Symbol=:receiver_cell_type`: column for target cell types.

## Appearance
- `cell_type_colormap::Union{Dict{String,String}, Nothing}=nothing`: label → hex color map.
  Falls back to `select_tableau_palette` if `nothing`.
- `directed::Bool=true`: if true, chords end in arrow tips at the receiver arc.
- `sort::Symbol=:size`: arc ordering — `:size` (largest first) or `:none`.
- `chordwidth::Float64=0.7`: controls chord curvature. Higher values make chords
  dip more toward the center.
- `fontsize::Float64=12.0`: label font size.
- `rotate_names::Bool=true`: whether to show rotated labels (false hides them).
- `label_orientation::Symbol=:radial`: label direction — `:radial` (orthogonal to arcs,
  pointing outward), `:tangential` (along the arcs), or `:none`.
- `pad::Float64=5.0`: extra space around plot edges.
- `gap::Float64=0.03`: angular gap between arcs (degrees).
- `figsize::Tuple{Int,Int}=(800, 800)`: plot size in pixels.
- `dpi::Int=300`: plot resolution.
- `bg_color=:white`: background color for the plot.
- `label_color=:black`: color for cell type labels.
- `chord_alpha::Float64=0.65`: alpha transparency for chords.
- `arc_width::Float64=0.1`: radial thickness of the outer arcs.

# Returns
A `Plots.Plot` object.
"""
function plot_chord(
    df::AbstractDataFrame;
    source_column::Symbol = :sender_cell_type,
    target_column::Symbol = :receiver_cell_type,
    cell_type_colormap::Union{Dict{String,String},Nothing} = nothing,
    directed::Bool = true,
    sort::Symbol = :size,
    chordwidth::Float64 = 0.7,
    fontsize::Float64 = 12.0,
    rotate_names::Bool = true,
    label_orientation::Symbol = :radial,
    pad::Float64 = 5.0,
    gap::Float64 = 0.03,
    figsize::Tuple{Int,Int} = (800, 800),
    dpi::Int = 300,
    bg_color = :white,
    label_color = :black,
    chord_alpha::Float64 = 0.65,
    arc_width::Float64 = 0.1,
)
    mat, labels = _build_flow_matrix(df, source_column, target_column)
    n = length(labels)

    # Resolve colors
    if cell_type_colormap === nothing
        palette = select_tableau_palette(n)
        colors = [parse(Colorant, palette[mod1(i, length(palette))]) for i = 1:n]
    else
        fallback_palette = select_tableau_palette(n)
        colors = [
            haskey(cell_type_colormap, l) ? parse(Colorant, cell_type_colormap[l]) :
            parse(Colorant, fallback_palette[mod1(i, length(fallback_palette))]) for
            (i, l) in enumerate(labels)
        ]
    end

    # Compute arc layout (degrees) with outgoing/incoming split
    arc_starts, arc_ends, out_ends, order, out_deg, in_deg, degree =
        _compute_arc_layout(mat, labels, sort, gap, directed)

    # Compute chord sub-positions within arcs
    pos = _compute_chord_positions(
        mat,
        arc_starts,
        arc_ends,
        out_ends,
        out_deg,
        in_deg,
        degree,
        directed,
        sort,
        gap,
    )

    outer_r = 1.0
    inner_r = 1.0 - arc_width
    chord_r = inner_r - 0.03  # gap between arc and chord
    label_r = 1.0 + 0.02 * pad

    # Determine effective label orientation (backward compat: rotate_names)
    eff_orientation = label_orientation
    if !rotate_names
        eff_orientation = :none
    end

    # Extra margin for radial labels
    label_margin = eff_orientation == :radial ? 0.35 : 0.1
    lim = label_r + label_margin
    lim_right = eff_orientation == :radial ? lim + 0.4 : lim

    # Initialize plot
    p = plot(;
        size = figsize,
        dpi = dpi,
        aspect_ratio = :equal,
        legend = false,
        grid = false,
        axis = false,
        ticks = false,
        framestyle = :none,
        xlims = (-lim, lim_right),
        ylims = (-lim, lim),
        background_color = bg_color,
    )

    # Draw chords first (behind arcs)
    for i = 1:n
        # Self-chords (undirected only)
        if !directed && mat[i, i] > 0 && haskey(pos, (i, i))
            s1, e1, _, _ = pos[(i, i)]
            _draw_self_chord!(
                p,
                s1,
                e1,
                chord_r,
                0.7 * chordwidth,
                colors[i];
                alpha = chord_alpha,
            )
        end

        targets = directed ? (1:n) : (1:(i-1))

        for j in targets
            if !haskey(pos, (i, j)) || mat[i, j] == 0.0
                # For undirected, also check mat[j, i]
                if !directed && mat[j, i] > 0 && haskey(pos, (i, j))
                    # draw it
                else
                    continue
                end
            end

            s1, e1, s2, e2 = pos[(i, j)]

            _draw_chord!(
                p,
                s1,
                e1,
                s2,
                e2,
                chord_r,
                chordwidth,
                colors[i];
                alpha = chord_alpha,
                directed = directed,
                gap = gap,
            )
        end
    end

    # Draw arcs (on top of chords)
    for i = 1:n
        _draw_arc!(p, arc_starts[i], arc_ends[i], inner_r, outer_r, colors[i])
    end

    # Draw labels
    _draw_labels!(
        p,
        labels,
        arc_starts,
        arc_ends,
        label_r,
        fontsize,
        eff_orientation,
        label_color,
    )

    return p
end
