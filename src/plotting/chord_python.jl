# Chord diagram via Python's mpl_chord_diagram (PythonCall wrapper).

"""
    plot_chord_python(df; kwargs...) -> PythonCall.Py

Create a chord diagram using Python's `mpl_chord_diagram` library via PythonCall.

Requires `matplotlib` and `mpl_chord_diagram` to be installed in the Python
environment used by PythonCall. If unavailable, throws an informative error.

# Arguments
- `df::AbstractDataFrame`: chain metadata with source/target cell type columns.

## Column selection
- `source_column::Symbol=:sender_cell_type`: column for source cell types.
- `target_column::Symbol=:receiver_cell_type`: column for target cell types.

## Appearance
- `cell_type_colormap::Union{Dict{String,String}, Nothing}=nothing`: label → hex color map.
  Falls back to `"tab20"` colormap if `nothing`.
- `directed::Bool=true`: if true, treat flows as directed.
- `sort::String="size"`: arc ordering — `"size"`, `"distance"`, or `""` (none).
- `chordwidth::Float64=0.7`: chord width scaling.
- `fontsize::Float64=50.0`: label font size (matplotlib scale).
- `rotate_names::Bool=true`: rotate labels tangentially.
- `pad::Float64=5.0`: padding between arcs and labels.
- `gap::Float64=0.025`: gap between arcs.
- `figsize::Tuple{Int,Int}=(35, 35)`: figure size in inches (matplotlib convention).
- `dpi::Int=300`: output resolution.

# Returns
A Python matplotlib `Figure` object (as `PythonCall.Py`).
"""
function plot_chord_python(
    df::AbstractDataFrame;
    source_column::Symbol = :sender_cell_type,
    target_column::Symbol = :receiver_cell_type,
    cell_type_colormap::Union{Dict{String,String},Nothing} = nothing,
    directed::Bool = true,
    sort::String = "size",
    chordwidth::Float64 = 0.7,
    fontsize::Float64 = 50.0,
    rotate_names::Bool = true,
    pad::Float64 = 5.0,
    gap::Float64 = 0.025,
    figsize::Tuple{Int,Int} = (35, 35),
    dpi::Int = 300,
)
    # Import Python modules
    local plt, mpl, mpl_chord, pyimport, pylist, pyconvert
    try
        PythonCall = Base.require(
            Base.PkgId(Base.UUID("6099a3de-0909-46bc-b1f4-468b9a2dfc0d"), "PythonCall"),
        )
        pyimport = PythonCall.pyimport
        pylist = PythonCall.pylist
        pyconvert = PythonCall.pyconvert
        plt = pyimport("matplotlib.pyplot")
        mpl = pyimport("matplotlib")
        mpl_chord = pyimport("mpl_chord_diagram")
    catch e
        error(
            "plot_chord_python requires PythonCall, matplotlib, and mpl_chord_diagram. " *
            "Install with: pip install matplotlib mpl_chord_diagram\n" *
            "Original error: $e",
        )
    end

    mat, labels = _build_flow_matrix(df, source_column, target_column)
    n = length(labels)

    # Build node colors
    np = pyimport("numpy")
    py_mat = np.array(mat)
    py_labels = pylist(labels)

    node_colors = nothing
    base_cmap_name = "tab20"
    if cell_type_colormap !== nothing
        base_cmap = mpl.colormaps[base_cmap_name]
        n_base = pyconvert(Int, base_cmap.N)
        node_colors = pylist([
            haskey(cell_type_colormap, l) ? cell_type_colormap[l] :
            base_cmap(mod(i - 1, n_base)) for (i, l) in enumerate(labels)
        ])
    end

    # Create figure
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)

    if node_colors === nothing
        mpl_chord.chord_diagram(
            py_mat;
            names = py_labels,
            sort = sort,
            directed = directed,
            cmap = base_cmap_name,
            chordwidth = chordwidth,
            fontsize = fontsize,
            ax = ax,
            show = false,
            rotate_names = rotate_names,
            pad = pad,
            gap = gap,
        )
    else
        mpl_chord.chord_diagram(
            py_mat;
            names = py_labels,
            sort = sort,
            directed = directed,
            colors = node_colors,
            chord_colors = node_colors,
            chordwidth = chordwidth,
            fontsize = fontsize,
            ax = ax,
            show = false,
            rotate_names = rotate_names,
            pad = pad,
            gap = gap,
        )
    end

    return fig
end
