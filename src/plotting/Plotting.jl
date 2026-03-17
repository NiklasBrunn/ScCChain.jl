"""
Spatial tissue visualization submodule.

Provides functions for plotting spatial transcriptomics data with communication
chain and cell pair overlays.
"""
module Plotting

using CategoricalArrays
using Colors
using DataFrames
using Plots
using Random
using Statistics

include("palettes.jl")
include("spatial.jl")
include("chains.jl")
include("cell_pairs.jl")
include("chord.jl")
include("chord_python.jl")
include("bars.jl")
include("eval_plots.jl")

export plot_spatial, plot_spatial!
export plot_chains!
export plot_cell_pairs!
export plot_chord
export plot_chord_python
export plot_bars
export plot_stacked_bars
export plot_gini_lines
export plot_top_lrpairs
export plot_downstream_boxplots

end # module Plotting
