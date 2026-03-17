# Color palettes for spatial visualizations.

const DEFAULT_MARKERSIZE = 1.0

const TABLEAU10 = [
    "#81a0c2",  # Dark Blue
    "#a1d9f9",  # Light Blue
    "#e17055",  # Dark Red
    "#f4a8a5",  # Light Red
    "#87bd81",  # Dark Green
    "#b8e994",  # Light Green
    "#f8a95d",  # Dark Orange
    "#fdcb6e",  # Light Orange
    "#c9a1bd",  # Dark Purple
    "#e1bee7",  # Light Purple
    "#d7ccc8",
]

const TABLEAU15 = [
    "#81a0c2",
    "#f8a95d",
    "#ed8d8c",
    "#9cccc9",
    "#87bd81",
    "#f3d975",
    "#c9a1bd",
    "#ffbac0",
    "#a1d9f9",
    "#fab1a0",
    "#b8e994",
    "#ffeaa7",
    "#dfe6e9",
    "#fdcb6e",
    "#e17055",
]

const TABLEAU25 = [
    "#81a0c2",
    "#f8a95d",
    "#ed8d8c",
    "#9cccc9",
    "#87bd81",
    "#f3d975",
    "#B083B9",
    "#ffbac0",
    "#5AC8FA",
    "#FF9966",
    "#b8e994",
    "#ffeaa7",
    "#C9A77F",
    "#fdcb6e",
    "#e17055",
    "#e1bee7",
    "#665A99",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#AA6677",
    "#d7ccc8",
]

const TABLEAU40 = [
    "#1f77b4",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#9edae5",
    "#357900",
    "#8a41be",
    "#ce7100",
    "#9286ff",
    "#24aa75",
    "#007971",
    "#8e8a41",
    "#b2b6a2",
    "#d75555",
    "#ae3992",
    "#9a79aa",
    "#4daaff",
    "#00e7be",
    "#ba8a71",
    "#8e5d00",
    "#69aa31",
    "#ce82ff",
    "#5d59ce",
    "#55714d",
    "#7d92b2",
    "#656586",
    "#ffa6ff",
    "#ae4510",
    "#79a29e",
    "#e39e04",
    "#865979",
    "#2d92a6",
    "#e7beba",
]

const COMMUNICATION_COLORS = [
    "#FF69B4",
    "#9400D3",
    "#FF073A",
    "#39FF14",
    "#00BFFF",
    "#FFD700",
    "#8A2BE2",
    "#228B22",
    "#FF8C00",
    "#00CED1",
    "#C71585",
    "#1E90FF",
    "#FF1493",
    "#7FFF00",
    "#FF4500",
    "#00FA9A",
    "#DC143C",
    "#00FFEF",
    "#FFA500",
    "#B22222",
]

"""
    select_tableau_palette(n::Int) -> Vector{String}

Select the appropriate tableau color palette based on number of categories.

Returns palette with enough colors for `n` categories.
"""
function select_tableau_palette(n::Int)
    if n > 25
        return TABLEAU40
    elseif n > 11
        return TABLEAU25
    elseif n > 4
        return TABLEAU10
    else
        return TABLEAU15
    end
end
