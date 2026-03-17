_DATABASE_ROOT = normpath(joinpath(@__DIR__, "..", "..", "data", "databases"))

"""
    default_cellchat_path(species::String="human") -> String

Return the bundled CellChat ligand-receptor pair database CSV path for `species`.
"""
function default_cellchat_path(species::String = "human")
    species in ("human", "mouse") ||
        throw(ArgumentError("species must be \"human\" or \"mouse\", got \"$species\""))
    filename =
        species == "human" ? "CellChatDB_human_interaction.csv" :
        "CellChatDB_mouse_interaction.csv"
    return joinpath(_DATABASE_ROOT, "cellchat", filename)
end
