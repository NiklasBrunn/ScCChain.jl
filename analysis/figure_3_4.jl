#=
Figure 3/4 reproduction script
==============================

Reproduces all panels of manuscript Figures 3 and 4 (10x Xenium human breast
cancer, single-LR-pair CXCL12—CXCR4 analysis).

Pipeline: load data → build graph → sample chains → train model → analyze

Each checkpoint saves:
  - PNG panel(s) to output/figure_3_4/panels/
  - Numerical results to output/figure_3_4/checkpoints/checkpoint_NN.jld2

Run: julia --project=. analysis/figure_3_4.jl
=#

using ScCChain
using DelimitedFiles, DataFrames, CSV, JLD2, JSON
using LinearAlgebra, Statistics, SparseArrays, Random
using Plots, StatsPlots, Flux

# ─── Paths ────────────────────────────────────────────────────────────────────
const PROJECT_ROOT = dirname(@__DIR__)
const DATA_PATH = joinpath(PROJECT_ROOT, "data", "examples", "xenium")
const PANELS_DIR = joinpath(PROJECT_ROOT, "output", "figure_3_4", "panels")
const CHECKPOINTS_DIR = joinpath(PROJECT_ROOT, "output", "figure_3_4", "checkpoints")

mkpath(PANELS_DIR)
mkpath(CHECKPOINTS_DIR)

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 1: Data loading
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 1: Loading data ───")

scdata = load_scdata(joinpath(DATA_PATH, "xenium_preprocessed.h5ad"))
X = Float32.(expression_matrix(scdata))
cell_locations = Float32.(spatial_coords(scdata))
genenames = String.(var_table(scdata).index)
cell_annotation = ScCChain.cell_annotation(scdata; column = "Cluster")

println("  X shape = $(size(X)), cell_locations shape = $(size(cell_locations))")
println("  n_genes = $(size(X, 2)), n_cells = $(length(cell_annotation))")

# PCA on non-zero columns of full X
pcs = pca(scdata; k = 30, promote_float64 = false)

# Cell type colormap from JSON
colormap_path = joinpath(DATA_PATH, "cell_states_colors_original.json")
color_dict = JSON.parsefile(colormap_path)
cell_type_colormap = Dict{String,String}(String(k) => String(v) for (k, v) in color_dict)
# Sorted hex codes for plot_spatial
sorted_pairs = sort(collect(cell_type_colormap); by = first)
hexcodes = last.(sorted_pairs)

# Panel 3a: Spatial cell type plot
default_markersize = 1.0
base_plot = plot_spatial(
    cell_locations,
    cell_annotation;
    custompalette = hexcodes,
    markersize = default_markersize,
    dpi = 400,
    plot_size = (1200, 600),
)

# Scale bar: 50 μm
xs = cell_locations[:, 1]
ys = cell_locations[:, 2]
xmax = maximum(xs)
tol = max(eps(eltype(ys)), 1e-8)
cands = findall(abs.(xs .- xmax) .<= tol)
idx = cands[argmin(ys[cands])]
x0, y0 = xs[idx], ys[idx]

shift_right = 200
plot!(
    base_plot,
    [x0 + shift_right, x0 + shift_right],
    [y0 - 1200, y0 - 1200 + 50];
    color = :black,
    lw = 10,
    label = false,
)
savefig(base_plot, joinpath(PANELS_DIR, "panel_3a_spatial.png"))
println("  Panel 3a saved.")

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_01.jld2") X cell_locations genenames cell_annotation pcs
println("  Checkpoint 1 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 2: LR pair DB
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 2: Loading LR pair DB ───")

db = load_cellchat_db(; species = "human", lrpairs = ["CXCL12—CXCR4"])

println("  n_lrpairs = $(n_lrpairs(db))")
println("  lrpair_names = $(lrpair_names(db))")

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_02.jld2") db
println("  Checkpoint 2 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 3: Cell graph
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 3: Building cell graph ───")

graph = build_cell_graph(
    scdata,
    db;
    radius = 50,
    alpha = 0.00075,
    dim_red = pcs,
    symmetrize_similarity = false,
    promote_float64 = false,
)

println("  Graph: $(length(graph.communication_layers)) communication layer(s)")
println("  Similarity layer nnz = $(nnz(graph.similarity_layer))")
println("  Communication layer nnz = $(nnz(graph.communication_layers[1]))")

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_03.jld2") graph
println("  Checkpoint 3 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 4: Chain sampling
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 4: Sampling communication chains ───")

chain_result = sample_chains(
    graph;
    programs = nothing,
    start_from = "sender",
    q0 = [0.1, 0.9],
    n_samples = 1,
    n_steps = 19,
    seed = 42,
    q = 0.5,
)

chains = chain_result.chains
stacked_matrix = chain_result.stacked_matrix
communication_labels = chain_result.communication_labels

println("  n_chains = $(length(chains))")
println("  unique communication_labels = $(unique(communication_labels))")

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_04.jld2") chains stacked_matrix communication_labels
println("  Checkpoint 4 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 5: Transformer training + prediction + attention
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 5: Transformer model ───")

model = train_model(
    chains,
    nothing,
    scdata;
    n_heads = 8,
    qk_dim = 10,
    v_dim = 10,
    hidden_dim = 10,
    p_dropout = 0.2,
    n_epochs = 1000,
    batch_size = 2^9,
    learning_rate = 0.001,
    weight_decay = 0.0,
    patience = 20,
    seed = 42,
    decoder_nlayers = 2,
    decoder_activation = tanh_fast,
)

model_result = ScCChain.predict(model, chains, scdata)
Y_all_pred = model_result.predictions
A = model_result.attention_per_head

println("  Y_all_pred shape = $(size(Y_all_pred))")
println("  A shape = $(size(A))")

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_05.jld2") Y_all_pred A
println("  Checkpoint 5 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 6: Metadata construction
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 6: Building metadata ───")

# Construct chain metadata
chain_metadata =
    construct_chain_metadata(chain_result; scdata = scdata, annotation_col = "Cluster")

# Add max attention info
add_max_attention_to_chain_metadata!(
    chain_metadata,
    model_result,
    chain_result;
    scdata = scdata,
    annotation_col = "Cluster",
)

# Add model errors (adj_mse mode, percentiles 10:10:100)
add_chain_model_errors_to_metadata!(
    chain_metadata,
    chain_result,
    model_result,
    scdata;
    mode = :adj_mse,
    error_pcts = collect(10:10:100),
)

# Add distances
add_distances_to_chain_metadata!(chain_metadata, scdata)

println("  Metadata columns: $(names(chain_metadata))")
println("  Metadata rows: $(nrow(chain_metadata))")

# Error-sorted indices for later use
baseline_adj_mse = chain_metadata.chain_wise_errors
sorted_chain_inds = sortperm(baseline_adj_mse)

# Top 20% indices
pct_kept = 20
n_chains = length(sorted_chain_inds)
top_pct_inds = sorted_chain_inds[1:ceil(Int, pct_kept/100*n_chains)]
top_pct_set = Set(top_pct_inds)
top_pct_mask = [i in top_pct_set for i = 1:n_chains]

println("  Top $(pct_kept)% chains: $(length(top_pct_inds))")
println("  Mean baseline_adj_mse = $(mean(baseline_adj_mse))")

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_06.jld2") chain_metadata sorted_chain_inds top_pct_inds
println("  Checkpoint 6 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 7: All chains spatial + chord (Panels 3b, 3c-left)
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 7: All chains spatial + chord ───")

communication_colormap = Dict(unique(communication_labels) .=> "#FF0000")

# Panel 3b / 3c-left: All chains spatial plot
pl_all = plot_spatial(
    cell_locations,
    cell_annotation;
    custompalette = hexcodes,
    alpha = 0.25,
    markersize = default_markersize,
    dpi = 400,
    plot_size = (1200, 600),
)
pl_all = plot_chains!(
    cell_locations,
    cell_annotation,
    stacked_matrix;
    communication_labels = communication_labels,
    communication_colormap = communication_colormap,
    error_vec = nothing,
    max_line_width = 0.25,
    min_line_width = 0.25,
    markersize = default_markersize,
    dpi = 400,
    plot_size = (1200, 600),
    base_plot = pl_all,
)
savefig(pl_all, joinpath(PANELS_DIR, "panel_3b_chains_all_spatial.png"))
savefig(pl_all, joinpath(PANELS_DIR, "panel_3c_chains_all.png"))
println("  Panel 3b/3c-left (all chains spatial) saved.")

# Panel 3b: Chord diagram for all chains (first → last cell type)
chord_df_all = DataFrame(
    sender_cell_type = cell_annotation[first.(chains)],
    receiver_cell_type = cell_annotation[last.(chains)],
)
pl_chord_all = plot_chord(
    chord_df_all;
    source_column = :sender_cell_type,
    target_column = :receiver_cell_type,
    cell_type_colormap = cell_type_colormap,
    directed = true,
    sort = :size,
    label_orientation = :radial,
    fontsize = 9.0,
    gap = 5.0,
    pad = 15.0,
    chord_alpha = 0.55,
    bg_color = :white,
    label_color = :black,
    figsize = (1000, 1000),
    dpi = 300,
)
savefig(pl_chord_all, joinpath(PANELS_DIR, "panel_3b_chord_all.png"))
println("  Panel 3b chord saved.")

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_07.jld2") chord_df_all
println("  Checkpoint 7 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 8: Top 20% chains spatial + chord (Panels 3c-right, 3d)
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 8: Top 20% chains spatial + chord ───")

top_chains = chains[top_pct_inds]
top_stacked = stacked_matrix[top_pct_inds, :]
top_errors = baseline_adj_mse[top_pct_inds]

# Panel 3c-right: Top 20% chains spatial (error-weighted)
pl_top = plot_spatial(
    cell_locations,
    cell_annotation;
    custompalette = hexcodes,
    alpha = 0.25,
    markersize = default_markersize,
    dpi = 400,
    plot_size = (1200, 600),
)
pl_top = plot_chains!(
    cell_locations,
    cell_annotation,
    top_stacked;
    communication_labels = communication_labels[top_pct_inds],
    communication_colormap = communication_colormap,
    error_vec = Float64.(top_errors),
    n_weight_bins = 10,
    max_line_width = 3.0,
    min_line_width = 0.5,
    alpha_min = 0.5,
    alpha_gamma = 2.0,
    line_gamma = 2.0,
    markersize = default_markersize,
    dpi = 400,
    plot_size = (1200, 600),
    base_plot = pl_top,
)
savefig(pl_top, joinpath(PANELS_DIR, "panel_3c_chains_top20.png"))
println("  Panel 3c-right (top 20% spatial) saved.")

# Panel 3d: Chord diagram for top 20% chains
chord_df_top = DataFrame(
    sender_cell_type = cell_annotation[first.(top_chains)],
    receiver_cell_type = cell_annotation[last.(top_chains)],
)
pl_chord_top = plot_chord(
    chord_df_top;
    source_column = :sender_cell_type,
    target_column = :receiver_cell_type,
    cell_type_colormap = cell_type_colormap,
    directed = true,
    sort = :size,
    label_orientation = :radial,
    fontsize = 9.0,
    gap = 5.0,
    pad = 15.0,
    chord_alpha = 0.55,
    bg_color = :white,
    label_color = :black,
    figsize = (1000, 1000),
    dpi = 300,
)
savefig(pl_chord_top, joinpath(PANELS_DIR, "panel_3d_chord_top20.png"))
println("  Panel 3d chord saved.")

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_08.jld2") chord_df_top top_pct_inds
println("  Checkpoint 8 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 9: Distance boxplots (Panel 3e)
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 9: Distance boxplots ───")

# Get the top 20% metadata subset
meta_top20 = subset_chain_metadata!(chain_metadata, pct_kept)

# Extract distance columns
first_last_dist = Float64.(meta_top20.first_sender_receiver_distance)
max_attn_dist = Float64.(coalesce.(meta_top20.max_attention_sender_receiver_distance, NaN))
max_attn_dist_clean = filter(!isnan, max_attn_dist)
second_last_dist = Float64.(coalesce.(meta_top20.penultimate_sender_receiver_distance, NaN))
second_last_dist_clean = filter(!isnan, second_last_dist)

dist_labels = ["First sender cells", "Max. attention\nsender cells", "Last sender cells"]
dist_data = [first_last_dist, max_attn_dist_clean, second_last_dist_clean]
dist_means = mean.(dist_data)

println(
    "  Distance means: first=$(dist_means[1]), max_attn=$(dist_means[2]), last=$(dist_means[3])",
)

# Build boxplot manually for exact control
pl_dist = plot(;
    legend = false,
    size = (1200, 600),
    dpi = 400,
    ylabel = "Distance to receiver cell (μm)",
    grid = false,
    framestyle = :box,
)

for (i, (data, label)) in enumerate(zip(dist_data, dist_labels))
    boxplot!(
        pl_dist,
        fill(i, length(data)),
        data;
        label = false,
        color = "steelblue",
        fillalpha = 0.35,
        mediancolor = :black,
        whiskercolor = :black,
    )
end

# Red cross markers for means
scatter!(
    pl_dist,
    1:3,
    dist_means;
    marker = :x,
    ms = 8,
    markerstrokewidth = 5,
    mc = :red,
    label = "Mean value",
)

xticks!(pl_dist, (1:3, dist_labels))

savefig(pl_dist, joinpath(PANELS_DIR, "panel_3e_distance_boxplot.png"))
println("  Panel 3e saved.")

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_09.jld2") dist_data dist_means dist_labels
println("  Checkpoint 9 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 10: Per-receiver-type spatial plots (Panel 4a)
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 10: Per-receiver-type spatial plots ───")

receiver_types =
    ["B_Cells", "CD4+_T_Cells", "Endothelial", "Invasive_Tumor", "Macrophages_1", "Stromal"]

# Get receiver types for top 20% chains
top_receiver_types = cell_annotation[last.(top_chains)]

per_type_chain_counts = Dict{String,Int}()

for rtype in receiver_types
    rtype_mask = top_receiver_types .== rtype
    rtype_inds = findall(rtype_mask)
    per_type_chain_counts[rtype] = length(rtype_inds)

    if isempty(rtype_inds)
        println("  [WARN] No chains for receiver type $(rtype), skipping.")
        continue
    end

    rtype_stacked = top_stacked[rtype_inds, :]
    rtype_errors = top_errors[rtype_inds]

    # Error-weighted spatial plot
    pl = plot_spatial(
        cell_locations,
        cell_annotation;
        custompalette = hexcodes,
        alpha = 0.25,
        markersize = default_markersize,
        dpi = 400,
        plot_size = (1200, 600),
    )
    pl = plot_chains!(
        cell_locations,
        cell_annotation,
        rtype_stacked;
        communication_labels = fill("CXCL12—CXCR4", length(rtype_inds)),
        communication_colormap = communication_colormap,
        error_vec = Float64.(rtype_errors),
        n_weight_bins = 10,
        max_line_width = 4.0,
        min_line_width = 0.5,
        alpha_min = 0.5,
        alpha_gamma = 2.0,
        line_gamma = 2.0,
        markersize = default_markersize,
        dpi = 400,
        plot_size = (1200, 600),
        base_plot = pl,
    )
    savefig(pl, joinpath(PANELS_DIR, "panel_4a_spatial_$(rtype).png"))
    println("  Panel 4a saved for $(rtype) ($(length(rtype_inds)) chains).")
end

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_10.jld2") per_type_chain_counts receiver_types
println("  Checkpoint 10 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 11: Per-receiver-type distance boxplots (Panel 4b)
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 11: Per-receiver-type distance boxplots ───")

# Max attention sender cell IDs for top 20% chains (from metadata).
top_meta = chain_metadata[top_pct_inds, :]
top_most_attended_sender_ids = Int.(top_meta.max_attention_sender_cell_ids)

per_type_distance_data = Dict{String,Any}()

for rtype in receiver_types
    rtype_mask = top_receiver_types .== rtype
    rtype_inds_local = findall(rtype_mask)

    if isempty(rtype_inds_local)
        println("  [WARN] No chains for receiver type $(rtype), skipping.")
        continue
    end

    # Get max-attention sender types and distances for this receiver type
    ma_sender_ids = top_most_attended_sender_ids[rtype_inds_local]
    receiver_ids = last.(top_chains[rtype_inds_local])
    ma_sender_types = cell_annotation[ma_sender_ids]

    # Compute distances per sender type, filtering out types with < min_obs counts
    min_obs = 20
    unique_sender_types = sort(unique(ma_sender_types))
    sender_distances = Dict{String,Vector{Float64}}()

    for stype in unique_sender_types
        s_mask = ma_sender_types .== stype
        s_inds = findall(s_mask)
        if length(s_inds) < min_obs
            continue
        end
        dists = Float64[
            norm(cell_locations[ma_sender_ids[j], :] .- cell_locations[receiver_ids[j], :]) for j in s_inds
        ]
        sender_distances[stype] = dists
    end
    unique_sender_types = sort(collect(keys(sender_distances)))

    per_type_distance_data[rtype] =
        (sender_types = unique_sender_types, sender_distances = sender_distances)

    # Build boxplot: one box per sender type, colored by cell type.
    n_stypes = length(unique_sender_types)
    pl_width = max(800, 120 * n_stypes)
    pl = plot(;
        legend = false,
        size = (pl_width, 600),
        dpi = 400,
        ylabel = "Distance to receiver cell (μm)",
        title = rtype,
        grid = false,
        framestyle = :box,
        ylims = (0, 250),
    )

    for (i, stype) in enumerate(unique_sender_types)
        dists = sender_distances[stype]
        box_color = get(cell_type_colormap, stype, "#888888")
        boxplot!(
            pl,
            fill(i, length(dists)),
            dists;
            label = false,
            color = box_color,
            fillalpha = 0.5,
            linecolor = box_color,
            mediancolor = box_color,
            whiskercolor = box_color,
            linealpha = 0.9,
            markerstrokecolor = box_color,
        )
        y_top = maximum(dists)
        annotate!(pl, i, y_top + 5, text("$(length(dists))", 9, :center, box_color))
    end

    xticks!(pl, (1:n_stypes, unique_sender_types); xrotation = 50)

    savefig(pl, joinpath(PANELS_DIR, "panel_4b_distance_$(rtype).png"))
    println("  Panel 4b saved for $(rtype).")
end

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_11.jld2") per_type_distance_data
println("  Checkpoint 11 saved.")

println("\n═══ Figure 3/4 pipeline complete! ═══")
println("Panels saved to: $(PANELS_DIR)")
println("Checkpoints saved to: $(CHECKPOINTS_DIR)")
