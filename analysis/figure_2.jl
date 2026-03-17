#=
Figure 2 reproduction script — new scCChain API
=================================================

Reproduces all 8 panels of manuscript Figure 2 (10x Visium human breast cancer
communication program analysis) using ONLY the new scCChain API.

Pipeline: load data → build graph → discover programs → sample chains → train model → analyze

Each checkpoint saves:
  - PNG panel(s) to output/figure_2/panels/
  - Numerical results to output/figure_2/checkpoints/checkpoint_NN.jld2

Run: julia --project=. analysis/figure_2.jl
=#

using ScCChain
using JLD2
using DataFrames
using Plots
using StatsPlots
using Statistics
using Random
using Flux

# ─── Paths ────────────────────────────────────────────────────────────────────
const PROJECT_ROOT = dirname(@__DIR__)
const DATA_PATH = joinpath(PROJECT_ROOT, "data", "examples", "visium")
const H5AD_PATH = joinpath(DATA_PATH, "visium_preprocessed.h5ad")
const PANELS_DIR = joinpath(PROJECT_ROOT, "output", "figure_2", "panels")
const CHECKPOINTS_DIR = joinpath(PROJECT_ROOT, "output", "figure_2", "checkpoints")

mkpath(PANELS_DIR)
mkpath(CHECKPOINTS_DIR)

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 1: Data loading → Panel (a)
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 1: Loading data ───")

scdata_full = load_scdata(H5AD_PATH)
scdata = load_scdata(H5AD_PATH; subset_highly_variable = true)

cell_annotation = ScCChain.cell_annotation(scdata; column = "Annotation")

# Coordinate transform: swap x/y and flip y to match manuscript orientation
# (isometric — preserves distances, so scdata can still be used for graph construction)
coords = Float32.(spatial_coords(scdata))
coords = coords[:, [2, 1]]
coords[:, 2] .= coords[:, 2] .* -1.0f0

full_gene_count = size(expression_matrix(scdata_full), 2)
println(
    "  HVG-only load active: kept $(size(expression_matrix(scdata), 2)) of $(full_gene_count) genes",
)

# Cell type colormap
color_json_path = joinpath(DATA_PATH, "cell_states_colors.json")
import JSON
color_dict = JSON.parsefile(color_json_path)
cell_type_colormap = Dict{String,String}(k => v for (k, v) in color_dict)

# Panel (a): Spatial cell type plot
base_plot = plot_spatial(
    coords,
    cell_annotation;
    custompalette = collect(values(cell_type_colormap)),
    markersize = 5.8,
    dpi = 400,
    plot_size = (1200, 600),
)
# Scale bar: 300 μm
ys = coords[:, 2]
xs = coords[:, 1]
ymin = minimum(ys)
cands = findall(abs.(ys .- ymin) .<= 1e-8)
idx = cands[argmax(xs[cands])]
x0, y0 = xs[idx], ys[idx]
shift_right = 200
plot!(
    base_plot,
    [x0 + shift_right, x0 + shift_right],
    [y0, y0 + 300];
    color = :black,
    lw = 10,
    label = false,
)
savefig(base_plot, joinpath(PANELS_DIR, "panel_a_spatial.png"))
println("  Panel (a) saved.")

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_01.jld2") coords cell_annotation full_gene_count
println("  Checkpoint 1 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 2: LR pair DB
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 2: Loading LR pair DB ───")

db = load_cellchat_db(; species = "human", communication_type = "Secreted Signaling")

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_02.jld2") db
println("  Checkpoint 2 saved. n_lrpairs=$(n_lrpairs(db))")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 3: Cell graph construction
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 3: Building cell graph ───")

# PCA on HVG-only matrix for similarity layer
pcs = pca(scdata; k = 30, standardize = true)

graph = build_cell_graph(
    scdata,
    db;
    radius = 300,
    alpha = 0.00002,
    dim_red = pcs,
    symmetrize_similarity = false,
)

println(
    "  Graph: $(length(graph.communication_layers)) communication layers, $(length(graph.kept_communication_indices)) kept",
)

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_03.jld2") graph
println("  Checkpoint 3 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 4: Communication program discovery
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 4: Discovering communication programs ───")

programs = discover_programs_bae(
    graph;
    n_programs = 8,
    max_iter = 10000,
    tol = 1e-5,
    batchsize = 512,
    η = 0.01,
    λ = 0.1,
    ϵ = 0.01,
    M = 1,
    seed = 42,
    basis_selection = true,
    basis_nbins = 1024,
    basis_j_threshold = 0.85,
    min_obs = 20,
    min_features = 3,
    soft_clustering = true,
)

n_programs_kept = length(unique(programs.cluster_labels))
println("  Programs kept: $(n_programs_kept)")

# Panel (b): Pathway bar chart — uses post-filtering communication names from programs
pathway_lookup = Dict{String,String}()
for r in db.records
    pathway_lookup[r.name] = r.pathway
end
communication_names_filtered = programs.metadata["communication_names"]
pathway_names_filtered =
    [get(pathway_lookup, name, "Unknown") for name in communication_names_filtered]

pl_b = plot_bars(
    pathway_names_filtered;
    title = "LR pairs per Pathway Distribution",
    ylabel = "LR pair counts",
    sort_by = :count,
    dpi = 400,
    plot_size = (1400, 500),
)
savefig(pl_b, joinpath(PANELS_DIR, "panel_b_pathway_bars.png"))
println("  Panel (b) saved.")

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_04.jld2") n_programs_kept
println("  Checkpoint 4 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 5: Chain sampling
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 5: Sampling communication chains ───")

chain_result = sample_chains(
    graph,
    programs;
    q0 = [0.5, 0.5],
    n_samples = 5,
    n_steps = 5,
    seed = 42,
    q = 0.95,
)

chains = chain_result.chains
stacked_matrix = chain_result.stacked_matrix
communication_labels = chain_result.communication_labels
n_chains = length(chains)

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_05.jld2") chains stacked_matrix communication_labels n_chains
println("  Checkpoint 5 saved. Chains: $(n_chains)")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 6: Transformer training + prediction + attention
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 6: Training transformer model ───")

model = train_model(
    chains,
    programs,
    scdata;
    n_heads = 8,
    qk_dim = 10,
    v_dim = 10,
    hidden_dim = 10,
    p_dropout = 0.2,
    n_epochs = 1000,
    batch_size = 512,
    learning_rate = 1e-3,
    patience = 20,
    seed = 42,
    weight_decay = 0.0,
    decoder_nlayers = 2,
    decoder_activation = tanh_fast,
)

model_result = ScCChain.predict(model, chains, scdata)
predictions = model_result.predictions
attention_per_head = model_result.attention_per_head

# Build chain metadata with errors and attention
metadata =
    construct_chain_metadata(chain_result; scdata = scdata, annotation_col = "Annotation")
add_chain_model_errors_to_metadata!(
    metadata,
    chain_result,
    model_result,
    scdata;
    mode = :adj_mse,
    error_pcts = [10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100],
)
add_max_attention_to_chain_metadata!(
    metadata,
    model_result,
    chain_result;
    scdata = scdata,
    annotation_col = "Annotation",
)

baseline_adj_mse = metadata.chain_wise_errors

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_06.jld2") predictions attention_per_head baseline_adj_mse metadata
println("  Checkpoint 6 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 7: Chain count bars → Panel (c)
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 7: Chain count bars ───")

# Create communication colormap using layer order (CP_1 through CP_N)
cp_colormap = Dict{String,String}()
for i = 1:n_programs_kept
    cp_colormap["CP_$(i)"] = ScCChain.Plotting.COMMUNICATION_COLORS[mod1(
        i,
        length(ScCChain.Plotting.COMMUNICATION_COLORS),
    )]
end

# Error-filtered subsets
error_pcts = [100, 50, 25]
chain_counts_per_cp = Dict{Int,Dict{String,Int}}()

for pct in error_pcts
    mask = metadata[!, Symbol(string(pct))]
    subset_inds = findall(mask)

    pl = plot_bars(
        communication_labels;
        subset_inds = subset_inds,
        colormap = cp_colormap,
        title = "Top $(pct)%",
        ylabel = "Communication chain counts",
        sort_by = :count,
        dpi = 400,
        plot_size = (600, 500),
    )
    savefig(pl, joinpath(PANELS_DIR, "panel_c_chain_counts_$(pct).png"))

    # Save counts for validation
    labs = communication_labels[subset_inds]
    counts = Dict{String,Int}()
    for l in labs
        counts[l] = get(counts, l, 0) + 1
    end
    chain_counts_per_cp[pct] = counts
end
println("  Panel (c) saved.")

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_07.jld2") chain_counts_per_cp
println("  Checkpoint 7 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 8: Spatial chain plots → Panels (d, e)
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 8: Spatial chain plots ───")

focus_cp = "CP_2"  # CP2 for panel (e)

for pct in error_pcts
    mask = metadata[!, Symbol(string(pct))]
    subset_inds = findall(mask)

    # Panel (d): All CPs
    pl_d = plot_chains!(
        coords,
        cell_annotation,
        stacked_matrix[subset_inds, :];
        communication_labels = communication_labels[subset_inds],
        communication_colormap = cp_colormap,
        error_vec = Float64.(baseline_adj_mse[subset_inds]),
        n_weight_bins = 10,
        max_line_width = 4.0,
        min_line_width = 0.5,
        alpha_min = 0.5,
        alpha_gamma = 2.0,
        line_gamma = 2.0,
        markersize = 5.8,
        dpi = 400,
        plot_size = (1200, 600),
    )
    savefig(pl_d, joinpath(PANELS_DIR, "panel_d_chains_all_$(pct).png"))

    # Panel (e): CP2 only
    cp_mask = communication_labels[subset_inds] .== focus_cp
    cp_subset = subset_inds[cp_mask]

    if !isempty(cp_subset)
        pl_e = plot_chains!(
            coords,
            cell_annotation,
            stacked_matrix[cp_subset, :];
            communication_labels = communication_labels[cp_subset],
            communication_colormap = cp_colormap,
            error_vec = Float64.(baseline_adj_mse[cp_subset]),
            n_weight_bins = 10,
            max_line_width = 4.0,
            min_line_width = 0.5,
            alpha_min = 0.5,
            alpha_gamma = 2.0,
            line_gamma = 2.0,
            markersize = 5.8,
            dpi = 400,
            plot_size = (1200, 600),
        )
        savefig(pl_e, joinpath(PANELS_DIR, "panel_e_chains_cp2_$(pct).png"))
    end
end
println("  Panels (d, e) saved.")

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_08.jld2") focus_cp
println("  Checkpoint 8 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 9: Gini impurity → Panel (f)
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 9: Gini impurity scores ───")

gini_pcts = [10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
n_unique_cell_types = length(unique(cell_annotation))
cp_nums = ["CP_$(i)" for i = 1:n_programs_kept]

# Compute Gini per CP across error percentiles
gini_scores_receiver = Matrix{Float64}(undef, length(gini_pcts), length(cp_nums))
gini_scores_max_attention = Matrix{Float64}(undef, length(gini_pcts), length(cp_nums))
gini_scores_first = Matrix{Float64}(undef, length(gini_pcts), length(cp_nums))

sorted_inds = sortperm(baseline_adj_mse)

for (j, cp) in enumerate(cp_nums)
    cp_inds = findall(communication_labels .== cp)
    cp_set = Set(cp_inds)

    for (i, pct) in enumerate(gini_pcts)
        n_all = length(sorted_inds)
        k_global = min(n_all, cld(pct * n_all, 100))
        top_global = @views sorted_inds[1:k_global]
        top_cp = [idx for idx in top_global if idx in cp_set]

        if isempty(top_cp)
            gini_scores_receiver[i, j] = NaN
            gini_scores_max_attention[i, j] = NaN
            gini_scores_first[i, j] = NaN
            continue
        end

        top_chains = chains[top_cp]
        receivers = cell_annotation[last.(top_chains)]
        firsts = cell_annotation[first.(top_chains)]

        # Max attention senders
        A_top = attention_per_head[:, :, top_cp]
        A_mean = dropdims(mean(A_top; dims = 1); dims = 1)
        max_pos = mapslices(argmax, A_mean; dims = 1)[:]
        ma_cell_ids = [top_chains[k][max_pos[k]] for k = 1:length(top_chains)]
        ma_types = cell_annotation[ma_cell_ids]

        gini_scores_receiver[i, j] =
            gini_impurity_normalized(receivers; total_K = n_unique_cell_types)
        gini_scores_max_attention[i, j] =
            gini_impurity_normalized(ma_types; total_K = n_unique_cell_types)
        gini_scores_first[i, j] =
            gini_impurity_normalized(firsts; total_K = n_unique_cell_types)
    end
end

# Panel (f): Gini lines for CP2
cp2_idx = findfirst(cp_nums .== focus_cp)
pl_f = plot_gini_lines(
    gini_pcts,
    gini_scores_receiver[:, cp2_idx],
    gini_scores_max_attention[:, cp2_idx],
    gini_scores_first[:, cp2_idx];
    title = "Gini score $(focus_cp)",
    dpi = 400,
)
savefig(pl_f, joinpath(PANELS_DIR, "panel_f_gini_cp2.png"))
println("  Panel (f) saved.")

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_09.jld2") gini_scores_receiver gini_scores_max_attention gini_scores_first gini_pcts
println("  Checkpoint 9 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 10: Top LR pairs → Panel (g)
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 10: Top LR pairs ───")

# top_features keys use the ORIGINAL cluster numbering (before _densify_integers).
# cp_mapping maps densified cluster ID → original cluster ID.
focus_cp_num = parse(Int, replace(focus_cp, "CP_" => ""))
mapped_key = "$(programs.cp_mapping[focus_cp_num])"
println(
    "  focus_cp=$(focus_cp) → densified=$(focus_cp_num) → mapped top_features key=$(mapped_key)",
)

tf_cp2 = programs.top_features[mapped_key]
top_k = 10
n_show = min(top_k, nrow(tf_cp2))
top_names = String.(tf_cp2.Features[1:n_show])
top_coeffs = Float64.(tf_cp2.normScores[1:n_show])
normalized_coeffs = Float64.(tf_cp2.normScores)

# Panel (g)
pl_g = plot_top_lrpairs(
    top_names,
    Float64.(top_coeffs);
    title = "Top $(top_k) selected LR pairs $(focus_cp)",
    dpi = 400,
)
savefig(pl_g, joinpath(PANELS_DIR, "panel_g_top_lrpairs_cp2.png"))
println("  Panel (g) saved.")

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_10.jld2") top_names top_coeffs normalized_coeffs
println("  Checkpoint 10 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Checkpoint 11: Downstream gene scores → Panel (h)
# ╚══════════════════════════════════════════════════════════════════════════════
println("─── Checkpoint 11: Downstream gene scores ───")

ppi_db = load_ppi_database("human")

# Top 3 LR pairs are already sorted in tf_cp2

top3_names = String.(tf_cp2.Features[1:min(3, nrow(tf_cp2))])
# Extract receptor gene names from LR pair names (format: "LIGAND—RECEPTOR")
receptors = String[]
for name in top3_names
    lr_parts = split(name, "—")
    if length(lr_parts) == 2
        receptor_str = lr_parts[2]
        for r in split(receptor_str, "_")
            push!(receptors, r)
        end
    end
end
receptors = unique(receptors)

downstream_genes = extract_downstream_genes(
    ppi_db,
    receptors;
    top_percent = 20,
    top_n = nothing,
    include_immediate = true,
)

# Compute scores at 25%, 50%, 100%, and non-receivers
cp2_chain_inds = findall(communication_labels .== focus_cp)
ds_pcts = [25, 50, 100]
score_groups = Vector{Vector{Float64}}()
group_labels = String[]

for pct in ds_pcts
    mask = metadata[!, Symbol(string(pct))]
    pct_inds = findall(mask)
    cp2_pct_inds = intersect(pct_inds, cp2_chain_inds)

    scores = downstream_gene_activity_score(
        scdata,
        chains,
        downstream_genes;
        chain_inds = cp2_pct_inds,
    )
    push!(score_groups, Float64.(scores))
    push!(group_labels, "$(pct)")
end

# Non-receivers baseline
all_receiver_ids = unique([last(chains[i]) for i in cp2_chain_inds])
n_cells = size(expression_matrix(scdata), 1)
non_receiver_ids = setdiff(1:n_cells, all_receiver_ids)
non_rec_avg, _, _ =
    get_avg_expression(scdata, downstream_genes; cell_inds = non_receiver_ids)
push!(score_groups, Float64.(non_rec_avg))
push!(group_labels, "Non receivers")

# Panel (h)
pl_h = plot_downstream_boxplots(
    score_groups,
    group_labels;
    title = "Downstream score $(focus_cp)",
    dpi = 400,
)
savefig(pl_h, joinpath(PANELS_DIR, "panel_h_downstream_cp2.png"))
println("  Panel (h) saved.")

JLD2.@save joinpath(CHECKPOINTS_DIR, "checkpoint_11.jld2") score_groups group_labels downstream_genes
println("  Checkpoint 11 saved.")

println("\n═══ Figure 2 reproduction complete! ═══")
println("Panels saved to: $(PANELS_DIR)")
println("Checkpoints saved to: $(CHECKPOINTS_DIR)")
println(
    "HVG-only note: this workflow now loads Visium data with subset_highly_variable=true, so parity drift versus the earlier full-data path is expected at the data-loading boundary.",
)
