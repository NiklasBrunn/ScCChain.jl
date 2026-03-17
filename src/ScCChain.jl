"""
    ScCChain

A Julia framework for spatial cell–cell communication program analysis.

Pipeline stages:
1. `build_cell_graph`    — multi-layer weighted cell graph (src/graph/)
2. `discover_programs`   — structured dimensionality reduction (src/programs/)
3. `sample_chains`       — communication chains via random walks (src/chains/)
4. `train_model`/`predict` — transformer-based prioritization (src/model/)

See README.md for a quickstart example.
"""
module ScCChain

include("io/IO.jl")
using .IO

include("utils/Utils.jl")
using .Utils

include("graph/Graph.jl")
using .Graph

include("programs/Programs.jl")
using .Programs

include("model/Model.jl")
using .Model

include("chains/Chains.jl")
using .Chains

include("eval/Eval.jl")
using .Eval

include("plotting/Plotting.jl")
using .Plotting

# Re-export public API
export load_lrpair_db                              # io
export load_cellchat_db                            # io
export load_custom_lrpair_db                       # io
export merge_lrpair_dbs                            # io
export load_scdata                                 # io
export preprocess_scdata                           # io
export expression_matrix, raw_expression_matrix    # io accessors
export obs_table, var_table                        # io accessors
export cell_annotation                             # io accessors
export spatial_coords                              # io accessors
export obsm, varm, obsp, varp                      # io accessors
export layers, layer                               # io accessors
export uns                                         # io accessors
export load_example_dataset_manifest               # io datasets
export example_dataset_path, download_example_dataset, resolve_example_dataset # io datasets
export default_cellchat_path                       # io paths
export LRPairRecord, LRPairDB                      # io types
export scData                                      # io types
export n_lrpairs, lrpair_names                     # io accessors
export all_ligands, all_receptors, to_dataframe    # io accessors
export load_ppi_database, default_ppi_database_path # io ppi
export extract_downstream_genes                    # io ppi
export pca                                         # utils
export build_cell_graph                            # graph
export CellGraph                                   # graph
export discover_programs                           # programs
export discover_programs_bae                       # programs
export ProgramResult                               # programs
export programs_to_communication_layers            # programs
export top_features_per_program                    # programs
export sample_chains                               # chains
export ChainResult                                 # chains
export construct_chain_metadata                    # chains
export add_max_attention_to_chain_metadata!        # chains
export add_chain_model_errors_to_metadata!         # chains
export add_distances_to_chain_metadata!            # chains
export add_similarities_to_chain_metadata!         # chains
export subset_chain_metadata!                      # chains
export train_model, predict                        # model
export ModelResult                                 # model
export gini_impurity, gini_impurity_normalized             # eval
export downstream_gene_activity_score, get_avg_expression  # eval
export plot_spatial, plot_spatial!                          # plotting
export plot_chains!                                        # plotting
export plot_cell_pairs!                                    # plotting
export plot_chord                                          # plotting
export plot_chord_python                                   # plotting
export plot_bars                                           # plotting
export plot_stacked_bars                                   # plotting
export plot_gini_lines                                     # plotting
export plot_top_lrpairs                                    # plotting
export plot_downstream_boxplots                            # plotting

end # module ScCChain
