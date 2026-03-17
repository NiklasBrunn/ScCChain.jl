"""
    top_features_per_program(programs)

Compute ranked communication features per program from encoder weights.

For each program, extracts the LR pairs with positive encoder weight,
normalizes scores to [0, 1], and returns them sorted by score (descending).

# Arguments
- `programs`: [`ProgramResult`](@ref) returned by `discover_programs_bae` (must contain
  `encoder_weights` and `metadata["communication_names"]`)

# Returns
- `Dict{String, DataFrame}` keyed by program index string (`"1"`, `"2"`, ...),
  each with columns `Features`, `Scores`, `normScores`.
"""
function _top_features_from_loadings(
    W::AbstractMatrix{<:Real},
    communication_names::Vector{String},
)
    n_features, n_cols = size(W)

    top_features = Dict{String,DataFrame}()
    for l = 1:n_cols
        pos_inds = findall(x -> x > 0, W[:, l])

        if isempty(pos_inds)
            top_features["$l"] =
                DataFrame(Features = String[], Scores = Float32[], normScores = Float32[])
        else
            scores = Float32.(W[pos_inds, l])
            norm_scores = scores ./ maximum(scores)
            df = DataFrame(
                Features = communication_names[pos_inds],
                Scores = scores,
                normScores = norm_scores,
            )
            sort!(df, :Scores; rev = true)
            top_features["$l"] = df
        end
    end

    return top_features
end

"""
    top_features_per_program(programs)

Compute ranked communication features per program from split-nonneg loadings.

For each split-nonneg column, extracts the LR pairs with positive loading,
normalizes scores to [0, 1], and returns them sorted by score (descending).

# Arguments
- `programs`: [`ProgramResult`](@ref) returned by `discover_programs_bae`

# Returns
- `Dict{String, DataFrame}` keyed by split-nonneg column index string,
  each with columns `Features`, `Scores`, `normScores`.
"""
function top_features_per_program(programs)
    return programs.top_features
end

"""
    discover_programs(graph; n_programs=10, kwargs...)

Communication-program API returning typed BAE artifacts used by downstream
chain sampling and modeling.

# Arguments
- `graph`: `CellGraph` from `build_cell_graph`.
- `n_programs::Int=10`: Number of latent communication programs.
- `kwargs...`: Forwarded to `discover_programs_bae`.

# Returns
- [`ProgramResult`](@ref)
"""
function discover_programs(graph; n_programs::Int = 10, kwargs...)
    return discover_programs_bae(graph; n_programs = n_programs, kwargs...)
end

"""
    discover_programs_bae(graph; n_programs=10, kwargs...)

Advanced BAE API returning detailed typed artifacts and metadata for communication program
discovery on directed cell-pair observations derived from graph communication layers.

# Arguments
- `graph`: `CellGraph` from `build_cell_graph`.
- `n_programs::Int=10`: Number of latent communication programs.

## BAE hyperparameters (via `kwargs`)
- `seed::Int=42`: Random seed used for batching and initialization.
- `n_runs::Int=1`: Number of encoder restarts.
- `max_iter::Int=100`: Maximum epochs per run.
- `batchsize::Int=512`: Mini-batch size (capped at number of observations).
- `η::Real=1e-2`: Decoder optimizer learning rate.
- `λ::Real=1e-1`: Decoder weight decay.
- `ϵ::Real=1e-3`: Boosting step size.
- `M::Int=1`: Per-latent boosting iterations.
- `tol::Union{Nothing,Real}=1e-5`: Reserved stopping tolerance parameter.

## Filtering (via `kwargs`)
- `min_obs::Int=0`: Minimum non-zero observations per communication feature (column filtering).
- `min_features::Int=0`: Minimum non-zero features per cell pair (row filtering).

## Basis selection (via `kwargs`)
- `basis_selection::Bool=false`: Enable post-hoc basis selection.
- `basis_nbins::Int=1024`: Number of bins for entropy estimation (must be >= 16).
- `basis_j_threshold::Float64=0.85`: Jensen-Shannon divergence threshold (must be >= 0).
- `basis_min_support::Union{Nothing,Int}=nothing`: Minimum support per basis; defaults to 1% of observations.
- `basis_return_stats::Bool=true`: Include per-basis statistics in result.

## Clustering (via `kwargs`)
- `soft_clustering::Bool=false`: Use soft cluster assignments.

# Returns
- [`ProgramResult`](@ref)
"""
function discover_programs_bae(graph; n_programs::Int = 10, kwargs...)
    basis_selection = Bool(get(kwargs, :basis_selection, false))
    basis_nbins = Int(get(kwargs, :basis_nbins, 1024))
    basis_j_threshold = Float64(get(kwargs, :basis_j_threshold, 0.85))
    basis_min_support_kw = get(kwargs, :basis_min_support, nothing)
    basis_return_stats = Bool(get(kwargs, :basis_return_stats, true))
    basis_nbins >= 16 || throw(ArgumentError("basis_nbins must be >= 16"))
    basis_j_threshold >= 0 || throw(ArgumentError("basis_j_threshold must be >= 0"))

    hp = BAEHyperparameters(
        zdim = n_programs,
        n_runs = get(kwargs, :n_runs, 1),
        max_iter = get(kwargs, :max_iter, 100),
        tol = isnothing(get(kwargs, :tol, 1.0f-5)) ? nothing :
              Float32(get(kwargs, :tol, 1.0f-5)),
        batchsize = get(kwargs, :batchsize, 512),
        η = Float32(get(kwargs, :η, 1.0f-2)),
        λ = Float32(get(kwargs, :λ, 1.0f-1)),
        ϵ = Float32(get(kwargs, :ϵ, 1.0f-3)),
        M = get(kwargs, :M, 1),
    )
    seed = get(kwargs, :seed, 42)
    min_obs = Int(get(kwargs, :min_obs, 0))
    min_features = Int(get(kwargs, :min_features, 0))
    soft_clustering = Bool(get(kwargs, :soft_clustering, false))

    pair_matrix =
        build_pair_feature_matrix(graph; min_obs = min_obs, min_features = min_features)
    result = train_bae(pair_matrix.X, hp; seed = seed, soft_clustering = soft_clustering)
    n_obs = size(result.cluster_probs, 2)
    basis_min_support =
        isnothing(basis_min_support_kw) ? max(1, Int(round(n_obs * 0.01))) :
        Int(basis_min_support_kw)
    basis_min_support >= 1 || throw(ArgumentError("basis_min_support must be >= 1"))

    sender_index = pair_matrix.sender_index
    receiver_index = pair_matrix.receiver_index
    communication_names = pair_matrix.communication_names
    encoder_weights = result.encoder_weights
    loadings_split_nonnegative = result.loadings_split_nonnegative
    latent = result.latent
    cluster_probs = result.cluster_probs
    cluster_labels = Int.(result.cluster_labels)
    latent_split_softmax = result.latent_split_softmax
    cp_mapping = Dict(i => i for i = 1:size(cluster_probs, 1))

    basis_stats = nothing
    basis_keep = collect(1:size(cluster_probs, 1))
    basis_drop = Int[]

    if basis_selection
        t = _select_basis(
            cluster_probs';
            nbins = basis_nbins,
            j_threshold = basis_j_threshold,
            min_support = basis_min_support,
            return_stats = basis_return_stats,
        )
        isempty(t.keep) && throw(
            ArgumentError(
                "No basis survived basis selection. Lower basis_j_threshold or basis_min_support.",
            ),
        )

        keep = sort!(collect(t.keep))
        keep_set = Set(keep)
        pair_keep = findall(x -> x in keep_set, cluster_labels)
        isempty(pair_keep) && throw(
            ArgumentError("No observations remain after filtering by kept basis labels."),
        )

        cluster_probs = cluster_probs[keep, pair_keep]
        loadings_split_nonnegative = loadings_split_nonnegative[:, keep]
        encoder_weights = result.encoder_weights
        latent = permutedims(cluster_probs)
        latent_split_softmax = cluster_probs
        sender_index = sender_index[pair_keep]
        receiver_index = receiver_index[pair_keep]

        cluster_labels_old = cluster_labels[pair_keep]
        cluster_labels, _, cp_mapping =
            _densify_integers(cluster_labels_old; order = :numeric)
        latent = permutedims(cluster_probs)
        latent_split_softmax = cluster_probs

        basis_keep = keep
        basis_drop = collect(t.drop)
        basis_stats = basis_return_stats ? t.stats : nothing
    end

    metadata = copy(result.metadata)
    metadata["sender_index"] = sender_index
    metadata["receiver_index"] = receiver_index
    metadata["communication_names"] = communication_names

    # Compute top_features from the FULL split-nonneg loadings (all 2*zdim columns),
    # matching legacy topFeatures_per_Cluster which uses pre-basis-selection loadings.
    full_loadings = result.loadings_split_nonnegative
    top_features = _top_features_from_loadings(full_loadings, communication_names)

    programs_result = ProgramResult(
        encoder_weights = encoder_weights,
        loadings_split_nonnegative = loadings_split_nonnegative,
        latent = latent,
        cluster_probs = cluster_probs,
        cluster_labels = cluster_labels,
        latent_split_softmax = latent_split_softmax,
        pair_metadata = (sender_index = sender_index, receiver_index = receiver_index),
        basis_selection = (
            enabled = basis_selection,
            nbins = basis_nbins,
            j_threshold = basis_j_threshold,
            min_support = basis_min_support,
            keep = basis_keep,
            drop = basis_drop,
            stats = basis_stats,
        ),
        cp_mapping = cp_mapping,
        metadata = metadata,
        top_features = top_features,
    )

    return programs_result
end
