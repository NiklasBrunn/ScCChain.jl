import SparseArrays

function build_pair_feature_matrix(graph; min_obs::Int = 0, min_features::Int = 0)
    # Use distance scores from graph metadata for row ordering (column-major via findnz),
    # matching legacy CCIM construction where rows = findnz(distance_scores).
    dist_scores = get(graph.metadata, "_distance_scores", nothing)

    if !isnothing(dist_scores)
        rows, cols, _ = SparseArrays.findnz(dist_scores)
    else
        rows = Int[]
        cols = Int[]
        for layer in graph.communication_layers
            I, J, V = SparseArrays.findnz(layer)
            append!(rows, I)
            append!(cols, J)
        end
    end

    isempty(rows) && throw(
        ArgumentError(
            "No directed cell-pair observations available from communication layers",
        ),
    )

    if !isnothing(dist_scores)
        # All candidate edges from distance scores — no deduplication needed
        m = length(rows)
    else
        uniq_pairs = unique(zip(rows, cols))
        m = length(uniq_pairs)
        rows = Int[first(t) for t in uniq_pairs]
        cols = Int[last(t) for t in uniq_pairs]
    end

    p = length(graph.communication_layers)
    X = zeros(Float32, m, p)
    sidx = copy(rows)
    ridx = copy(cols)

    # Build lookup from (sender, receiver) → row index
    pair_to_row = Dict{Tuple{Int,Int},Int}()
    for i = 1:m
        pair_to_row[(rows[i], cols[i])] = i
    end

    for (f, layer) in enumerate(graph.communication_layers)
        I, J, V = SparseArrays.findnz(layer)
        for t in eachindex(I)
            row_idx = get(pair_to_row, (I[t], J[t]), 0)
            row_idx > 0 && (X[row_idx, f] = Float32(V[t]))
        end
    end

    inames = copy(graph.communication_names)

    # Filter observations (rows) by minimum number of non-zero features
    if min_features > 0
        nnz_per_row = vec(sum(X .!= 0; dims = 2))
        row_mask = nnz_per_row .>= min_features
        X = X[row_mask, :]
        sidx = sidx[row_mask]
        ridx = ridx[row_mask]
    end

    # Filter features (columns) by minimum number of non-zero observations
    if min_obs > 0
        nnz_per_col = vec(sum(X .!= 0; dims = 1))
        col_mask = nnz_per_col .>= min_obs
        X = X[:, col_mask]
        inames = inames[col_mask]
    end

    isempty(X) &&
        throw(ArgumentError("No observations remain after min_obs/min_features filtering"))

    return PairFeatureMatrix(X, sidx, ridx, inames)
end
