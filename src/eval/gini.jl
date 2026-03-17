# Internal helper: compute normalized class proportions and observed class count.
function _class_proportions(labels; weights = nothing)
    n = length(labels)
    n == 0 && return Float64[], 0

    counts = Dict{eltype(labels),Float64}()
    if weights === nothing
        @inbounds for y in labels
            counts[y] = get(counts, y, 0.0) + 1.0
        end
    else
        length(weights) == n || throw(ArgumentError("weights must match labels length"))
        @inbounds for (y, w) in zip(labels, weights)
            w < 0 && throw(ArgumentError("weights must be ≥ 0"))
            counts[y] = get(counts, y, 0.0) + float(w)
        end
    end

    ps = Float64[]
    total = 0.0
    for v in values(counts)
        if v > 0
            total += v
            push!(ps, v)
        end
    end
    total == 0 && return Float64[], 0
    @inbounds for i in eachindex(ps)
        ps[i] /= total
    end
    return ps, length(ps)
end

"""
    gini_impurity(labels; weights=nothing) -> Float64

Gini impurity for categorical `labels`.

# Arguments
- `labels`: categorical labels (any type supporting equality)
- `weights::Union{Nothing,AbstractVector}=nothing`: optional nonnegative weights,
  same length as `labels`

# Returns
- `Float64` in `[0, 1 - 1/K]` where `K` is the number of observed classes.
  Returns `0.0` for empty or single-class input.
"""
function gini_impurity(labels; weights = nothing)
    ps, _ = _class_proportions(labels; weights = weights)
    isempty(ps) && return 0.0
    s = 0.0
    @inbounds for p in ps
        s += p * p
    end
    return 1 - s
end

"""
    gini_impurity_normalized(labels; weights=nothing, total_K=nothing) -> Float64

Normalized Gini impurity in `[0, 1]`.

Normalization: `G* = G / (1 - 1/K)`, where `K = total_K` if provided, otherwise
the number of observed classes.

# Arguments
- `labels`: categorical labels
- `weights::Union{Nothing,AbstractVector}=nothing`: optional nonnegative weights
- `total_K::Union{Nothing,Int}=nothing`: total number of possible classes for
  normalization ceiling. Must be `≥` the number of observed classes.

# Returns
- `Float64` in `[0, 1]`. Returns `0.0` if `K ≤ 1` or input is empty.
"""
function gini_impurity_normalized(
    labels;
    weights = nothing,
    total_K::Union{Nothing,Int} = nothing,
)
    ps, K_obs = _class_proportions(labels; weights = weights)
    isempty(ps) && return 0.0

    s = 0.0
    @inbounds for p in ps
        s += p * p
    end
    G = 1 - s

    K_use = isnothing(total_K) ? K_obs : total_K
    K_use < K_obs &&
        throw(ArgumentError("total_K ($total_K) must be ≥ observed classes ($K_obs)."))
    K_use <= 1 && return 0.0
    return G / (1 - 1 / K_use)
end
