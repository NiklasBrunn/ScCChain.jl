"""
Shared utilities submodule.
"""
module Utils

using Printf
using ProgressMeter
using Statistics
using LinearAlgebra
using ..IO: scData, expression_matrix

"""
    pca(X; k=30, standardize=true, corrected=true, promote_float64=true)

Compute the first `k` principal components of `X` (n_obs × n_features).

Optionally standardizes columns (mean-center and divide by std) before SVD.

Columns with zero or non-finite std are mean-centered only (not scaled).

# Arguments
- `X::AbstractMatrix{<:Real}`: input matrix (observations × features)
- `k::Int=30`: number of principal components to return
- `standardize::Bool=true`: whether to standardize columns before SVD
- `corrected::Bool=true`: use Bessel-corrected std (N-1 denominator)
- `promote_float64::Bool=true`: if `true`, promote input to Float64 before SVD
  for numerical stability; set to `false` to operate in the input element type
  (e.g. Float32) for exact legacy reproduction

# Returns
- `Matrix` of shape `(n_obs, k)` — the first `k` principal component scores
"""
function pca(
    X::AbstractMatrix{<:Real};
    k::Int = 30,
    standardize::Bool = true,
    corrected::Bool = true,
    promote_float64::Bool = true,
)
    Xf = promote_float64 ? Float64.(X) : float.(X)
    n_obs, n_feat = size(Xf)
    k = min(k, n_feat, n_obs)

    if standardize
        μ = vec(mean(Xf; dims = 1))
        σ = vec(std(Xf; dims = 1, corrected = corrected))
        for j = 1:n_feat
            if isfinite(σ[j]) && σ[j] > 0
                @views Xf[:, j] .= (Xf[:, j] .- μ[j]) ./ σ[j]
            else
                @views Xf[:, j] .= Xf[:, j] .- μ[j]
            end
        end
    end

    U, S, _ = svd(Xf)
    return U[:, 1:k] * Diagonal(S[1:k])
end

"""
    pca(scdata::scData; k=30, standardize=true, corrected=true, promote_float64=true)

Convenience overload that extracts the expression matrix from `scdata`, drops
zero-sum columns, and forwards to the matrix-based `pca`.

# Arguments
- `scdata::scData`: spatial data handle
- `k`, `standardize`, `corrected`, `promote_float64`: see matrix-based `pca`

# Returns
- `Matrix` of shape `(n_obs, k)` — the first `k` principal component scores
"""
function pca(
    scdata::scData;
    k::Int = 30,
    standardize::Bool = true,
    corrected::Bool = true,
    promote_float64::Bool = true,
)
    X = expression_matrix(scdata)
    nonzero_cols = findall(x -> x > 0, vec(sum(X; dims = 1)))
    return pca(
        X[:, nonzero_cols];
        k = k,
        standardize = standardize,
        corrected = corrected,
        promote_float64 = promote_float64,
    )
end

export pca

end # module Utils
