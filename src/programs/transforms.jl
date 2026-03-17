using Flux

function split_loadings_nonnegative(W::AbstractMatrix{<:Real})
    p, k = size(W)
    out = zeros(Float32, p, 2k)
    for j = 1:k
        w = Float32.(W[:, j])
        out[:, 2j-1] .= max.(w, 0.0f0)
        out[:, 2j] .= max.(-w, 0.0f0)
    end
    return out
end

function _split_latent(Z::AbstractMatrix)
    d, n = size(Z)
    Zsplit = reshape(permutedims(cat(Z, -Z; dims = 3), (3, 1, 2)), 2d, n)
    return Zsplit
end

function _split_softmax(Z::AbstractMatrix)
    return Flux.softmax(_split_latent(Z); dims = 1)
end

function split_softmax_latent(Z::AbstractMatrix{<:Real})
    Zf = Float32.(Z)
    return _split_softmax(Zf)
end
