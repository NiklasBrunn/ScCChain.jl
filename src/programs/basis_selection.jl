@inline function _bin_index(v::Float64, nbins::Int)
    idx = Int(floor(v * nbins)) + 1
    return ifelse(idx > nbins, nbins, ifelse(idx < 1, 1, idx))
end

function _select_basis(
    M::AbstractMatrix{<:Real};
    nbins::Int = 1024,
    j_threshold::Float64 = 0.5,
    min_support::Int = 1,
    return_stats::Bool = true,
)
    nbins < 16 && throw(ArgumentError("nbins must be >= 16"))
    min_support < 1 && throw(ArgumentError("min_support must be >= 1"))
    n, p = size(M)
    n < 1 && throw(ArgumentError("M must have at least one observation"))
    p < 1 && throw(ArgumentError("M must have at least one basis column"))

    maxval = Vector{Float64}(undef, n)
    argmax_idx = Vector{Int}(undef, n)

    @inbounds begin
        col = @view M[:, 1]
        for i = 1:n
            v = Float64(col[i])
            maxval[i] = v
            argmax_idx[i] = 1
        end
        for j = 2:p
            col = @view M[:, j]
            for i = 1:n
                v = Float64(col[i])
                if v > maxval[i]
                    maxval[i] = v
                    argmax_idx[i] = j
                end
            end
        end
    end

    own_hist = zeros(Int64, nbins, p)
    other_hist = zeros(Int64, nbins, p)
    own_count = zeros(Int, p)

    @inbounds for j = 1:p
        col = @view M[:, j]
        for i = 1:n
            v = Float64(col[i])
            b = _bin_index(v, nbins)
            if argmax_idx[i] == j
                own_hist[b, j] += 1
                own_count[j] += 1
            else
                other_hist[b, j] += 1
            end
        end
    end

    keep = Int[]
    drop = Int[]
    stats = Vector{NamedTuple}(undef, p)

    total = n
    @inbounds for j = 1:p
        pos = own_count[j]
        neg = total - pos

        if pos < min_support || neg == 0
            push!(drop, j)
            stats[j] = (
                Jmax = 0.0,
                threshold = 1.0,
                err_rate = pos / total,
                auc = 0.5,
                support = pos,
                share = pos / total,
                label = j,
            )
            continue
        end

        own_tail = similar(own_hist[:, j], Int64)
        other_tail = similar(other_hist[:, j], Int64)

        s = 0
        for k = nbins:-1:1
            s += own_hist[k, j]
            own_tail[k] = s
        end
        s = 0
        for k = nbins:-1:1
            s += other_hist[k, j]
            other_tail[k] = s
        end

        bestJ = -Inf
        bestk = nbins
        best_err = 1.0
        prev_tpr = 1.0
        prev_fpr = 1.0
        auc = 0.0

        for k = 1:nbins
            tpr = own_tail[k] / pos
            fpr = other_tail[k] / neg
            J = tpr - fpr
            fp = other_tail[k]
            fn = pos - own_tail[k]
            err = (fp + fn) / total

            if J > bestJ
                bestJ = J
                bestk = k
                best_err = err
            end

            if k > 1
                auc += (prev_fpr - fpr) * (prev_tpr + tpr) / 2
            end
            prev_tpr = tpr
            prev_fpr = fpr
        end

        thr = (bestk - 1) / nbins
        stats[j] = (
            Jmax = bestJ,
            threshold = thr,
            err_rate = best_err,
            auc = auc,
            support = pos,
            share = pos / total,
            label = j,
        )
        if bestJ >= j_threshold
            push!(keep, j)
        else
            push!(drop, j)
        end
    end

    return return_stats ? (; keep, drop, stats) : (; keep, drop)
end

function _densify_integers(xs::AbstractVector{<:Integer}; order::Symbol = :numeric)
    any(<=(0), xs) && throw(ArgumentError("labels must be positive integers"))
    uniq = if order === :numeric
        sort!(collect(Set(xs)))
    elseif order === :appearance
        unique(xs)
    else
        throw(ArgumentError("order must be :numeric or :appearance"))
    end
    mapping = Dict(v => i for (i, v) in enumerate(uniq))
    densified = getindex.(Ref(mapping), xs)
    inverse_mapping = Dict(i => v for (i, v) in enumerate(uniq))
    return densified, mapping, inverse_mapping
end
