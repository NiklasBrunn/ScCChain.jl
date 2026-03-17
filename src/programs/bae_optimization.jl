import Random
import Flux

function init_bae(
    n_features::Int,
    hp::BAEHyperparameters;
    seed::Int = 42,
    soft_clustering::Bool = false,
)
    Random.seed!(seed)
    encoder_weights = zeros(Float32, n_features, hp.zdim)
    if soft_clustering
        decoder = Flux.Chain(
            _split_softmax,
            Flux.Dense(2 * hp.zdim => n_features, Flux.tanh_fast),
            Flux.Dense(n_features => n_features),
        )
    else
        decoder = Flux.Chain(
            Flux.Dense(hp.zdim => n_features, tanh),
            Flux.Dense(n_features => n_features),
        )
    end
    return BAEModel(encoder_weights, decoder, hp)
end

@inline function get_unibeta(
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    denom::AbstractVector{T},
) where {T<:AbstractFloat}
    b = X' * y
    @. b = ifelse(denom == 0, zero(T), b / denom)
    return b
end

function compL2Boost!(
    model::BAEModel,
    l::Int,
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    denom::AbstractVector{T},
) where {T<:AbstractFloat}
    β = view(model.encoder_weights, :, l)
    update_inds = findall(x -> x != 0, β)

    for _ = 1:model.hp.M
        res = y .- (X * β)
        unibeta = get_unibeta(X, res, denom)
        optindex = argmax(unibeta .^ 2 .* denom)
        update_inds = union(update_inds, optindex)
        β[update_inds] .+= unibeta[update_inds] .* model.hp.ϵ
    end
    return nothing
end

function find_zero_columns(W::AbstractMatrix)
    inds = Int[]
    for j in axes(W, 2)
        all(iszero, @view W[:, j]) && push!(inds, j)
    end
    return inds
end

function _standardize_rows(
    X::AbstractMatrix{T};
    corrected::Bool = true,
) where {T<:AbstractFloat}
    Y = similar(X)
    for i in axes(X, 1)
        row = @view X[i, :]
        μ = mean(row)
        σ = std(row; corrected = corrected)
        if !isfinite(σ) || σ <= eps(T)
            Y[i, :] .= row .- μ
        else
            Y[i, :] .= (row .- μ) ./ σ
        end
    end
    return Y
end

function _standardize_cols(X::AbstractMatrix{<:Real}; corrected::Bool = true)
    Xf = Float32.(X)
    μ = Float32.(vec(mean(Xf, dims = 1)))
    σ = Float32.(vec(std(Xf, dims = 1; corrected = corrected)))
    bad = findall(s -> !isfinite(s) || s <= eps(Float32), σ)
    isempty(bad) || throw(
        ArgumentError(
            "Cannot standardize BAE input: zero/non-finite std in feature columns $(bad)",
        ),
    )
    return (Xf .- reshape(μ, 1, :)) ./ reshape(σ, 1, :)
end

function disentangled_compL2Boost!(
    model::BAEModel,
    X::AbstractMatrix{T},
    grads::AbstractMatrix{T},
) where {T<:AbstractFloat}
    denom = vec(sum(X .^ 2, dims = 1))
    Y = _standardize_rows(-grads)
    zdim = model.hp.zdim

    for l = 1:zdim
        inds = union(find_zero_columns(model.encoder_weights), l)
        target = vec(@view Y[l, :])

        if length(inds) == zdim
            y = target
        else
            keep = setdiff(collect(1:zdim), inds)
            curdata = X * @view(model.encoder_weights[:, keep])
            estimate = qr(curdata, ColumnNorm()) \ target
            y = _standardize_rows(reshape(target .- (curdata * estimate), 1, :))[1, :]
        end

        compL2Boost!(model, l, X, y, denom)
    end
    return nothing
end

function train_bae(
    X::AbstractMatrix{<:Real},
    hp::BAEHyperparameters;
    seed::Int = 42,
    soft_clustering::Bool = false,
)
    X_st = _standardize_cols(X; corrected = true)
    all(isfinite, X_st) ||
        throw(DomainError(:X_st, "Non-finite values in standardized pair matrix"))

    n_obs, n_features = size(X_st)
    batchsize = min(hp.batchsize, n_obs)
    model = init_bae(n_features, hp; seed = seed, soft_clustering = soft_clustering)
    opt = Flux.AdamW(hp.η, (0.9, 0.999), hp.λ)
    opt_state = Flux.setup(opt, model.decoder)
    Random.seed!(seed)

    prev_loss = Inf
    for run = 1:hp.n_runs
        if run > 1
            model.encoder_weights .= 0.0f0
        end

        for _ = 1:hp.max_iter
            sample_inds = Random.randperm(n_obs)[1:batchsize]
            batch = @view X_st[sample_inds, :]
            batch_t = permutedims(batch)
            Z = transpose(model.encoder_weights) * batch_t

            batch_loss, grads = Flux.withgradient(model.decoder, Z) do m, z
                Xhat = m(z)
                Flux.mse(Xhat, batch_t)
            end

            disentangled_compL2Boost!(model, batch, grads[2])
            Flux.update!(opt_state, model.decoder, grads[1])

            if !isnothing(hp.tol) && abs(batch_loss - prev_loss) < hp.tol
                break
            end
            prev_loss = batch_loss
        end
    end

    all(isfinite, model.encoder_weights) ||
        throw(DomainError(:encoder_weights, "Non-finite BAE encoder weights"))

    latent = X_st * model.encoder_weights
    loadings_split_nonnegative = split_loadings_nonnegative(model.encoder_weights)
    cluster_probs = split_softmax_latent(permutedims(latent))
    cluster_labels = [argmax(cluster_probs[:, i]) for i in axes(cluster_probs, 2)]
    latent_split_softmax = cluster_probs
    metadata = Dict{String,Any}(
        "sparsity" =>
            1.0f0 - (count(!iszero, model.encoder_weights) / length(model.encoder_weights)),
        "n_observations" => n_obs,
        "n_features" => n_features,
    )

    return (
        encoder_weights = copy(model.encoder_weights),
        loadings_split_nonnegative = loadings_split_nonnegative,
        latent = latent,
        cluster_probs = cluster_probs,
        cluster_labels = cluster_labels,
        latent_split_softmax = latent_split_softmax,
        metadata = metadata,
    )
end
