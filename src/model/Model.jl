"""
Transformer model submodule.

Transformer-based architecture that predicts receiver-cell gene expression from
the sender cells in each communication chain. The model:
1. Takes communication chains as input (each chain = sequence of cell embeddings)
2. Uses multi-head attention (query from receiver cell) over sender cells
3. Predicts receiver-cell expression from the attention-weighted sender representation
4. Attention weights localize communication hotspots
"""
module Model

using LinearAlgebra
using Statistics
using Random
using Flux
using NNlib
using Zygote
using ProgressMeter
using ..IO: scData, expression_matrix

Base.@kwdef mutable struct SingleQueryMHA{D<:Dense,M<:AbstractMatrix{<:AbstractFloat},L}
    qkv::D
    W_o::M
    n_heads::Int
    qk_dim::Int
    v_dim::Int
    attn_drop::L
    receiver_first::Bool = true
end

Base.@kwdef mutable struct ReceiverQueryModel
    encoder::Any
    decoder::Any
    training_parameters::Dict{String,Any}
    is_trained::Bool = false
end

Base.@kwdef struct ModelResult
    predictions::Any
    attention_per_head::Any
    latent::Any
end

function _single_query_attention(
    Q::AbstractArray{T,3},
    K::AbstractArray{T,4},
    V::AbstractArray{T,4};
    attn_mask::Union{AbstractArray{T,2},Nothing} = nothing,
    fdrop = identity,
) where {T}
    n_heads, qk_dim, n_senders, batch_size = size(K)
    temp = convert(T, sqrt(qk_dim))

    K_reshaped = permutedims(K, (3, 2, 1, 4))
    Q_reshaped = reshape(permutedims(Q, (2, 1, 3)), qk_dim, 1, n_heads, batch_size)
    attn_scores = batched_mul(K_reshaped, Q_reshaped) ./ temp

    if !isnothing(attn_mask)
        attn_scores = attn_scores .+ reshape(attn_mask, n_senders, 1, 1, batch_size)
    end

    attn_weights = fdrop(softmax(attn_scores; dims = 1))
    V_reshaped = permutedims(V, (2, 3, 1, 4))
    Z_heads =
        reshape(batched_mul(V_reshaped, attn_weights), size(V, 2), n_heads, batch_size)

    return Z_heads, attn_weights
end

function _single_query_forward_with_attention(
    sq_mha::SingleQueryMHA,
    x::AbstractArray{T,3};
    attn_mask::Union{AbstractArray{T,2},Nothing} = nothing,
) where {T}
    n_features, sequence_length, batch_size = size(x)
    qk_dim = sq_mha.qk_dim
    v_dim = sq_mha.v_dim
    n_heads = sq_mha.n_heads

    x_reshape = reshape(x, n_features, sequence_length * batch_size)
    qkv = sq_mha.qkv(x_reshape)
    qkv = reshape(qkv, n_heads, 2 * qk_dim + v_dim, sequence_length, batch_size)

    Q = @view qkv[:, 1:qk_dim, 1, :]
    if sq_mha.receiver_first
        K = @view qkv[:, (qk_dim+1):(2*qk_dim), 2:end, :]
        V = @view qkv[:, (2*qk_dim+1):end, 2:end, :]
    else
        K = @view qkv[:, (qk_dim+1):(2*qk_dim), :, :]
        V = @view qkv[:, (2*qk_dim+1):end, :, :]
    end

    Z_heads, attn_weights =
        _single_query_attention(Q, K, V; attn_mask = attn_mask, fdrop = sq_mha.attn_drop)
    Z = sq_mha.W_o * reshape(Z_heads, n_heads * v_dim, batch_size)

    n_senders = size(K, 3)
    attn_per_head =
        permutedims(reshape(attn_weights, n_senders, n_heads, batch_size), (2, 1, 3))

    return Z, attn_per_head
end

function _single_query_forward(
    sq_mha::SingleQueryMHA,
    x::AbstractArray{T,3};
    attn_mask::Union{AbstractArray{T,2},Nothing} = nothing,
) where {T}
    n_features, sequence_length, batch_size = size(x)
    qk_dim = sq_mha.qk_dim
    v_dim = sq_mha.v_dim
    n_heads = sq_mha.n_heads

    x_reshape = reshape(x, n_features, sequence_length * batch_size)
    qkv = sq_mha.qkv(x_reshape)
    qkv = reshape(qkv, n_heads, 2 * qk_dim + v_dim, sequence_length, batch_size)

    Q = @view qkv[:, 1:qk_dim, 1, :]
    if sq_mha.receiver_first
        K = @view qkv[:, (qk_dim+1):(2*qk_dim), 2:end, :]
        V = @view qkv[:, (2*qk_dim+1):end, 2:end, :]
    else
        K = @view qkv[:, (qk_dim+1):(2*qk_dim), :, :]
        V = @view qkv[:, (2*qk_dim+1):end, :, :]
    end

    Z_heads =
        _single_query_attention(Q, K, V; attn_mask = attn_mask, fdrop = sq_mha.attn_drop)[1]
    return sq_mha.W_o * reshape(Z_heads, n_heads * v_dim, batch_size)
end

function (sq_mha::SingleQueryMHA)(
    x::AbstractArray{T,3};
    attn_mask::Union{AbstractArray{T,2},Nothing} = nothing,
) where {T}
    return _single_query_forward(sq_mha, x; attn_mask = attn_mask)
end

function (sq_mha::SingleQueryMHA)(
    x::AbstractArray{T,2};
    attn_mask::Union{AbstractArray{T,1},Nothing} = nothing,
) where {T}
    x3 = reshape(x, size(x, 1), size(x, 2), 1)
    mask3 = isnothing(attn_mask) ? nothing : reshape(attn_mask, size(attn_mask, 1), 1)
    Z = sq_mha(x3; attn_mask = mask3)
    return vec(Z)
end

Flux.@layer SingleQueryMHA trainable=(qkv, W_o)

function _default_training_parameters()
    return Dict{String,Any}(
        "n_epochs" => 100,
        "batch_size" => 256,
        "learning_rate" => 1e-3,
        "weight_decay" => 1e-6,
        "train_data_pct" => (0.7, 0.15, 0.15),
        "n_reports" => 10,
        "replace" => false,
        "padding" => true,
        "patience" => 10,
        "hidden_dim" => 64,
        "n_heads" => 8,
        "qk_dim" => 10,
        "v_dim" => 10,
        "p_dropout" => 0.0f0,
    )
end

function _init_receiver_query_model(
    n_genes::Int,
    hidden_dim::Int;
    n_heads::Int = 8,
    qk_dim::Int = 10,
    v_dim::Int = 10,
    p_dropout::Float32 = 0.0f0,
    decoder_nlayers::Int = 1,
    decoder_activation = identity,
)
    qkv = Dense(n_genes, n_heads * (2 * qk_dim + v_dim), identity; bias = false)
    W_o = Flux.glorot_uniform(hidden_dim, n_heads * v_dim)
    attn_drop = Dropout(p_dropout)
    encoder = SingleQueryMHA(qkv, W_o, n_heads, qk_dim, v_dim, attn_drop, true)

    if decoder_nlayers == 1
        decoder = Dense(hidden_dim, n_genes)
    elseif decoder_nlayers == 2
        decoder =
            Chain(Dense(hidden_dim, n_genes, decoder_activation), Dense(n_genes, n_genes))
    else
        throw(ArgumentError("decoder_nlayers must be 1 or 2"))
    end

    return ReceiverQueryModel(
        encoder = encoder,
        decoder = decoder,
        training_parameters = _default_training_parameters(),
        is_trained = false,
    )
end

function _forward_with_attention(
    model::ReceiverQueryModel,
    x::AbstractArray{T,3};
    attn_mask::Union{AbstractArray{T,2},Nothing} = nothing,
) where {T}
    Z, attn = _single_query_forward_with_attention(model.encoder, x; attn_mask = attn_mask)
    y_hat = model.decoder(Z)
    return y_hat, attn
end

function _validate_model_inputs(chains, expr)
    isempty(chains) && throw(ArgumentError("chains cannot be empty"))
    ndims(expr) == 2 || throw(ArgumentError("expr must be a 2D matrix"))
    all(isfinite, expr) || throw(ArgumentError("expr contains non-finite values"))

    n_cells = size(expr, 1)
    for (i, chain) in enumerate(chains)
        length(chain) >= 2 || throw(ArgumentError("chain $i must have length >= 2"))
        all(1 .<= chain .<= n_cells) ||
            throw(ArgumentError("chain $i has out-of-bounds cell ids"))
    end

    return nothing
end

function _build_batch(
    expr,
    chains,
    sampled_inds;
    receiver_first::Bool = true,
    padding::Bool = true,
)
    batchsize = length(sampled_inds)
    n_genes = size(expr, 2)
    max_chain_len = maximum(length(chains[i]) for i in sampled_inds)
    sampled_inds_vec = collect(sampled_inds)

    Y_batch = Matrix{Float32}(undef, n_genes, batchsize)

    if padding
        seq_len = receiver_first ? max_chain_len : max_chain_len - 1
        X_batch = zeros(Float32, n_genes, seq_len, batchsize)
        attn_mask = zeros(Float32, max_chain_len - 1, batchsize)

        for (j, chain_idx) in enumerate(sampled_inds_vec)
            chain = chains[chain_idx]
            chain_expr = Float32.(permutedims(expr[chain, :])) # genes x chain_len
            Y_batch[:, j] = @view chain_expr[:, end]

            if receiver_first
                input_len = length(chain)
                X_batch[:, 1:input_len, j] =
                    hcat(chain_expr[:, end], chain_expr[:, 1:(end-1)])
                if input_len < max_chain_len
                    attn_mask[input_len:end, j] .= -Inf32
                end
            else
                input_len = length(chain) - 1
                X_batch[:, 1:input_len, j] = chain_expr[:, 1:(end-1)]
                if input_len < max_chain_len - 1
                    attn_mask[(input_len+1):end, j] .= -Inf32
                end
            end
        end

        return X_batch, Y_batch, attn_mask, sampled_inds_vec
    end

    X_batch = Vector{Matrix{Float32}}(undef, batchsize)
    for (j, chain_idx) in enumerate(sampled_inds_vec)
        chain = chains[chain_idx]
        chain_expr = Float32.(permutedims(expr[chain, :])) # genes x chain_len
        Y_batch[:, j] = @view chain_expr[:, end]
        X_batch[j] =
            receiver_first ? hcat(chain_expr[:, end], chain_expr[:, 1:(end-1)]) :
            chain_expr[:, 1:(end-1)]
    end

    return X_batch, Y_batch, nothing, sampled_inds_vec
end

function _sample_inds(pool::Vector{Int}, batch_size::Int; replace::Bool = false)
    if replace
        return [pool[rand(1:length(pool))] for _ = 1:batch_size]
    end

    if batch_size >= length(pool)
        return copy(pool)
    end

    return shuffle(pool)[1:batch_size]
end

function _split_chain_inds(n::Int, split::NTuple{3,Float64})
    abs(sum(split) - 1.0) < 1e-8 || throw(ArgumentError("train_data_pct must sum to 1.0"))
    n >= 1 || throw(ArgumentError("at least one chain is required"))

    shuffled = shuffle(collect(1:n))
    n_train = Int(floor(split[1] * n))
    n_val = Int(floor(split[2] * n))

    n_train = clamp(n_train, 1, n)
    n_val = clamp(n_val, 0, n - n_train)

    train_inds = shuffled[1:n_train]
    val_inds = shuffled[(n_train+1):(n_train+n_val)]
    test_inds = shuffled[(n_train+n_val+1):end]

    return train_inds, val_inds, test_inds
end

function _batch_mse(model::ReceiverQueryModel, Xb, Yb, mask)
    y_hat =
        isnothing(mask) ? model.decoder(model.encoder(Xb)) :
        model.decoder(model.encoder(Xb; attn_mask = mask))
    return Flux.Losses.mse(y_hat, Yb)
end

"""
    train_model(chains, programs, expr; kwargs...)
    train_model(chains, programs, scdata::scData; kwargs...)

Train the receiver-query prioritization model from communication chains.

Each chain contributes one supervised target from its last cell ID (`chain[end]`).
Input features are arranged receiver-first for attention (`[receiver, senders...]`).

# Arguments
- `chains`: communication chains as vectors of 1-based cell IDs
- `programs`: reserved for future use in the compatibility v1 path
- `expr`: dense expression matrix with shape `(n_cells, n_genes)`
- `scdata::scData`: spatial data handle, internally mapped via `expression_matrix(scdata)`
- `kwargs...`: training hyperparameters, including `n_epochs`, `batch_size`, `learning_rate`,
  `weight_decay`, `patience`, `train_data_pct`, `hidden_dim`, `n_heads`, `qk_dim`, `v_dim`,
  `p_dropout`, `seed`, `decoder_nlayers` (1 or 2), `decoder_activation` (e.g. `tanh_fast`)

# Returns
- `ReceiverQueryModel` with learned encoder/decoder weights and `is_trained=true`
"""
function train_model(chains, programs, expr::AbstractMatrix; kwargs...)
    _validate_model_inputs(chains, expr)

    defaults = _default_training_parameters()
    n_epochs = Int(get(kwargs, :n_epochs, defaults["n_epochs"]))
    batch_size = Int(get(kwargs, :batch_size, defaults["batch_size"]))
    learning_rate = Float64(get(kwargs, :learning_rate, defaults["learning_rate"]))
    weight_decay = Float64(get(kwargs, :weight_decay, defaults["weight_decay"]))
    train_data_pct = get(kwargs, :train_data_pct, defaults["train_data_pct"])
    patience = Int(get(kwargs, :patience, defaults["patience"]))
    replace = Bool(get(kwargs, :replace, defaults["replace"]))
    padding = Bool(get(kwargs, :padding, defaults["padding"]))
    hidden_dim = Int(get(kwargs, :hidden_dim, defaults["hidden_dim"]))
    n_heads = Int(get(kwargs, :n_heads, defaults["n_heads"]))
    qk_dim = Int(get(kwargs, :qk_dim, defaults["qk_dim"]))
    v_dim = Int(get(kwargs, :v_dim, defaults["v_dim"]))
    p_dropout = Float32(get(kwargs, :p_dropout, defaults["p_dropout"]))
    decoder_nlayers = Int(get(kwargs, :decoder_nlayers, 1))
    decoder_activation = get(kwargs, :decoder_activation, identity)
    seed = Int(get(kwargs, :seed, 42))

    Random.seed!(seed)

    n_genes = size(expr, 2)
    model = _init_receiver_query_model(
        n_genes,
        hidden_dim;
        n_heads = n_heads,
        qk_dim = qk_dim,
        v_dim = v_dim,
        p_dropout = p_dropout,
        decoder_nlayers = decoder_nlayers,
        decoder_activation = decoder_activation,
    )
    training_parameters = Dict{String,Any}(
        "n_epochs" => n_epochs,
        "batch_size" => batch_size,
        "learning_rate" => learning_rate,
        "weight_decay" => weight_decay,
        "train_data_pct" => train_data_pct,
        "replace" => replace,
        "padding" => padding,
        "patience" => patience,
        "hidden_dim" => hidden_dim,
        "n_heads" => n_heads,
        "qk_dim" => qk_dim,
        "v_dim" => v_dim,
        "p_dropout" => p_dropout,
        "seed" => seed,
    )
    model.training_parameters = training_parameters

    train_inds, val_inds, _ = _split_chain_inds(length(chains), train_data_pct)
    isempty(val_inds) && (val_inds = copy(train_inds))
    Xv, Yv, maskv, _ =
        _build_batch(expr, chains, val_inds; receiver_first = true, padding = padding)

    opt = AdamW(learning_rate, (0.9, 0.999), weight_decay)
    opt_state = Flux.setup(opt, (model.encoder, model.decoder))

    best_loss = Inf32
    best_model = deepcopy(model)
    wait = 0

    for _ = 1:n_epochs
        batch_inds = _sample_inds(train_inds, batch_size; replace = replace)
        Xb, Yb, maskb, _ =
            _build_batch(expr, chains, batch_inds; receiver_first = true, padding = padding)

        _, grads = Flux.withgradient(model.encoder, model.decoder) do encoder, decoder
            y_hat =
                isnothing(maskb) ? decoder(encoder(Xb)) :
                decoder(encoder(Xb; attn_mask = maskb))
            Flux.Losses.mse(y_hat, Yb)
        end
        Flux.update!(opt_state, (model.encoder, model.decoder), grads)

        val_loss = _batch_mse(model, Xv, Yv, maskv)

        if val_loss < best_loss
            best_loss = val_loss
            best_model = deepcopy(model)
            wait = 0
        else
            wait += 1
            if wait >= patience
                break
            end
        end
    end

    best_model.is_trained = true
    return best_model
end

train_model(chains, programs, scdata::scData; kwargs...) =
    train_model(chains, programs, expression_matrix(scdata); kwargs...)


"""
    predict(model, chains, expr; kwargs...)
    predict(model, chains, scdata::scData; kwargs...)

Run inference on communication chains and return predictions plus per-head attention.

Chains are batched with receiver-first ordering and sender-only attention masking.

# Arguments
- `model::ReceiverQueryModel`: trained receiver-query model
- `chains`: communication chains as vectors of 1-based cell IDs
- `expr`: dense expression matrix with shape `(n_cells, n_genes)`
- `scdata::scData`: spatial data handle, internally mapped via `expression_matrix(scdata)`

# Returns
- [`ModelResult`](@ref) with fields:
  - `predictions`: matrix `(n_genes, n_chains)`
  - `attention_per_head`: tensor `(n_heads, n_sender_positions, n_chains)`
  - `latent`: encoder latent representation before the decoder
"""
function predict(model::ReceiverQueryModel, chains, expr::AbstractMatrix; kwargs...)
    _validate_model_inputs(chains, expr)

    sampled_inds = collect(1:length(chains))
    Xb, _, mask, _ =
        _build_batch(expr, chains, sampled_inds; receiver_first = true, padding = true)
    latent, attn_per_head =
        _single_query_forward_with_attention(model.encoder, Xb; attn_mask = mask)
    y_hat = model.decoder(latent)

    return ModelResult(
        predictions = y_hat,
        attention_per_head = attn_per_head,
        latent = latent,
    )
end

predict(model::ReceiverQueryModel, chains, scdata::scData; kwargs...) =
    predict(model, chains, expression_matrix(scdata); kwargs...)


export ModelResult, train_model, predict

end # module Model
