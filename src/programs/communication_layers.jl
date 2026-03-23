using SparseArrays

@inline function _keepmax!(v::AbstractVector{<:Real})
    m = maximum(v)
    z = zero(eltype(v))
    @inbounds @simd for i in eachindex(v)
        v[i] = v[i] == m ? v[i] : z
    end
    return v
end

function _extract_pair_indices(pair_metadata)
    if hasproperty(pair_metadata, :sender_index) &&
       hasproperty(pair_metadata, :receiver_index)
        return Int.(getproperty(pair_metadata, :sender_index)),
        Int.(getproperty(pair_metadata, :receiver_index))
    end
    if hasproperty(pair_metadata, :sender_cell_IDs) &&
       hasproperty(pair_metadata, :receiver_cell_IDs)
        return Int.(getproperty(pair_metadata, :sender_cell_IDs)),
        Int.(getproperty(pair_metadata, :receiver_cell_IDs))
    end
    throw(
        ArgumentError(
            "pair_metadata must provide sender_index/receiver_index or sender_cell_IDs/receiver_cell_IDs",
        ),
    )
end

"""
    programs_to_communication_layers(
        CPmat,
        n_cells::Int,
        pair_metadata;
        CP_cutoff::Bool=false,
        self_comm::Bool=true,
        materialize_dense::Bool=false,
        cutoff::Real=1e-5,
    )

Convert a per-pair communication-program matrix into per-program cell-cell communication layers.

# Arguments
- `CPmat`: Pair-by-CP matrix (`n_pairs × n_CPs`), dense or sparse.
- `n_cells::Int`: Number of cells/spots (layer size will be `n_cells × n_cells`).
- `pair_metadata`: Metadata aligned to rows of `CPmat`, providing sender/receiver indices.

# Returns
- `Vector` of communication layers, one per program:
  - sparse input returns `Vector{SparseMatrixCSC{Float32,Int}}`
  - dense input returns vector of dense slices/views (`Matrix` if `materialize_dense=true`)
"""
function programs_to_communication_layers(
    CPmat::Union{AbstractMatrix{<:Real},SparseMatrixCSC{<:Real,Int}},
    n_cells::Int,
    pair_metadata;
    CP_cutoff::Bool = false,
    self_comm::Bool = true,
    materialize_dense::Bool = false,
    cutoff::Real = 1e-5,
)
    n_cells >= 1 || throw(ArgumentError("n_cells must be >= 1"))
    cutoff >= 0 || throw(ArgumentError("cutoff must be >= 0"))

    sender, receiver = _extract_pair_indices(pair_metadata)
    n_rows, n_CPs = size(CPmat)
    length(sender) == n_rows ||
        throw(ArgumentError("pair_metadata length must match CPmat rows"))
    length(receiver) == n_rows ||
        throw(ArgumentError("pair_metadata length must match CPmat rows"))
    all(1 .<= sender .<= n_cells) || throw(ArgumentError("sender indices out of bounds"))
    all(1 .<= receiver .<= n_cells) ||
        throw(ArgumentError("receiver indices out of bounds"))

    if issparse(CPmat)
        communication_layers = Vector{SparseMatrixCSC{Float32,Int}}(undef, n_CPs)
        row_max = nothing

        if CP_cutoff
            rm = fill(zero(Float32), n_rows)
            @inbounds for j = 1:n_CPs
                col_start = CPmat.colptr[j]
                col_end = CPmat.colptr[j+1] - 1
                for p = col_start:col_end
                    i = CPmat.rowval[p]
                    v = Float32(CPmat.nzval[p])
                    if v > rm[i]
                        rm[i] = v
                    end
                end
            end
            row_max = rm
        end

        @inbounds for inter = 1:n_CPs
            col_start = CPmat.colptr[inter]
            col_end = CPmat.colptr[inter+1] - 1

            row_inds = CPmat.rowval[col_start:col_end]
            values = Float32.(CPmat.nzval[col_start:col_end])
            senders = sender[row_inds]
            receivers = receiver[row_inds]

            if !self_comm
                mask = senders .!= receivers
                row_inds = row_inds[mask]
                values = values[mask]
                senders = senders[mask]
                receivers = receivers[mask]
            end

            if CP_cutoff && !isempty(row_inds)
                keep = values .== row_max[row_inds]
                if cutoff > 0
                    keep .&= abs.(values) .>= cutoff
                end
                row_inds = row_inds[keep]
                values = values[keep]
                senders = senders[keep]
                receivers = receivers[keep]
            elseif cutoff > 0 && !isempty(values)
                keep = abs.(values) .>= cutoff
                row_inds = row_inds[keep]
                values = values[keep]
                senders = senders[keep]
                receivers = receivers[keep]
            end

            communication_layers[inter] =
                sparse(senders, receivers, values, n_cells, n_cells)
        end
        return communication_layers
    end

    A = zeros(Float32, n_cells, n_cells, n_CPs)
    cut = Float32(cutoff)

    @inbounds @views for i = 1:n_rows
        s = sender[i]
        t = receiver[i]

        if !self_comm && s == t
            continue
        end

        row_vals = Float32.(collect(CPmat[i, :]))
        if CP_cutoff
            _keepmax!(row_vals)
        end
        if cutoff > 0
            row_vals .= ifelse.(abs.(row_vals) .< cut, 0.0f0, row_vals)
        end
        A[s, t, :] .= row_vals
    end

    if !self_comm
        @inbounds for k = 1:n_CPs, c = 1:n_cells
            A[c, c, k] = 0.0f0
        end
    end

    slices = [@view A[:, :, k] for k = 1:n_CPs]
    return materialize_dense ? [copy(S) for S in slices] : slices
end
