# Markov random fields on a neighbour graph: Potts labels (q states) and the
# Ising / autologistic model (Bool states, Besag 1974). Labels are updated by
# single-site Gibbs sweeps from their exact conditionals; the interaction
# strength is estimated through the pseudo-likelihood, because the full
# likelihood's normalising constant is intractable.

"""
    Neighbours(adjacency; weights=nothing)

A neighbour graph: `adjacency[i]` lists the neighbours of node `i` (symmetric:
`j ∈ adjacency[i]` exactly when `i ∈ adjacency[j]`), with optional edge weights
`weights[i]` in the same order. Build one from a grid, coordinates or a matrix:
[`grid_neighbours`](@ref), [`distance_neighbours`](@ref), `Neighbours(A::AbstractMatrix)`.
"""
struct Neighbours
    adjacency::Vector{Vector{Int}}
    weights::Vector{Vector{Float64}}
    function Neighbours(adj::AbstractVector; weights=nothing)
        adj = [collect(Int, a) for a in adj]
        w = weights === nothing ? [ones(length(a)) for a in adj] : [collect(Float64, x) for x in weights]
        length(w) == length(adj) && all(length.(w) .== length.(adj)) ||
            throw(DimensionMismatch("weights must match the adjacency lists"))
        for (i, a) in enumerate(adj), j in a
            i in adj[j] || throw(ArgumentError("the graph is not symmetric: $j is a neighbour of $i but not the reverse"))
        end
        return new(adj, w)
    end
end

"`Neighbours(A)`: from a symmetric adjacency (or weight) matrix; nonzero entries are edges."
function Neighbours(A::AbstractMatrix)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("adjacency matrix must be square"))
    adj = [[j for j in 1:n if j != i && A[i, j] != 0] for i in 1:n]
    return Neighbours(adj; weights=[[A[i, j] for j in adj[i]] for i in 1:n])
end

Base.length(nb::Neighbours) = length(nb.adjacency)

"""
    grid_neighbours(nrow, ncol; diagonal=false)

The rook (or, with `diagonal=true`, queen) neighbours of an `nrow × ncol` grid,
with cells numbered column by column, as Julia stores a matrix.
"""
function grid_neighbours(nrow::Integer, ncol::Integer; diagonal::Bool=false)
    idx = LinearIndices((nrow, ncol))
    steps = diagonal ? [(a, b) for a in -1:1 for b in -1:1 if (a, b) != (0, 0)] : [(-1, 0), (1, 0), (0, -1), (0, 1)]
    adj = [Int[] for _ in 1:nrow*ncol]
    for c in 1:ncol, r in 1:nrow, (dr, dc) in steps
        rr, cc = r + dr, c + dc
        1 <= rr <= nrow && 1 <= cc <= ncol && push!(adj[idx[r, c]], idx[rr, cc])
    end
    return Neighbours(adj)
end

"""
    distance_neighbours(coords; radius)

Sites within `radius` of each other are neighbours. `coords` is a matrix with one
row per site (or a vector of points).
"""
function distance_neighbours(coords::AbstractMatrix; radius::Real)
    n = size(coords, 1)
    d(i, j) = sqrt(sum(abs2, view(coords, i, :) .- view(coords, j, :)))
    return Neighbours([[j for j in 1:n if j != i && d(i, j) <= radius] for i in 1:n])
end
distance_neighbours(points::AbstractVector; radius::Real) =
    distance_neighbours(permutedims(reduce(hcat, collect.(points))); radius)

# ---------------------------------------------------------------- Potts

"""
    PottsLabels(data, name, neighbours; q)

A Gibbs kernel for the labels `name` (one `Int` in `1:q` per node) of a Potts
model on `neighbours`:

    p(x) ∝ exp(Σᵢ h[i, xᵢ] + β Σ_{i~j} w_ij [xᵢ = xⱼ])

`data(c)` returns `(β, h)` for the current state: the interaction strength and an
`n × q` matrix of log-weights for each node and label (the data likelihood and any
label-specific prior), or `nothing` for none. One call sweeps every node once, in
random order, drawing each label from its exact conditional. A two-label Potts
model is the Ising model; for presence/absence states use [`IsingStates`](@ref).
"""
struct PottsLabels{F} <: AbstractGibbsKernel
    name::Symbol
    data::F
    neighbours::Neighbours
    q::Int
end
PottsLabels(data, name::Symbol, nb::Neighbours; q::Integer) = PottsLabels(name, data, nb, Int(q))

function PracticalBayes.gibbs_step(rng, k::PottsLabels, names, c::ModelConditional)
    names == (k.name,) || error("PottsLabels owns block $(k.name), got $names")
    β, h = k.data(c)
    x = collect(Int, getfield(c.values, k.name))
    n = length(x)
    length(k.neighbours) == n || throw(DimensionMismatch("$n labels for a graph of $(length(k.neighbours)) nodes"))
    lp = zeros(k.q)
    for i in randperm(rng, n)
        fill!(lp, 0.0)
        h === nothing || (lp .+= view(h, i, :))
        for (j, w) in zip(k.neighbours.adjacency[i], k.neighbours.weights[i])
            lp[x[j]] += β * w
        end
        x[i] = _draw_index(rng, lp)
    end
    return NamedTuple{(k.name,)}((x,))
end

"""
    potts_pseudologlik(x, β, neighbours; h=nothing, q=maximum(x))

Besag's pseudo-log-likelihood of Potts labels `x`: the sum over nodes of each
label's log conditional probability given its neighbours. Differentiable in `β`
and `h`, so it can stand in for the intractable likelihood when estimating the
interaction strength (for example in a NUTS block with `@addlogprob!`).
"""
function potts_pseudologlik(x::AbstractVector{<:Integer}, β, nb::Neighbours; h=nothing, q::Integer=maximum(x))
    T = promote_type(typeof(β), h === nothing ? Float64 : eltype(h))
    total = zero(T)
    lp = zeros(T, q)
    for i in eachindex(x)
        for s in 1:q
            lp[s] = h === nothing ? zero(T) : h[i, s]
        end
        for (j, w) in zip(nb.adjacency[i], nb.weights[i])
            lp[x[j]] += β * w
        end
        total += lp[x[i]] - logsumexp(lp)
    end
    return total
end

# ---------------------------------------------------------------- Ising / autologistic

"""
    IsingStates(data, name, neighbours)

A Gibbs kernel for Bool states `name` of an Ising (autologistic) model on
`neighbours`:

    p(z) ∝ exp(Σᵢ hᵢ zᵢ + J Σ_{i~j} w_ij zᵢ zⱼ)

so that `zᵢ` given its neighbours is `Bernoulli(logistic(hᵢ + J Σⱼ w_ij zⱼ))`.
`data(c)` returns `(h, J)`: the node log-odds and the interaction strength. For
autologistic occupancy, `h` is the occupancy log-odds plus the detection
evidence `log P(y | z=1) - log P(y | z=0)` (`Inf` where a species was detected).
One call sweeps every node once in random order.
"""
struct IsingStates{F} <: AbstractGibbsKernel
    name::Symbol
    data::F
    neighbours::Neighbours
end
IsingStates(data, name::Symbol, nb::Neighbours) = IsingStates(name, data, nb)

function PracticalBayes.gibbs_step(rng, k::IsingStates, names, c::ModelConditional)
    names == (k.name,) || error("IsingStates owns block $(k.name), got $names")
    h, J = k.data(c)
    z = collect(Bool, getfield(c.values, k.name))
    length(k.neighbours) == length(z) || throw(DimensionMismatch("$(length(z)) states for a graph of $(length(k.neighbours)) nodes"))
    for i in randperm(rng, length(z))
        s = sum((w * z[j] for (j, w) in zip(k.neighbours.adjacency[i], k.neighbours.weights[i])); init=0.0)
        z[i] = rand(rng) < logistic(h[i] + J * s)
    end
    return NamedTuple{(k.name,)}((z,))
end

"""
    autologistic_pseudologlik(z, η, J, neighbours)

Pseudo-log-likelihood of Bool states `z` under the autologistic model with node
log-odds `η` (a number or one per node) and interaction `J`. Differentiable in
`η` and `J`.
"""
function autologistic_pseudologlik(z::AbstractVector{Bool}, η, J, nb::Neighbours)
    T = promote_type(eltype(η), typeof(J))
    total = zero(T)
    for i in eachindex(z)
        s = sum((w * z[j] for (j, w) in zip(nb.adjacency[i], nb.weights[i])); init=zero(T))
        ηi = (η isa Number ? η : η[i]) + J * s
        total += z[i] ? -log1pexp(-ηi) : -log1pexp(ηi)
    end
    return total
end
