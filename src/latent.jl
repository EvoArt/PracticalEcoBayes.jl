# Exact full-conditional kernels for the latent states of hierarchical
# detection models: occupancy (present/absent) and N-mixture abundance.

"""
    OccupancyStates(data, name)

A Gibbs kernel for the latent occupancy states `name` (one `Bool` per site) of a
single-species occupancy model (MacKenzie et al. 2002):

    z[i] ~ Bernoulli(ψ[i]),   y[i, j] ~ Bernoulli(z[i] * p[i, j])

`data(c)` returns `(ψ, p, y)` for the current state: occupancy probabilities (one
per site), detection probabilities and detections (sites × visits; `missing` for a
visit that did not happen). A site with a detection is occupied; otherwise

    P(z = 1 | y) = ψ ∏(1 - p) / (ψ ∏(1 - p) + 1 - ψ).
"""
struct OccupancyStates{F} <: AbstractGibbsKernel
    name::Symbol
    data::F
end
OccupancyStates(data, name::Symbol) = OccupancyStates(name, data)

function PracticalBayes.gibbs_step(rng, k::OccupancyStates, names, c::ModelConditional)
    names == (k.name,) || error("OccupancyStates owns block $(k.name), got $names")
    ψ, p, y = k.data(c)
    z = map(axes(y, 1)) do i
        any(v -> !ismissing(v) && v > 0, view(y, i, :)) && return true
        l1 = log(ψ[i])
        for j in axes(y, 2)
            ismissing(y[i, j]) || (l1 += log1p(-p[i, j]))
        end
        rand(rng) < logistic(l1 - log1p(-ψ[i]))
    end
    return NamedTuple{(k.name,)}((z,))
end

"""
    NMixtureAbundance(data, name; K=nothing)

A Gibbs kernel for the latent abundances `name` (one `Int` per site) of a binomial
N-mixture model (Royle 2004):

    N[i] ~ Poisson(λ[i]),   y[i, j] ~ Binomial(N[i], p[i, j])

`data(c)` returns `(λ, p, y)` (sites; sites × visits; `missing` for no visit).
Each `N[i]` is drawn exactly from its conditional on `max(y[i, :]):K`. `K`
defaults to a bound well into the upper tail of the conditional.
"""
struct NMixtureAbundance{F} <: AbstractGibbsKernel
    name::Symbol
    data::F
    K::Union{Nothing,Int}
end
NMixtureAbundance(data, name::Symbol; K=nothing) = NMixtureAbundance(name, data, K)

function PracticalBayes.gibbs_step(rng, k::NMixtureAbundance, names, c::ModelConditional)
    names == (k.name,) || error("NMixtureAbundance owns block $(k.name), got $names")
    λ, p, y = k.data(c)
    N = map(axes(y, 1)) do i
        obs = [(y[i, j], p[i, j]) for j in axes(y, 2) if !ismissing(y[i, j])]
        lo = isempty(obs) ? 0 : maximum(first, obs)
        # unobserved individuals given N ≥ lo are Poisson(λ ∏(1-p)): bound well above
        q = isempty(obs) ? 1.0 : prod(1 - last(o) for o in obs)
        hi = k.K === nothing ? lo + ceil(Int, λ[i] * q + 10sqrt(λ[i] * q + 1) + 10) : k.K
        hi >= lo || throw(ArgumentError("K = $hi is below the largest count $lo at site $i"))
        support = lo:hi
        lp = [logpdf(Poisson(λ[i]), n) + sum((logpdf(Binomial(n, pj), yj) for (yj, pj) in obs); init=0.0)
              for n in support]
        support[_draw_index(rng, lp)]
    end
    return NamedTuple{(k.name,)}((N,))
end

# Index drawn with probabilities ∝ exp(lp)
function _draw_index(rng, lp)
    w = exp.(lp .- maximum(lp))
    u = rand(rng) * sum(w)
    acc = 0.0
    for (i, wi) in enumerate(w)
        acc += wi
        u <= acc && return i
    end
    return length(w)
end
