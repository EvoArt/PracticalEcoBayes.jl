```@meta
CurrentModule = PracticalEcoBayes
```

# PracticalEcoBayes.jl

Gibbs kernels for ecological models, for the `Gibbs` sampler of
[PracticalBayes.jl](https://github.com/EvoArt/PracticalBayes.jl). Each kernel draws one block
of a model from its exact full conditional, so discrete latent states and logistic
regressions need no gradients and mix well.

| kernel | block | model |
|---|---|---|
| [`PolyaGammaLogit`](@ref) | logistic-regression coefficients | occupancy, detection, presence/absence |
| [`OccupancyStates`](@ref) | latent presence per site | single-species occupancy |
| [`NMixtureAbundance`](@ref) | latent abundance per site | binomial N-mixture |
| [`IsingStates`](@ref) | Bool states on a graph | autologistic (spatial) occupancy |
| [`PottsLabels`](@ref) | labels `1:q` on a graph | spatial clustering |

Every kernel takes a function of the current `ModelConditional` that returns what it
needs (probabilities, designs, data), so it fits any model layout.

## A single-species occupancy model

```julia
using PracticalBayes, PracticalEcoBayes, Distributions
using LogExpFunctions: logistic

@model function occupancy(y)
    β ~ Normal(0, 2.5)                     # occupancy log-odds
    α ~ Normal(0, 2.5)                     # detection log-odds
    z = Vector{Bool}(undef, size(y, 1))
    for i in axes(y, 1)
        z[i] ~ Bernoulli(logistic(β))
        for j in axes(y, 2)
            ismissing(y[i, j]) || (y[i, j] ~ Bernoulli(z[i] * logistic(α)))
        end
    end
end

visits = [(i, j) for i in axes(y, 1), j in axes(y, 2) if !ismissing(y[i, j])]
spl = Gibbs(
    :z => OccupancyStates(c -> (fill(logistic(c.values.β), size(y, 1)),
                                fill(logistic(c.values.α), size(y)...), y), :z),
    :β => PolyaGammaLogit(c -> (ones(size(y, 1), 1), c.values.z), :β),
    :α => PolyaGammaLogit(:α) do c
        v = [(i, j) for (i, j) in visits if c.values.z[i]]
        (ones(length(v), 1), [y[i, j] for (i, j) in v])
    end)
chn = sample(occupancy(y), spl, 2000)
```

## Markov random fields

`PottsLabels` and `IsingStates` sweep a neighbour graph built with
[`grid_neighbours`](@ref), [`distance_neighbours`](@ref) or `Neighbours(A)`. Their
normalising constants are intractable, so the interaction strength is estimated with
the pseudo-likelihood ([`potts_pseudologlik`](@ref), [`autologistic_pseudologlik`](@ref)),
for example in a NUTS block via `@addlogprob!`.

For multispecies occupancy with species interactions, see
[MultispeciesOccupancy.jl](https://github.com/EvoArt/MultispeciesOccupancy.jl).

## API reference

```@autodocs
Modules = [PracticalEcoBayes]
```
