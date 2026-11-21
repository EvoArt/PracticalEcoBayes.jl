# PracticalEcoBayes.jl

Gibbs kernels for ecological models, for the `Gibbs` sampler of
[PracticalBayes.jl](https://github.com/EvoArt/PracticalBayes.jl). Each kernel draws one
block of a model from its exact full conditional, so discrete latent states and
logistic regressions need no gradients.

| kernel | block | model |
|---|---|---|
| `PolyaGammaLogit` | logistic-regression coefficients | occupancy, detection, presence/absence (Pólya–Gamma augmentation) |
| `OccupancyStates` | latent presence per site | single-species occupancy |
| `NMixtureAbundance` | latent abundance per site | binomial N-mixture |
| `IsingStates` | Bool states on a graph | autologistic (spatial) occupancy, Ising |
| `PottsLabels` | labels `1:q` on a graph | spatial clustering, Potts |

`potts_pseudologlik` and `autologistic_pseudologlik` give the interaction strength a
differentiable stand-in for the intractable likelihood; `grid_neighbours` and
`distance_neighbours` build the graphs.

```julia
using PracticalBayes, PracticalEcoBayes, Distributions
using LogExpFunctions: logistic

# single-species occupancy: y is sites × visits (missing = no visit)
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
        v = [(i, j) for (i, j) in visits if c.values.z[i]]   # visits to occupied sites
        (ones(length(v), 1), [y[i, j] for (i, j) in v])
    end)
chn = sample(occupancy(y), spl, 2000)
```

For multi-species occupancy with species interactions (Rota et al. 2016), see
[MultispeciesOccupancy.jl](https://github.com/EvoArt/MultispeciesOccupancy.jl), which is
built on these kernels.
