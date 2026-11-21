"""
    PracticalEcoBayes

Gibbs kernels for ecological models, to plug into [PracticalBayes](https://github.com/EvoArt/PracticalBayes.jl)'s
`Gibbs` sampler:

- [`PolyaGammaLogit`](@ref): exact draws of logistic-regression coefficients by
  Pólya–Gamma augmentation (occupancy, detection, presence/absence models);
  [`rand_polyagamma`](@ref) draws the augmentation variables themselves.
- [`OccupancyStates`](@ref), [`NMixtureAbundance`](@ref): the latent presence and
  abundance of hierarchical detection models.
- [`PottsLabels`](@ref), [`IsingStates`](@ref): Markov random fields on a neighbour
  graph ([`Neighbours`](@ref), [`grid_neighbours`](@ref), [`distance_neighbours`](@ref)),
  for spatial clustering and autologistic occupancy, with pseudo-likelihoods
  [`potts_pseudologlik`](@ref) and [`autologistic_pseudologlik`](@ref) for the
  interaction strength.

Each kernel takes a function of the current `ModelConditional` that returns the
quantities it needs, so it works with any model layout.
"""
module PracticalEcoBayes

using PracticalBayes
using PracticalBayes: AbstractGibbsKernel, ModelConditional
using Distributions: Normal, Poisson, Binomial, logpdf, logcdf
using LinearAlgebra: Symmetric, Diagonal, cholesky
using LogExpFunctions: logistic, logsumexp, log1pexp
using Random: Random, AbstractRNG, randexp, randperm

export rand_polyagamma, polyagamma_logit_draw, PolyaGammaLogit
export OccupancyStates, NMixtureAbundance
export Neighbours, grid_neighbours, distance_neighbours
export PottsLabels, potts_pseudologlik, IsingStates, autologistic_pseudologlik

include("latent.jl")
include("polyagamma.jl")
include("potts.jl")

end
