using Test, Statistics, StableRNGs
using PracticalBayes, PracticalEcoBayes
using Distributions: Normal, Bernoulli, Poisson, Binomial, Categorical, logpdf
using LinearAlgebra: diag, inv
using LogExpFunctions: logistic, logsumexp

# Exact mean and variance of PG(b, c)
pg_mean(b, c) = c == 0 ? b / 4 : b / (2c) * tanh(c / 2)
pg_var(b, c) = c == 0 ? b / 24 : b * (sinh(c) - c) / (4c^3 * cosh(c / 2)^2)

# Empirical distribution of a block across a chain of Gibbs sweeps
draws_of(chn, name) = vec(chn[name])

@testset "PracticalEcoBayes" begin

@testset "Pólya–Gamma draws" begin
    rng = StableRNG(1)
    for b in (1, 3), c in (0.0, 0.7, 2.5, 12.0, -4.0)
        x = [rand_polyagamma(rng, b, c) for _ in 1:40_000]
        @test all(>(0), x)
        @test mean(x) ≈ pg_mean(b, abs(c)) rtol = 0.02
        @test var(x) ≈ pg_var(b, abs(c)) rtol = 0.08
    end
end

@testset "README Julia examples" begin
    readme = read(joinpath(@__DIR__, "..", "README.md"), String)
    docs_home = read(joinpath(@__DIR__, "..", "docs", "src", "index.md"), String)
    examples = [m.captures[1] for m in eachmatch(r"(?ms)^```julia\s*\n(.*?)^```\s*$", readme)]
    docs_examples = [m.captures[1] for m in eachmatch(r"(?ms)^```julia\s*\n(.*?)^```\s*$", docs_home)]
    @test !isempty(examples)
    @test examples == docs_examples
    mktempdir() do dir
        for (i, code) in enumerate(examples)
            path = joinpath(dir, "readme_example_$i.jl")
            write(path, code)
            example = Module(Symbol("ReadmeExample", i))
            Base.include(example, path)
            @test isdefined(example, :chn)
        end
    end
end

@testset "PolyaGammaLogit recovers logistic coefficients" begin
    rng = StableRNG(2)
    n = 800
    X = hcat(ones(n), randn(rng, n))
    βtrue = [-0.5, 1.2]
    y = rand(rng, n) .< logistic.(X * βtrue)
    @model function logit(X, y)
        β ~ filldist(Normal(0, 2.5), 2)
        for i in eachindex(y)
            y[i] ~ Bernoulli(logistic(X[i, 1] * β[1] + X[i, 2] * β[2]))
        end
    end
    spl = Gibbs(:β => PolyaGammaLogit(c -> (X, y), :β))
    chn = sample(rng, logit(X, y), spl, 1500; discard_initial=300, progress=false)
    β̂ = mean(draws_of(chn, :β))
    # this data set's maximum-likelihood estimate (IRLS); with a weak prior and
    # n = 800 the posterior mean sits very close to it
    mle = zeros(2)
    for _ in 1:25
        μ = logistic.(X * mle)
        mle .+= (X' * (μ .* (1 .- μ) .* X)) \ (X' * (y .- μ))
    end
    @test β̂ ≈ mle atol = 0.04
    # posterior sd close to the asymptotic one from the Fisher information
    p = logistic.(X * mle)
    se = sqrt.(diag(inv(X' * (p .* (1 .- p) .* X))))
    @test std(getindex.(draws_of(chn, :β), 2)) ≈ se[2] rtol = 0.25
end

@testset "OccupancyStates matches the exact conditional" begin
    rng = StableRNG(3)
    ψ, p = 0.6, 0.3
    y = Union{Missing,Int}[1 0 0; 0 0 0; 0 missing 0; 0 0 0]
    @model function occ(y, ψ, p)
        z = Vector{Bool}(undef, size(y, 1))
        for i in axes(y, 1)
            z[i] ~ Bernoulli(ψ)
        end
    end
    spl = Gibbs(:z => OccupancyStates(c -> (fill(ψ, 4), fill(p, 4, 3), y), :z))
    chn = sample(rng, occ(y, ψ, p), spl, 20_000; progress=false)
    zz = reduce(hcat, draws_of(chn, :z))
    @test all(zz[1, :])                                     # detected: occupied
    exact(J) = ψ * (1 - p)^J / (ψ * (1 - p)^J + 1 - ψ)
    @test mean(zz[2, :]) ≈ exact(3) atol = 0.015
    @test mean(zz[3, :]) ≈ exact(2) atol = 0.015            # one missed visit
end

@testset "NMixtureAbundance matches the exact conditional" begin
    rng = StableRNG(4)
    λ, p = 4.0, 0.4
    y = [2 1 3; 0 0 0]
    @model function nmix(y, λ)
        N = Vector{Int}(undef, size(y, 1))
        for i in axes(y, 1)
            N[i] ~ Poisson(λ)
        end
    end
    spl = Gibbs(:N => NMixtureAbundance(c -> (fill(λ, 2), fill(p, 2, 3), y), :N))
    chn = sample(rng, nmix(y, λ), spl, 20_000; progress=false)
    NN = reduce(hcat, draws_of(chn, :N))
    for i in 1:2
        support = maximum(y[i, :]):40
        lp = [logpdf(Poisson(λ), n) + sum(logpdf(Binomial(n, p), y[i, j]) for j in 1:3) for n in support]
        w = exp.(lp .- logsumexp(lp))
        @test mean(NN[i, :]) ≈ sum(support .* w) atol = 0.06
        @test minimum(NN[i, :]) >= maximum(y[i, :])
    end
end

@testset "neighbour graphs" begin
    nb = grid_neighbours(2, 3)
    @test length(nb) == 6
    @test sort(nb.adjacency[1]) == [2, 3]                     # (1,1): below and right
    @test length(grid_neighbours(3, 3; diagonal=true).adjacency[5]) == 8
    @test distance_neighbours([0 0; 1 0; 5 5]; radius=1.5).adjacency == [[2], [1], Int[]]
    @test_throws ArgumentError Neighbours([[2], Int[]])     # not symmetric
    A = [0 2 0; 2 0 1; 0 1 0]
    @test Neighbours(A).weights[2] == [2.0, 1.0]
end

# a 3-node path graph 1–2–3, small enough to enumerate
path = Neighbours([[2], [1, 3], [2]])

@testset "PottsLabels samples the Potts distribution" begin
    rng = StableRNG(5)
    β, q = 0.8, 3
    h = [0.5 0.0 -0.5; 0.0 0.0 0.0; -1.0 0.3 0.0]
    @model function potts(n, q)
        x = Vector{Int}(undef, n)
        for i in 1:n
            x[i] ~ Categorical(fill(1 / q, q))
        end
    end
    spl = Gibbs(:x => PottsLabels(c -> (β, h), :x, path; q))
    chn = sample(rng, potts(3, q), spl, 60_000; discard_initial=100, progress=false)
    xs = draws_of(chn, :x)
    states = [(a, b, c) for a in 1:q, b in 1:q, c in 1:q]
    energy(s) = sum(h[i, s[i]] for i in 1:3) + β * ((s[1] == s[2]) + (s[2] == s[3]))
    lp = vec(energy.(states))
    exact = exp.(lp .- logsumexp(lp))
    freq = [count(x -> Tuple(x) == s, xs) / length(xs) for s in vec(states)]
    @test maximum(abs.(freq .- exact)) < 0.012
    # pseudo-likelihood against a hand calculation at one state
    x = [1, 1, 2]
    manual = sum(begin
        lp_i = [h[i, s] + β * count(j -> x[j] == s, path.adjacency[i]) for s in 1:q]
        lp_i[x[i]] - logsumexp(lp_i)
    end for i in 1:3)
    @test potts_pseudologlik(x, β, path; h, q) ≈ manual
end

@testset "IsingStates samples the autologistic distribution" begin
    rng = StableRNG(6)
    h, J = [0.3, -0.5, 1.0], 1.1
    @model function ising(n)
        z = Vector{Bool}(undef, n)
        for i in 1:n
            z[i] ~ Bernoulli(0.5)
        end
    end
    spl = Gibbs(:z => IsingStates(c -> (h, J), :z, path))
    chn = sample(rng, ising(3), spl, 60_000; discard_initial=100, progress=false)
    zs = draws_of(chn, :z)
    states = vec([Bool[a, b, c] for a in (false, true), b in (false, true), c in (false, true)])
    lp = [sum(h .* s) + J * (s[1] * s[2] + s[2] * s[3]) for s in states]
    exact = exp.(lp .- logsumexp(lp))
    freq = [count(==(s), zs) / length(zs) for s in states]
    @test maximum(abs.(freq .- exact)) < 0.012
    z = Bool[1, 0, 1]
    manual = sum(begin
        η = h[i] + J * sum(z[j] for j in path.adjacency[i])
        z[i] ? log(logistic(η)) : log(1 - logistic(η))
    end for i in 1:3)
    @test autologistic_pseudologlik(z, h, J, path) ≈ manual
end

end
