# Pólya–Gamma draws and the Pólya–Gamma logistic-regression kernel
# (Polson, Scott & Windle 2013, JASA 108:1339). PG(1, c) is drawn by Devroye's
# alternating-series method for J*(1, c/2), as in the reference BayesLogit code;
# PG(b, c) for integer b is a sum of b such draws.

const _PG_T = 0.64           # truncation point of the two-piece proposal
const _PG_INV_T = 1 / _PG_T

# Coefficient n of the alternating series for the J*(1, 0) density at x
function _pg_a(n::Int, x::Float64)
    K = (n + 0.5) * π
    if x > _PG_T
        return K * exp(-0.5 * K^2 * x)
    elseif x > 0
        return exp(-1.5 * (log(0.5π) + log(x)) + log(K) - 2 * (n + 0.5)^2 / x)
    end
    return 0.0
end

# Probability of the exponential (right) piece of the proposal
function _pg_mass_texpon(z::Float64)
    t = _PG_T
    fz = 0.125 * π^2 + 0.5 * z^2
    b = sqrt(1 / t) * (t * z - 1)
    a = -sqrt(1 / t) * (t * z + 1)
    x0 = log(fz) + fz * t
    xb = x0 - z + logcdf(Normal(), b)
    xa = x0 + z + logcdf(Normal(), a)
    return 1 / (1 + 4 / π * (exp(xb) + exp(xa)))
end

# Inverse-Gaussian(1/z, 1) truncated to (0, t)
function _pg_rtigauss(rng::AbstractRNG, z::Float64)
    t = _PG_T
    x = t + 1
    if _PG_INV_T > z                       # mean 1/z beyond t: chi-square proposal
        α = 0.0
        while rand(rng) > α
            e1, e2 = randexp(rng), randexp(rng)
            while e1^2 > 2 * e2 / t          # (not `2e2`, which Julia reads as 200.0)
                e1, e2 = randexp(rng), randexp(rng)
            end
            x = t / (1 + e1 * t)^2
            α = exp(-0.5 * z^2 * x)
        end
    else
        μ = 1 / z
        while x > t
            y = randn(rng)^2
            μy = μ * y
            x = μ + 0.5μ * μy - 0.5μ * sqrt(4μy + μy^2)
            rand(rng) > μ / (μ + x) && (x = μ^2 / x)
        end
    end
    return x
end

function _pg1(rng::AbstractRNG, c::Real)
    z = 0.5 * abs(Float64(c))
    fz = 0.125 * π^2 + 0.5 * z^2
    while true
        x = rand(rng) < _pg_mass_texpon(z) ? _PG_T + randexp(rng) / fz : _pg_rtigauss(rng, z)
        s = _pg_a(0, x)
        y = rand(rng) * s
        n = 0
        while true
            n += 1
            if isodd(n)
                s -= _pg_a(n, x)
                y <= s && return 0.25 * x
            else
                s += _pg_a(n, x)
                y > s && break
            end
        end
    end
end

"""
    rand_polyagamma([rng], b::Integer, c::Real)

One draw from the Pólya–Gamma distribution PG(b, c), the latent variable of the
Polson–Scott–Windle data augmentation for logistic models. Its mean is
`b / (2c) * tanh(c / 2)` (`b / 4` at `c = 0`).
"""
function rand_polyagamma(rng::AbstractRNG, b::Integer, c::Real)
    b >= 0 || throw(ArgumentError("b must be non-negative"))
    s = 0.0
    for _ in 1:b
        s += _pg1(rng, c)
    end
    return s
end
rand_polyagamma(b::Integer, c::Real) = rand_polyagamma(Random.default_rng(), b, c)

"""
    PolyaGammaLogit(data, name; prior_mean=0, prior_sd=2.5)

A Gibbs kernel that draws the coefficient vector `name` of a logistic (binomial)
regression exactly, by Pólya–Gamma data augmentation:

    y[i] ~ Binomial(n[i], logistic(X[i, :]'β + offset[i])),   β ~ Normal(prior_mean, prior_sd)

`data(c)` receives PracticalBayes' `ModelConditional`
and returns `(X, y)`, `(X, y, n)` or `(X, y, n, offset)` for the current state, so
the rows can depend on other variables (for detection given occupancy, only the
visits to occupied sites). `n` defaults to 1 (Bernoulli). The model's prior on
`name` should match `prior_mean` and `prior_sd` (numbers or vectors).

```julia
pk = PolyaGammaLogit(:α) do c
    occupied = c.values.z[site_of_visit]
    (Xdet[occupied, :], ydet[occupied])
end
spl = Gibbs(:α => pk, :z => OccupancyStates(...), :β => NUTS())
```
"""
struct PolyaGammaLogit{F,M,S} <: AbstractGibbsKernel
    name::Symbol
    data::F
    prior_mean::M
    prior_sd::S
end
PolyaGammaLogit(data, name::Symbol; prior_mean=0.0, prior_sd=2.5) =
    PolyaGammaLogit(name, data, prior_mean, prior_sd)

_unpack_logit(d::Tuple{Any,Any}) = (d[1], d[2], 1, 0.0)
_unpack_logit(d::Tuple{Any,Any,Any}) = (d[1], d[2], d[3], 0.0)
_unpack_logit(d::Tuple{Any,Any,Any,Any}) = d

"""
    polyagamma_logit_draw([rng], X, y, β; n=1, offset=0, prior_mean=0, prior_sd=2.5)

One Pólya–Gamma Gibbs update of logistic-regression coefficients: given the
current `β`, draw the augmentation variables and return a new `β` from its exact
Gaussian conditional, for `y[i] ~ Binomial(n[i], logistic(X[i, :]'β + offset[i]))`
and independent `Normal(prior_mean, prior_sd)` priors. The engine of
[`PolyaGammaLogit`](@ref), for kernels that update several regressions at once.
"""
function polyagamma_logit_draw(rng::AbstractRNG, X::AbstractMatrix, y::AbstractVector, β::AbstractVector;
                               n=1, offset=0.0, prior_mean=0.0, prior_sd=2.5)
    p = length(β)
    size(X, 2) == p || throw(DimensionMismatch("X has $(size(X, 2)) columns for $p coefficients"))
    size(X, 1) == length(y) || throw(DimensionMismatch("$(size(X, 1)) rows of X for $(length(y)) responses"))
    nv = n isa Number ? fill(Int(n), size(X, 1)) : Int.(n)
    ov = offset isa Number ? fill(Float64(offset), size(X, 1)) : Float64.(offset)
    η = X * Float64.(β) .+ ov
    ω = [rand_polyagamma(rng, nv[i], η[i]) for i in eachindex(η)]
    κ = Float64.(y) .- nv ./ 2
    m0 = prior_mean isa Number ? fill(Float64(prior_mean), p) : Float64.(prior_mean)
    v0 = prior_sd isa Number ? fill(Float64(prior_sd)^2, p) : Float64.(prior_sd) .^ 2
    C = cholesky(Symmetric(X' * (ω .* X) + Diagonal(1 ./ v0)))
    mean = C \ (X' * (κ .- ω .* ov) .+ m0 ./ v0)
    return mean .+ (C.U \ randn(rng, p))         # covariance (X'ΩX + V₀⁻¹)⁻¹
end
polyagamma_logit_draw(X::AbstractMatrix, y::AbstractVector, β::AbstractVector; kw...) =
    polyagamma_logit_draw(Random.default_rng(), X, y, β; kw...)

function PracticalBayes.gibbs_step(rng, k::PolyaGammaLogit, names, c::ModelConditional)
    names == (k.name,) || error("PolyaGammaLogit owns block $(k.name), got $names")
    X, y, n, offset = _unpack_logit(k.data(c))
    current = getfield(c.values, k.name)
    draw = polyagamma_logit_draw(rng, X, y, vec(collect(Float64, current)); n, offset,
                                 prior_mean=k.prior_mean, prior_sd=k.prior_sd)
    return NamedTuple{(k.name,)}((_like(current, draw),))
end

# Return the draw in the shape of the current value (a vector, or a scalar for p = 1)
_like(x::Number, draw) = only(draw)
_like(x::AbstractVector, draw) = draw
_like(x::AbstractArray, draw) = reshape(draw, size(x))
