"""
    KMarkovEnvironment{OV, M<:POMDP, S, R<:AbstractRNG} <: AbstractEnvironment
A k-markov wrapper for MDPs and POMDPs, given a MDP or POMDP create an AbstractEnvironment where s_t = (o_t, ..., o_t-k)
The K-Markov observation is represented by a vector of k observations.
"""
mutable struct KMarkovEnvironment{OV, M<:POMDP, S, R<:AbstractRNG} <: AbstractEnvironment
    problem::M
    k::Int64
    state::S
    obs::Vector{OV}
    rng::R
end
function KMarkovEnvironment(problem::M,
                            ov::Type{A} = obsvector_type(problem);
                            k::Int64=1, rng::AbstractRNG=MersenneTwister(0)) where {A<:AbstractArray, M, R<:AbstractRNG}
    # determine size of obs vector
    s = initialstate(problem, rng)
    o = gen(DDNOut{:o}, problem, s, rng)
    obs = convert_o(ov, o, problem)
    # init vector of obs
    obsvec = fill(zeros(eltype(ov), size(obs)...), k)
    return KMarkovEnvironment(problem, k, initialstate(problem, rng), 
                              obsvec, rng)
end

"""
    reset!(env::KMarkovEnvironment{OV})
Reset an POMDP environment by sampling an initial state,
generating an observation and returning it.
"""
function reset!(env::KMarkovEnvironment{OV}) where OV
    s = initialstate(env.problem, env.rng)
    env.state = s
    o = gen(DDNOut(:o), env.problem, s, env.rng)
    obs = convert_o(OV, o, env.problem)
    fill!(env.obs, obs)
    return env.obs
end

"""
    step!(env::POMDPEnvironment{OV}, a::A)
Take in an POMDP environment, and an action to execute, and
step the environment forward. Return the observation, reward,
terminal flag and info
"""
function step!(env::KMarkovEnvironment{OV}, a::A) where {OV, A}
    s, o, r, info = gen(DDNOut(:sp, :o, :r, :info), env.problem, env.state, a, env.rng)
    env.state = s
    t = isterminal(env.problem, s)
    info = nothing
    obs = convert_o(OV, o, env.problem)
    # shift the old observation to lower indices
    for i=1:env.k-1
        env.obs[i] = env.obs[i + 1]
    end
    env.obs[env.k] = obs
    return env.obs, r, t
end

"""
    actions(env::KMarkovEnvironment)
Return an action object that can be sampled with rand.
"""
function POMDPs.actions(env::KMarkovEnvironment)
    return actions(env.problem)
end

"""
    sample_action(env::Union{POMDPEnvironment, MDPEnvironment})
Sample an action from the action space of the environment.
"""
function sample_action(env::KMarkovEnvironment)
    return rand(env.rng, actions(env))
end


"""
    obs_dimensions(env::KMarkovEnvironment{OV})
returns the size of the observation vector.
The object return by `step!` and `reset!` is a vector of k observation vector of size `obs_dimensions(env)`
It generates an initial state, converts it to an array and returns its size.
"""
function obs_dimensions(env::KMarkovEnvironment{OV}) where OV
    obs_dim = size(convert_o(OV, gen(DDNOut(:o), env.problem, initialstate(env.problem, env.rng), env.rng), env.problem))
    return (obs_dim..., env.k)
end
