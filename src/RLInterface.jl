module RLInterface

using POMDPs
using POMDPModelTools

# for the ZMQ part
using ZMQ
using JSON
using Random


export
    # Environment types
    AbstractEnvironment,
    POMDPEnvironment,
    MDPEnvironment,
    KMarkovEnvironment,
    # supporting methods
    reset!,
    step!,
    actions,
    sample_action,
    n_actions,
    obs_dimensions,
    render,
    # deprecated
    reset


abstract type AbstractEnvironment end

"""
    obsvector_type(::Union{MDP, POMDP})

Returns the type of the observation vector associated with a specific problem. 
The `MDPEnvironment` and `POMDPEnvironment` wrappers will convert observations to an object of such type when `reset!` or `step!` is called.
"""
obsvector_type(::Union{MDP, POMDP}) = Vector{Float32}

mutable struct MDPEnvironment{OV, M<:MDP, S, R<:AbstractRNG} <: AbstractEnvironment 
    problem::M
    state::S
    rng::R
end
function MDPEnvironment(problem::M,
                        ov::Type{A} = obsvector_type(problem);
                        rng::R=MersenneTwister(0)) where {A<:AbstractArray, M<:MDP, R<:AbstractRNG}
    S = statetype(problem)
    return MDPEnvironment{ov, M, S, R}(problem, initialstate(problem, rng), rng)
end

mutable struct POMDPEnvironment{OV, M<:POMDP, S, R<:AbstractRNG} <: AbstractEnvironment
    problem::M
    state::S
    rng::R
end
function POMDPEnvironment(problem::M,
                          ov::Type{A} = obsvector_type(problem);
                          rng::R=MersenneTwister(0)) where {A<:AbstractArray, M<:POMDP, R<:AbstractRNG}
    S = statetype(problem)
    return POMDPEnvironment{ov, M, S, R}(problem, initialstate(problem, rng), rng)
end

"""
    reset!(env::MDPEnvironment{OV})
Reset an MDP environment by sampling an initial state returning it.
"""
function reset!(env::MDPEnvironment{OV}) where OV
    s = initialstate(env.problem, env.rng)
    env.state = s
    return convert_s(OV, s, env.problem)
end

"""
    reset!(env::POMDPEnvironment{OV})
Reset an POMDP environment by sampling an initial state,
generating an observation and returning it.
"""
function reset!(env::POMDPEnvironment{OV}) where OV
    s = initialstate(env.problem, env.rng)
    env.state = s
    a = first(actions(env))
    o = generate_o(env.problem, s, a, s, env.rng)
    return convert_o(OV, o, env.problem)
end

"""
    step!(env::MDPEnvironment{OV}, a::A)
Take in an POMDP environment, and an action to execute, and
step the environment forward. Return the state, reward,
terminal flag and info
"""
function step!(env::MDPEnvironment{OV}, a::A) where {OV, A}
    s, r, info = generate_sri(env.problem, env.state, a, env.rng)
    env.state = s
    t = isterminal(env.problem, s)
    obs = convert_s(OV, s, env.problem)
    return obs, r, t, info
end

"""
    step!(env::POMDPEnvironment{OV}, a::A)
Take in an MDP environment, and an action to execute, and
step the environment forward. Return the observation, reward,
terminal flag and info
"""
function step!(env::POMDPEnvironment{OV}, a::A) where {OV, A}
    s, o, r, info = generate_sori(env.problem, env.state, a, env.rng)
    env.state = s
    t = isterminal(env.problem, s)
    obs = convert_o(OV, o, env.problem)
    return obs, r, t, info
end

"""
    actions(env::Union{POMDPEnvironment, MDPEnvironment})
Return an action object that can be sampled with rand.
"""
function POMDPs.actions(env::Union{POMDPEnvironment, MDPEnvironment})
    return actions(env.problem)
end

"""
    sample_action(env::Union{POMDPEnvironment, MDPEnvironment})
Sample an action from the action space of the environment.
"""
function sample_action(env::Union{POMDPEnvironment, MDPEnvironment})
    return rand(env.rng, actions(env))
end

"""
    n_actions(env::Union{POMDPEnvironment, MDPEnvironment})
Return the number of actions in the environment (environments with discrete action spaces only)
"""
function POMDPs.n_actions(env::Union{POMDPEnvironment, MDPEnvironment})
    return n_actions(env.problem)
end

"""
    obs_dimensions(env::MDPEnvironment{OV}) where OV
returns the size of the observation vector.
It generates an initial state, converts it to an array and returns its size.
"""
function obs_dimensions(env::MDPEnvironment{OV}) where OV
    return size(convert_s(OV, initialstate(env.problem, env.rng), env.problem))
end

"""
    obs_dimensions(env::POMDPEnvironment{OV}) where OV
returns the size of the observation vector.
It generates an initial observation, converts it to an array and returns its size.
"""
function obs_dimensions(env::POMDPEnvironment{OV}) where OV
    s = initialstate(env.problem, env.rng)
    a = first(actions(env))
    return size(convert_o(OV, generate_o(env.problem, s,a,s, env.rng), env.problem))
end

"""
    render(env::AbstractEnvironment)
Renders a graphic of the environment
"""
function render(env::AbstractEnvironment) end

include("ZMQServer.jl")
include("k_markov.jl")

# deprecations
import Base.reset
@deprecate reset(env::KMarkovEnvironment) reset!(env)
@deprecate reset(env::POMDPEnvironment) reset!(env)
@deprecate reset(env::MDPEnvironment) reset!(env)

end # module
