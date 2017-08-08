#__precompile__()


module DeepRL

using POMDPs


export 
    # Environment types
    AbstractEnvironment,
    POMDPEnvironment,
    MDPEnvironment,
    # supporting methods
    reset,
    step!,
    actions,
    sample_action,
    n_actions,
    obs_dimensions,
    render


abstract type AbstractEnvironment end

mutable struct MDPEnvironment{S} <: AbstractEnvironment
    problem::MDP 
    state::S
    rng::AbstractRNG
end
function MDPEnvironment(problem::MDP; rng::AbstractRNG=MersenneTwister(0))
    return MDPEnvironment(problem, initial_state(problem, rng), rng)
end

mutable struct POMDPEnvironment{S} <: AbstractEnvironment
    problem::POMDP 
    state::S
    rng::AbstractRNG
end
function POMDPEnvironment(problem::POMDP; rng::AbstractRNG=MersenneTwister(0))
    return POMDPEnvironment(problem, initial_state(problem, rng), rng)
end

"""
    reset(env::MDPEnvironment)
Reset an MDP environment by sampling an initial state returning it.
"""
function Base.reset(env::MDPEnvironment)
    s = initial_state(env.problem, env.rng)
    env.state = s
    return convert(Array{Float64}, s, env.problem)
end

"""
    reset(env::POMDPEnvironment)
Reset an POMDP environment by sampling an initial state, 
generating an observation and returning it.
"""
function Base.reset(env::POMDPEnvironment)
    s = initial_state(env.problem, env.rng)
    env.state = s
    o = generate_o(env.problem, s, env.rng)
    return convert(Array{Float64, 1}, o, env.problem)
end


"""
    step!{A}(env::POMDPEnvironment, a::A)
Take in an POMDP environment, and an action to execute, and 
step the environment forward. Return the state, reward, 
terminal flag and info
"""
function step!{A}(env::MDPEnvironment, a::A)
    s, r = generate_sr(env.problem, env.state, a, env.rng)
    env.state = s
    t = isterminal(env.problem, s)
    info = nothing
    obs = convert(Array{Float64}, s, env.problem)
    return obs, r, t, info
end

"""
    step!{A}(env::MDPEnvironment, a::A)
Take in an MDP environment, and an action to execute, and 
step the environment forward. Return the observation, reward, 
terminal flag and info
"""
function step!{A}(env::POMDPEnvironment, a::A)
    s, o, r = generate_sor(env.problem, env.state, a, env.rng)
    env.state = s
    t = isterminal(env.problem, s)
    info = nothing
    obs = convert(Array{Float64}, o, env.problem)
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


function obs_dimensions(env::MDPEnvironment)
    return size(convert(Array{Float64}, initial_state(env.problem, env.rng), env.problem))
end


function obs_dimensions(env::POMDPEnvironment)
    return size(convert(Array{Float64}, generate_o(env.problem, initial_state(env.problem, env.rng), env.rng), env.problem))
end

"""
    render(env::AbstractEnvironment)
Renders a graphic of the environment
"""
function render(env::AbstractEnvironment) end

end # module
