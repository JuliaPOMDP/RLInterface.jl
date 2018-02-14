# a k-markov wrapper for MDPs and POMDPs
# given a MDP or POMDP create an AbstractEnvironment where s_t = (o_t, ..., o_t-k)

mutable struct KMarkovEnvironment{S} <: AbstractEnvironment
    problem::POMDP
    k::Int64
    state::S
    obs::Array{Float64}
    rng::AbstractRNG
end
function KMarkovEnvironment(problem::POMDP; k::Int64=1, rng::AbstractRNG=MersenneTwister(0))
    return KMarkovEnvironment(problem, k, initial_state(problem, rng), zeros(k), rng)
end

"""
    reset(env::KMarkovEnvironment)
Reset an POMDP environment by sampling an initial state,
generating an observation and returning it.
"""
function Base.reset(env::KMarkovEnvironment)
    s = initial_state(env.problem, env.rng)
    env.state = s
    o = generate_o(env.problem, s, env.rng)
    obs = convert_o(Array{Float64, 1}, o, env.problem)
    # build a matrix of size (obs_dim..., k)
    obs_stacked = zeros(size(obs)..., env.k)
    for i=1:env.k
        obs_stacked[Base.setindex(indices(obs_stacked), i, ndims(obs_stacked))...] = obs
    end
    env.obs = obs_stacked
    return env.obs
end

"""
    step!{A}(env::POMDPEnvironment, a::A)
Take in an POMDP environment, and an action to execute, and
step the environment forward. Return the observation, reward,
terminal flag and info
"""
function step!(env::KMarkovEnvironment, a::A) where A
    s, o, r = generate_sor(env.problem, env.state, a, env.rng)
    env.state = s
    t = isterminal(env.problem, s)
    info = nothing
    obs = convert_o(Array{Float64, 1}, o, env.problem)
    # build a matrix of size (k, obs_dim..) , shift the old observation to lower indices
    stack_obs = zeros(size(env.obs))
    for i=1:env.k-1
        stack_obs[Base.setindex(indices(env.obs), i, ndims(env.obs))...] = slicedim(env.obs, ndims(env.obs), i+1)
    end
    stack_obs[Base.setindex(indices(env.obs), env.k, ndims(env.obs))...] = obs
    env.obs = stack_obs
    return stack_obs, r, t, info
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
    n_actions(env::KMarkovEnvironment)
Return the number of actions in the environment (environments with discrete action spaces only)
"""
function POMDPs.n_actions(env::KMarkovEnvironment)
    return n_actions(env.problem)
end

function obs_dimensions(env::KMarkovEnvironment)
    obs_dim = size(convert_o(Array{Float64,1}, generate_o(env.problem, initial_state(env.problem, env.rng), env.rng), env.problem))
    return (env.k, obs_dim...)
end
