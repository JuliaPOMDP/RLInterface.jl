using Revise
# using Profile
# using ProfileView
using RLInterface
using POMDPs
using POMDPModels
using Test
using Random
# using BenchmarkTools
# using POMDPModelTools

function sim(env, nsteps=100, rng=MersenneTwister(0))
    o = reset!(env)
    step = 1
    done = false
    r_tot = 0.0
    na = length(actions(env))
    dims = obs_dimensions(env)
    while !done && step <= nsteps
        action = sample_action(env)
        obs, rew, done, info = step!(env, action)
        r_tot += rew
        step += 1
    end
    return r_tot
end


env = MDPEnvironment(SimpleGridWorld())
# env = POMDPEnvironment(TigerPOMDP())

sim(env)

@inferred sim(env)

@btime sim($env, $4)

@code_warntype sim(env)

@inferred reset!(env)

@inferred step!(env, :up)

@code_warntype step!(env, :up)

@code_warntype gen(DDNOut(:sp, :r), env.problem, env.state, :up, env.rng)


@inferred gen(DDNOut(:sp, :r), env.problem, env.state, :up, env.rng)

function teststep(env, a)
    s, r = gen(DDNOut(:sp, :r), env.problem, env.state, :up, env.rng)
    return (s, r)
end

@inferred teststep(env, :up)

@code_warntype teststep(env, :up)


using POMDPs, POMDPModels, Random, Test

mdp = SimpleGridWorld()
rng = MersenneTwister(1)
s = initialstate(mdp, rng)
a = :up

@inferred gen(DDNOut(:sp, :r), mdp, s, a, rng)

struct MDPEnv{M, S, R}
    prob::M
    s::S
    rng::R
end

function step(env::MDPEnv, a)
    return gen(DDNOut(:sp, :r), env.prob, env.s, a, env.rng)
end

env = MDPEnv(mdp, s, rng)

@inferred step(env, a)

@code_warntype step(env, a)







# Profile.clear()
# @profile for i=1:1000; sim(env); end

# ProfileView.view()