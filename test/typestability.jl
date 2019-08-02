using Revise
using Profile
using ProfileView
using RLInterface
using POMDPs
using POMDPModels
using Test
using Random
using BenchmarkTools
using POMDPModelTools

function sim(env, nsteps=100, rng=MersenneTwister(0))
    o = reset!(env)
    step = 1
    done = false
    r_tot = 0.0
    na = n_actions(env)
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

@code_warntype step!(env, :up)

Profile.clear()
@profile for i=1:1000; sim(env); end

ProfileView.view()