using RLInterface
using POMDPModels
using Test
using Random

include("zmq.jl")

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

@testset "Sim" begin
    envs = [MDPEnvironment(SimpleGridWorld()),
            MDPEnvironment(InvertedPendulum()),
            MDPEnvironment(MountainCar()),
            POMDPEnvironment(TMaze()),
            POMDPEnvironment(BabyPOMDP()),
            POMDPEnvironment(TigerPOMDP()),
            KMarkovEnvironment(TMaze(), k=4),
            KMarkovEnvironment(BabyPOMDP(), k=4),
            KMarkovEnvironment(TigerPOMDP(), k=4)]

    for env in envs
        r = sim(env)
        process(env)
    end
end

if VERSION >= v"1.1"
    @testset "type stability" begin
        env = MDPEnvironment(SimpleGridWorld())
        @inferred reset!(env)
        @inferred step!(env, :up)
        @inferred sim(env)

        env = POMDPEnvironment(TigerPOMDP())
        @inferred reset!(env)
        @inferred step!(env, 0)
        @inferred sim(env)

        env = KMarkovEnvironment(TigerPOMDP(), k=4)
        @inferred reset!(env)
        @inferred step!(env, 0)
        @inferred sim(env)
    end
end
