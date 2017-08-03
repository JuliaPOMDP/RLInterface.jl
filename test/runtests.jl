using DeepRL
using POMDPModels
using Base.Test

function sim(env, nsteps=100, rng=MersenneTwister(0))
    o = reset(env)
    step = 1
    done = false
    r_tot = 0.0
    na = n_actions(env)
    dims = obs_dimensions(env)
    while !done && step <= nsteps
        action = sample_action(env)
        obs, rew, done, info = step!(env, action)
        #println(obs, " ", rew, " ", done, " ", info)
        r_tot += rew
        step += 1
    end
    return r_tot
end

envs = [MDPEnvironment(GridWorld()),
        MDPEnvironment(InvertedPendulum()),
        MDPEnvironment(MountainCar()),
        POMDPEnvironment(TMaze()),
        POMDPEnvironment(BabyPOMDP()),
        POMDPEnvironment(TigerPOMDP())]

for env in envs
    r = sim(env)
end


