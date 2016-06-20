# DeepRL

This package provides an interface for working with deep reinfrocement learning problems in Julia.
It is closely intergrated [POMDPs.jl]() and [GenerativeModels.jl]() to easily wrap problems define in those formats. 
While the focus of this interface is on partially observable Markov decision process (POMDP) reinforcement learning, it
is flexible and can easily handle problems that are fully observable as well. 
The interface is very similar to that of [OpenAI Gym](). This allows algorithms that work with Gym to be used with problems that
are defined in this interface and vica versa.
A shared interface between POMDPs.jl allows easy comparison of reinforcement learning solutions to approximate dynamic
porgramming solutions when a complete model of the problem is defined. 


## Interface

The interface prvoides an `AbstractEnvironment` type from which all custom environemnts
should inherit. For an example see how this is done with [OpenAI Gym](). 

Running a simulation can be done like so, we use a problem from [POMDPModels]() as an example:

```julia
o = reset(env)
step = 1
done = false
r_tot = 0.0
na = n_actions(env)
dims = obs_dimensions(env)
while !done && step <= nsteps
    action = sample_action(env)
    obs, rew, done, info = step!(env, action)
    println(obs, " ", rew, " ", done, " ", info)
    r_tot += rew
    step += 1
end
```

