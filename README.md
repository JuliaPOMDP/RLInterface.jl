***THIS PACKAGE IS DEPRECATED IN FAVOR OF [CommonRLInterface](https://github.com/JuliaReinforcementLearning/CommonRLInterface.jl)

# RLInterface

[![Build Status](https://travis-ci.org/JuliaPOMDP/RLInterface.jl.svg?branch=master)](https://travis-ci.org/JuliaPOMDP/RLInterface.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaPOMDP/RLInterface.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaPOMDP/RLInterface.jl?branch=master)

This package provides an interface for working with deep reinfrocement learning problems in Julia.
It is closely integrated with [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) to easily wrap problems defined in those formats. 
While the focus of this interface is on partially observable Markov decision process (POMDP) reinforcement learning, it
is flexible and can easily handle problems that are fully observable as well. 
The interface is very similar to that of [OpenAI Gym](https://gym.openai.com/). This allows algorithms that work with Gym to be used with problems that
are defined in this interface and vice versa.
A shared interface between POMDPs.jl allows easy comparison of reinforcement learning solutions to approximate dynamic
programming solutions when a complete model of the problem is defined.

## Simulation

Running a simulation can be done like so, we use a problem from
[POMDPModels](https://github.com/JuliaPOMDP/POMDPModels.jl) as an example:

```julia
using POMDPModels # for TigerPOMDP
using RLInterface

env = POMDPEnvironment(TigerPOMDP())

function simulate(env::AbstractEnvironment, nsteps::Int = 10)
    done = false
    r_tot = 0.0
    step = 1
    o = reset!(env)
    while !done && step <= nsteps
        action = sample_action(env) # take random action 
        obs, rew, done, info = step!(env, action)
        @show obs, rew, done, info
        r_tot += rew
        step += 1
    end
    return r_tot
end

@show simulate(env)
```

## Interface

The interface provides an `AbstractEnvironment` type from which all custom environments
should inherit. For an example see how this is done with [OpenAI Gym](https://github.com/sisl/Gym.jl). 

Currently, the following functions make up the interface. See the docstrings for more information
```julia
reset!
step!
actions
sample_action
render
```


## Requirements for POMDPs.jl models

To use POMDPs.jl models, the [generative interface](https://juliapomdp.github.io/POMDPs.jl/latest/generative/), including [`initialstate`](https://juliapomdp.github.io/POMDPs.jl/latest/api/#POMDPs.initialstate) must be implemented. In addition, the function
```julia
convert_s(::Type{Vector{Float32}}, s::S, m::M)
```
where `M` is the `MDP` type with states of type `S`, or
```julia
convert_o(::Type{Vector{Float32}}, o::O, m::M)
```
where `M` is a `POMDP` with observation type `O`, will be used to convert the observation into a vector (Sometimes `Vector{Float32}` will be replaced with a different `AbstractVector` type if the environment is configured differently).

The type of the observation vector returned by an environment may be specified as an argument to the environment constructor.
The default for a particular MDP or POMDP may be specified by implementing `obsvector_type`. This function default to `Vector{Float32}`. 
It will be called when initializing any of the wrappers to determine the type to give as input to the `convert_s` function.
