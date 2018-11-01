# RLInterface

[![Build Status](https://travis-ci.org/sisl/DeepRL.jl.svg?branch=master)](https://travis-ci.org/sisl/DeepRL.jl)
[![Coverage Status](https://coveralls.io/repos/github/sisl/DeepRL.jl/badge.svg?branch=master)](https://coveralls.io/github/sisl/DeepRL.jl?branch=master)

This package provides an interface for working with deep reinfrocement learning problems in Julia.
It is closely integrated with [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) to easily wrap problems defined in those formats. 
While the focus of this interface is on partially observable Markov decision process (POMDP) reinforcement learning, it
is flexible and can easily handle problems that are fully observable as well. 
The interface is very similar to that of [OpenAI Gym](https://gym.openai.com/). This allows algorithms that work with Gym to be used with problems that
are defined in this interface and vice versa.
A shared interface between POMDPs.jl allows easy comparison of reinforcement learning solutions to approximate dynamic
programming solutions when a complete model of the problem is defined.

**Note**: Only environments with discrete action spaces are supported. Pull requests to support problems with continuous action spaces are welcome.

## Interface

The interface provides an `AbstractEnvironment` type from which all custom environments
should inherit. For an example see how this is done with [OpenAI Gym](https://github.com/sisl/Gym.jl). 

Running a simulation can be done like so, we use a problem from
[POMDPModels](https://github.com/JuliaPOMDP/POMDPModels.jl) as an example:

```julia
using POMDPModels # for TigerPOMDP
using RLInterface

env = POMDPEnvironment(TigerPOMDP())

nsteps = 1
done = false
r_tot = 0.0

o = reset(env)
while !done && step <= nsteps
    action = sample_action(env)
    obs, rew, done, info = step!(env, action)
    println(obs, " ", rew, " ", done, " ", info)
    r_tot += rew
    step += 1
end
```

## Key Functions to Implement

This interface relies on the following function:

```julia
convert_s(T::Type{A1}, state::A2, problem::Union{MDP, POMDP}) where A1<:AbstractArray
```

which should return something of type A1. As well as:

```julia
POMDPs.initialstate(problem::Union{MDP, POMDP}, rng::AbstractRNG)
```
which should return something of type State, where the argument problem is of type e.g. MDP{State, Action}

