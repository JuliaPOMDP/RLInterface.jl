function process(env)
    for (msg, key, res) in zip([Dict("cmd"=>"obs_dimensions"), Dict("cmd"=>"n_actions")],
                               ["obs_dim", "n_actions"],
                               [obs_dimensions(env), n_actions(env)])
        respmsg = DeepRL.process!(env, msg)
        @test respmsg[key] == res
    end
end
