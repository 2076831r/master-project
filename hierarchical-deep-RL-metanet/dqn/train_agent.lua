--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]
require 'xlua'
require 'optim'

-- require 'signal'
-- signal.signal("SIGPIPE", function() print("raised") end)

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-subgoal_index', 12, 'the index of the subgoal that we want to reach. used for slurm multiple runs')
cmd:option('-max_subgoal_index', 12, 'used as an index to run with all the subgoals instead of only one specific one')

cmd:option('-exp_folder', '', 'name of folder where current exp state is being stored')
cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', true,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 100000, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')

cmd:option('-subgoal_dims', 7, 'dimensions of subgoals')
cmd:option('-subgoal_nhid', 50, '')
cmd:option('-display_game', false, 'option to display game')
cmd:option('-port', 5550, 'Port for zmq connection')
cmd:option('-stepthrough', false, 'Stepthrough')
cmd:option('-subgoal_screen', true, 'overlay subgoal on screen')

cmd:option('-max_steps_episode', 5000, 'Max steps per episode')

cmd:option('-meta_agent', true, 'hierarchical training')
cmd:option('-max_objects', 6, 'max number of objects in scene that are parsed and used as subgoals')
cmd:option('-gif', false, 'gif on/off')




cmd:text()

local opt = cmd:parse(arg)
ZMQ_PORT = opt.port
SUBGOAL_SCREEN = opt.subgoal_screen
META_AGENT = opt.meta_agent

if not dqn then
    require "initenv"
end


print(opt.env_params)
print(opt.seed)

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local step = 0
time_history[1] = 0

local total_reward
local nrewards
local nepisodes
local episode_reward

local screen, reward, terminal = game_env:getState()

if opt.gif then
    gd = require "gd"
end


local function add_gif(previm, scr, gif_filename)
    local jpg = image.compressJPG(scr:squeeze(), 100)
    local im = gd.createFromJpegStr(jpg:storage():string())
    im:trueColorToPalette(false, 256)
    if previm then
        im:paletteCopy(previm)
    else
        im:gifAnimBegin(gif_filename, true, 0)
    end
    im:gifAnimAdd(gif_filename, false, 0, 0, 7, gd.DISPOSAL_NONE)
    return im
end

local prev_screen_im, prev_screen_subgoalim
if opt.gif then
    prev_screen_im = add_gif(nil, screen, '../gifs/screen_seed=' .. opt.seed .. '.gif')
    prev_screen_subgoalim = add_gif(nil, screen, '../gifs/subgoalscreen_seed=' .. opt.seed .. '.gif')
end

print("Iteration ..", step)
local win = nil

local subgoal

if META_AGENT then
    subgoal = agent:pick_subgoal(screen, 0, terminal, false)
else
    if opt.subgoal_index < opt.max_subgoal_index then
        subgoal = agent:pick_subgoal(screen, opt.subgoal_index)
    else
        subgoal = agent:pick_subgoal(screen)
    end
end

local action_list = {'no-op', 'fire', 'up', 'right', 'left', 'down', 'up-right','up-left','down-right','down-left',
                    'up-fire', 'right-fire','left-fire', 'down-fire','up-right-fire','up-left-fire',
                    'down-right-fire', 'down-left-fire'}

death_counter = 0 --to handle a bug in MZ atari

episode_step_counter = 0
metareward = 0
SAVE_NET_EXIT = false
cum_metareward = 0
numepisodes = 0

test_avg_R = test_avg_R or optim.Logger(paths.concat(opt.exp_folder , 'test_avgR.log'))

while step < opt.steps do
    xlua.progress(step, opt.steps)

    step = step + 1

    subgoal_screen = screen:clone() -- only do overlay on subgoal screen

    if opt.subgoal_screen then
        subgoal_screen[{1,{}, {30+subgoal[1]-5, 30+subgoal[1]+5}, {subgoal[2]-5,subgoal[2]+5} }] = 1
        if opt.display_game then win = image.display({image=subgoal_screen, win=win}) end
    end

    if opt.gif then
        prev_screen_im = add_gif(prev_screen_im, screen, '../gifs/screen_seed=' .. opt.seed .. '.gif')
        prev_screen_subgoalim = add_gif(prev_screen_subgoalim, subgoal_screen, '../gifs/subgoalscreen_seed=' .. opt.seed .. '.gif')
    end

    -- for i=1,#agent.objects do
    --     if agent.objects[i][1] > 0 and agent.objects[i][2] > 0 then
    --         screen[{1,{}, {30+agent.objects[i][1]-5, 30+agent.objects[i][1]+5}, {agent.objects[i][2]-5,agent.objects[i][2]+5} }] = 1
    --     end
    --     win = image.display({image=screen, win=win})
    -- end

    local action_index, isGoalReached, reward_ext, reward_tot,  qfunc = agent:perceive(subgoal, reward, subgoal_screen, terminal)
    metareward = metareward + reward_ext

    if opt.stepthrough then
        print("Reward Ext", reward_ext)
        print("Reward Tot", reward_tot)
        print("Q-func")
        if qfunc then
            for i=1, #action_list do
                print(action_list[i], qfunc[i])
            end
        end

        print("Action", action_index, action_list[action_index])
        io.read()
    end

    if false and new_game then--new_game then
        print("Q-func")
        if prev_Q then
            for i=1, #action_list do
                print(action_list[i], prev_Q[i])
            end
        end
        print("SUM OF PIXELS: ", screen:sum())
        new_game = false
    end

    if metareward > 100 then --end of game after door opens
        terminal = true
        death_counter = 4
    end

    -- game over? get next game!
    if not terminal and  episode_step_counter < opt.max_steps_episode then

        -- if isGoalReached and opt.subgoal_index < opt.max_subgoal_index then
        --     screen,reward, terminal = game_env:newGame()  -- restart game if focussing on single subgoal
        --     subgoal = agent:pick_subgoal(screen, opt.subgoal_index)
        --     if opt.subgoal_screen then
        --         screen[{1,{}, {30+subgoal[1]-5, 30+subgoal[1]+5}, {subgoal[2]-5,subgoal[2]+5} }] = 1
        --     end
        --     isGoalReached = false
        -- end

        screen, reward, terminal = game_env:step(game_actions[action_index], true)
        if not terminal then
            screen, tmp_reward, terminal = game_env:step(game_actions[1], true) -- noop
        end
        reward = reward + tmp_reward
        episode_step_counter = episode_step_counter + 1
        prev_Q = qfunc
    else
        death_counter = death_counter + 1
        -- print("TERMINAL ENCOUNTERED")
        if META_AGENT then
            -- Note: this screen is the death screen (terminal)
            subgoal = agent:pick_subgoal(screen, metareward, true, false)
            if metareward > 0 then
                print('METAR:', metareward)
            end
            cum_metareward = cum_metareward + metareward
            metareward = 0
        end

        if opt.random_starts > 0 then
            -- print("RANDOM GAME STARTING")
            screen, reward, terminal = game_env:nextRandomGame()
        else
            -- print("NEW GAME STARTING")
            screen, reward, terminal = game_env:newGame()
        end

        if death_counter == 5 then
            screen,reward, terminal = game_env:newGame()
            death_counter = 0
            numepisodes = numepisodes + 1
        end

        new_game = true
        isGoalReached = true --new game so reset goal
        episode_step_counter = 0
    end

    if isGoalReached then

        if META_AGENT then
            if metareward > 0 then
                print("METAREWARD: ", metareward, "| subgoal:", subgoal[-1])
                -- io.read()
            end
            subgoal = agent:pick_subgoal(screen, metareward, terminal, false)
            cum_metareward = cum_metareward + metareward
            metareward = 0
        else
            if opt.subgoal_index  < opt.max_subgoal_index then
                subgoal = agent:pick_subgoal(screen, opt.subgoal_index)
            else
                subgoal = agent:pick_subgoal(screen)
            end
        end

        isGoalReached = false
    end


    -- display screen
    if opt.display_game then
        if not opt.subgoal_screen then
            screen_cropped = screen:clone()
            screen_cropped = screen_cropped[{{},{},{30,210},{1,160}}]
            screen_cropped[{1,{}, {subgoal[1]-5, subgoal[1]+5}, {subgoal[2]-5,subgoal[2]+5} }] = 1
            win = image.display({image=screen_cropped, win=win})
        end
    end

    if step % opt.prog_freq == 0 then
        assert(step==agent.numSteps, 'trainer step: ' .. step ..
                ' & agent.numSteps: ' .. agent.numSteps)
        print("Steps: ", step)
        if agent.subgoal_total[8] and agent.subgoal_success[8] and agent.subgoal_total[8] > 0 then
            if agent.subgoal_total[8]>100 then
                if (agent.subgoal_success[8]/agent.subgoal_total[8])>0.9 then
                    print('SAVING NET:')
                    SAVE_NET_EXIT = true
                end
            end
        end

        agent:report(paths.concat(opt.exp_folder , 'subgoal_statistics_' .. step .. '.t7'))
        collectgarbage()
    end


    if step%1000 == 0 then collectgarbage() end

    -- evaluation
    if step % opt.eval_freq == 0 and step > learn_start then
        cum_metareward = cum_metareward / math.max(1,numepisodes)
        test_avg_R:add{['% Average Meta Reward'] = cum_metareward}
        test_avg_R:style{['% Average Meta Reward'] = '-'}; --test_avg_R:plot()
        numepisodes = 0
        cum_metareward = 0
    end

    --     print("Testing ...")

    --     local cum_reward_ext = 0
    --     local cum_reward_tot = 0

    --     screen, reward, terminal = game_env:newGame()
    --     if META_AGENT then
    --         subgoal = agent:pick_subgoal(screen, nil, terminal, true, 0.1)
    --     else
    --         subgoal = agent:pick_subgoal(screen)
    --     end


    --     test_avg_Q = test_avg_Q or optim.Logger(paths.concat(opt.exp_folder , 'test_avgQ.log'))
    --     test_avg_R = test_avg_R or optim.Logger(paths.concat(opt.exp_folder , 'test_avgR.log'))
    --     test_avg_R2 = test_avg_R2 or optim.Logger(paths.concat(opt.exp_folder , 'test_avgR2.log'))

    --     total_reward = 0
    --     nrewards = 0
    --     nepisodes = 0
    --     episode_reward = 0

    --     death_counter_eval = 0

    --     local eval_time = sys.clock()
    --     for estep=1,opt.eval_steps do
    --         xlua.progress(estep, opt.eval_steps)

    --         subgoal_screen = screen:clone()
    --         if opt.subgoal_screen then
    --             subgoal_screen[{1,{}, {30+subgoal[1]-5, 30+subgoal[1]+5}, {subgoal[2]-5,subgoal[2]+5} }] = 1
    --             if opt.display_game  then win = image.display({image=subgoal_screen, win=win}) end
    --         end

    --         local action_index, isGoalReached, reward_ext, reward_tot = agent:perceive(subgoal, reward, subgoal_screen, terminal, true, 0.1)
    --         metareward = metareward + reward_ext

    --         -- remove death pen for metareward
    --         if metareward < -100 then
    --             metareward = 0
    --         end

    --         --if metareward > 0 then
    --         --end


    --         cum_reward_tot = cum_reward_tot + reward_tot
    --         cum_reward_ext = cum_reward_ext + reward_ext

    --         -- Play game in test mode (episodes don't end when losing a life)
    --         screen, reward, terminal = game_env:step(game_actions[action_index])
    --         if not terminal then
    --             screen, reward, terminal = game_env:step(game_actions[1]) -- noop
    --         end

    --         -- display screen (REDUNDANT?  - already being displayed above)
    --         -- if opt.display_game and not opt.subgoal_screen then
    --         --     screen_cropped = screen:clone()
    --         --     screen_cropped = screen_cropped[{{},{},{30,210},{1,160}}]
    --         --     screen_cropped[{1,{}, {subgoal[1]-5, subgoal[1]+5}, {subgoal[2]-5,subgoal[2]+5} }] = 1
    --         --     win = image.display({image=screen_cropped, win=win})
    --         -- end

    --         if estep%1000 == 0 then collectgarbage() end

    --         -- record every reward
    --         episode_reward = episode_reward + reward
    --         if reward ~= 0 then
    --            nrewards = nrewards + 1
    --         end

    --         if terminal then
    --             total_reward = total_reward + episode_reward
    --             episode_reward = 0
    --             nepisodes = nepisodes + 1

    --             if META_AGENT then
    --                 subgoal = agent:pick_subgoal(screen, nil, terminal, true, 0.1)
    --             end

    --             screen, reward, terminal = game_env:newGame()
    --             isGoalReached = true --new game so reset subgoal
    --             death_counter_eval = death_counter_eval + 1

    --             if death_counter_eval == 5 then
    --                 screen,reward, terminal = game_env:newGame()
    --                 death_counter_eval = 0
    --             end
    --         end

    --         if isGoalReached then
    --             if META_AGENT then
    --                 subgoal = agent:pick_subgoal(screen, nil, false, true, 0.1)
    --             else
    --                 subgoal = agent:pick_subgoal(screen)
    --             end
    --             isGoalReached = false
    --         end

    --     end

    --     eval_time = sys.clock() - eval_time
    --     start_time = start_time + eval_time
    --     -- agent:compute_validation_statistics()
    --     local ind = #reward_history+1
    --     total_reward = total_reward/math.max(1, nepisodes)

    --     cum_reward_ext = cum_reward_ext / math.max(1,nepisodes)
    --     cum_reward_tot = cum_reward_tot / math.max(1,nepisodes)

    --     if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
    --         agent.best_network_meta = agent.network_meta:clone()
    --     end

    --     if agent.v_avg then
    --         v_history[ind] = agent.v_avg
    --         td_history[ind] = agent.tderr_avg
    --         qmax_history[ind] = agent.q_max
    --     end
    --     print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])

    --     test_avg_R:add{['% Average Extrinsic Reward'] = cum_reward_ext}
    --     test_avg_R2:add{['% Average Total Reward'] = cum_reward_tot}
    --     test_avg_Q:add{['% Average Q'] = agent.v_avg}


    --     test_avg_R:style{['% Average Extrinsic Reward'] = '-'}; test_avg_R:plot()
    --     test_avg_R2:style{['% Average Total Reward'] = '-'}; test_avg_R2:plot()

    --     test_avg_Q:style{['% Average Q'] = '-'}; test_avg_Q:plot()

    --     reward_history[ind] = total_reward
    --     reward_counts[ind] = nrewards
    --     episode_counts[ind] = nepisodes

    --     time_history[ind+1] = sys.clock() - start_time

    --     local time_dif = time_history[ind+1] - time_history[ind]

    --     local training_rate = opt.actrep*opt.eval_freq/time_dif

    --     print(string.format(
    --         '\nSteps: %d (frames: %d), extrinsic reward: %.2f, total reward (I+E): %.2f, epsilon: %.2f, lr: %G, ' ..
    --         'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
    --         'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d',
    --         step, step*opt.actrep, cum_reward_ext, cum_reward_tot, agent.ep, agent.lr, time_dif,
    --         training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
    --         nepisodes, nrewards))
    -- end


    if SAVE_NET_EXIT or (step % opt.save_freq == 0 or step == opt.steps) then
        local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
            agent.valid_s2, agent.valid_term
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = nil, nil, nil, nil, nil, nil, nil
        local w_meta, dw_meta, g_meta, g2_meta, delta, delta2, deltas, deltas_meta, tmp_meta = agent.w_meta, agent.dw_meta,
            agent.g_meta, agent.g2_meta, agent.delta, agent.delta2, agent.deltas, agent.deltas_meta, agent.tmp_meta
        agent.w_meta, agent.dw_meta, agent.g_meta, agent.g2_meta, agent.delta, agent.delta2, agent.deltas,
            agent.deltas_meta, agent.tmp_meta = nil, nil, nil, nil, nil, nil, nil, nil, nil

        local filename = opt.name
        if opt.save_versions > 0 then
            filename = filename .. "_" .. math.floor(step / opt.save_versions)
        end
        filename = filename
        torch.save(filename .. ".t7", {agent = agent,
                                model = agent.network,
                                best_model = agent.best_network,
                                model_meta = agent.network_meta,
                                best_model_meta = agent.best_network_meta,
                                reward_history = reward_history,
                                reward_counts = reward_counts,
                                episode_counts = episode_counts,
                                time_history = time_history,
                                v_history = v_history,
                                td_history = td_history,
                                qmax_history = qmax_history,
                                arguments=opt})
        if opt.saveNetworkParams then
            local nets = {network=w_meta:clone():float()}
            torch.save(filename..'.params.t7', nets, 'ascii')
        end
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = s, a, r, s2, term
        agent.w_meta, agent.dw_meta, agent.g_meta, agent.g2_meta, agent.delta, agent.delta2, agent.deltas,
            agent.deltas_meta, agent.tmp_meta = w_meta, dw_meta, g_meta, g2_meta, delta, delta2, deltas, deltas_meta, tmp_meta
        print('Saved:', filename .. '.t7')
        io.flush()
        collectgarbage()

        if SAVE_NET_EXIT then
            io.read()
            print('Halted ..... ')
            SAVE_NET_EXIT = false
        end
    end
end
