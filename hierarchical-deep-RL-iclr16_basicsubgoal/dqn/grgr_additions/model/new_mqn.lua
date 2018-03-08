require 'grgr_additions.model.net'

local MQN, parent = torch.class('MQN', 'Net')
function MQN:build_model(args)
    local input = {}
    local init_states = {}
    local x = nn.Identity()()
    table.insert(input, x)
    for i=1, #self.init_states do
        local state = nn.Identity()()
        table.insert(input, state)
        table.insert(init_states, state)
    end
    local T = args.hist_len
    local edim = args.n_hid_enc
    local cnn_features = self:build_cnn(args, x)
end

function MQN:build_retrieval(args)

end

function MQN:build_context(args)

end

function MQN:build_subgoal(args, input)
    subgoal_proc = nn.Sequential()
        :add(nn.Linear(args.subgoal_dims*9, args.subgoal_nhid))
        :add(nn.ReLU())
        :add(nn.Linear(args.subgoal_nhid,args.subgoal_nhid))
        :add(nn.ReLU())
    return subgoal_proc(input)
end

function MQN:build_cnn(args, input)
    local reshape_input = nn.View(-1, unpack(args.image_dims))(input)
    local conv, conv_nl = {}, {}
    local prev_dim = args.ncols
    local prev_input = reshape_input
    for i=1,#args.n_units do
        conv[i] = nn.SpatialConvolution(prev_dim, args.n_units[i],
                            args.filter_size[i], args.filter_size[i],
                            args.filter_stride[i], args.filter_stride[i],
                            args.pad[i], args.pad[i])(prev_input)
        conv_nl[i] = nn.ReLU()(conv[i])
        prev_dim = args.n_units[i]
        prev_input = conv_nl[i]
    end

    local conv_flat = nn.View(-1):setNumInputDims(3)(conv_nl[#args.n_units])
    return nn.View(-1, args.hist_len, args.conv_dim):setNumInputDims(2)(conv_flat)
end
