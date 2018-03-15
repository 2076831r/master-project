require 'grgr_additions.model.net'

local MQN, parent = torch.class('MQN', 'Net')
function MQN:build_model(args)
    -- CNN input
    local input = {}
    local init_states = {}
    local x = nn.Identity()()
    table.insert(input, x)
    for i=1, #self.init_states do
        local state = nn.Identity()()
        table.insert(input, state)
        table.insert(init_states, state)
    end

    -- subgoal input
    local subgoal_input = {}
    local subgoal_init_states = {}
    local subgoal_x = nn.Identity()()
    table.insert(subgoal_input, subgoal_x)
    for i=1, #self.subgoal_init_states do
        local state = nn.Identity()()
        table.insert(subgoal_input, state)
        table.insert(subgoal_init_states, state)
    end

    local T = args.hist_len
    local edim = args.n_hid_enc
    local cnn_features = self:build_cnn(args, x)
    local subgoal_features = self:build_subgoal(args, subgoal_x)
    local history = nn.Narrow(2, 1, T-1)(cnn_features)
    local history_flat = nn.View(-1):setNumInputDims(1)(history)
    local key_blocks = nn.Linear(args.conv_dim, edim)(history_flat)
    local val_blocks = nn.Linear(args.conv_dim, edim)(history_flat)
    key_blocks = nn.View(-1, T-1, edim):setNumInputDims(2)(key_blocks)
    val_blocks = nn.View(-1, T-1, edim):setNumInputDims(2)(val_blocks)
    local hid, o = self:build_retrieval(args, key_blocks, val_blocks,
                cnn_features, args.conv_dim, unpack(init_states))
    local hid2dim = nn.View(-1):setNumInputDims(1)(hid):annotate{name = 'hid2dim'}
    local C = args.Linear(edim, edim)(hid2dim)
    local D = nn.CAddTable()({C, o})
    local hid_out
    if args.lindim == args.edim then
        hid_out = D
    elseif args.lindim == 0 then
        hid_out = nn.ReLU()(D)
    else
        local F = nn.Narrow(2,1,args.lindim)(D)
        local G = nn.Narrow(2,1+args.lindim,edim-args.lindim)(D)
        local K = nn.ReLU()(G)
        hid_out = nn.JoinTable(2)({F,K})
    end
    local out = nn.View(-1):setNumInputDims(1)(hid_out)
    out = nn.JoinTable(2)({out, subgoal_features})
    local q = nn.Linear(args.n_hid_enc+50, args.n_actions)(out)
    nngraph.annotateNodes()
    return nn.gModule({x, subgoal_x}, {q})
end

function MQN:build_subgoal(args, input)
    local subgoal_linear1 = nn.Linear(args.subgoal_dims*9, args.subgoal_nhid)(input)
    local subgoal_ReLu1 = nn.ReLU()(subgoal_linear1)
    local subgoal_linear2 = nn.Linear(args.subgoal_nhid,args.subgoal_nhid)(subgoal_ReLu1)
    return nn.ReLU()(subgoal_linear2)
end

function MQN:build_retrieval(args, key_blocks, val_blocks, cnn_features, conv_dim, c0, h0)
    local T = args.hist_len
    local edim = args.edim
    local memsize = math.min(T-1, args.memsize)
    local context = self:build_context(args, cnn_features, args.conv_dim, edim, c0, h0)
    local hid = nn.View(1, -1):setNumInputDims(1)(context)
    local key_blocks_t = nn.Narrow(2, T - memsize, memsize)(key_blocks):annotate{name = 'key_blocks_t'}
    local val_blocks_t = nn.Narrow(2, T - memsize, memsize)(val_blocks):annotate{name = 'val_blocks_t'}
    local MM_key = nn.MM(false, true)
    local key_out = MM_key({hid, key_blocks_t})
    local key_out2dim = nn.View(-1):setNumInputDims(2)(key_out)
    local P = nn.SoftMax()(key_out2dim)
    local probs3dim = nn.View(1, -1):setNumInputDims(1)(P):annotate{name = 'probs3dim'}
    local MM_val = nn.MM(false, false)
    local o = MM_val({probs3dim, val_blocks_t})
    if args.gpu and args.gpu > 0 then
        MM_key = MM_key:cuda()
        MM_val = MM_val:cuda()
    end
    return context, o
end

function MQN:build_context(args, x, xdim, edim, c0, h0)
    local context = nn.Narrow(2, args.hist_len, 1)(x)
    local context_flat = nn.View(-1):setNumInputDims(1)(context)
    return nn.Linear(xdim, edim)(context_flat)
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
                            args.pad[i], args.pad[i])(prev_input):annotate{name = 'conv'..i}
        conv_nl[i] = nn.ReLU()(conv[i]):annotate{name = 'conv_nl'..i}
        prev_dim = args.n_units[i]
        prev_input = conv_nl[i]
    end
    local conv_flat = nn.View(-1):setNumInputDims(3)(conv_nl[#args.n_units])
    return nn.View(-1, args.hist_len, args.conv_dim):setNumInputDims(2)(conv_flat)
end
