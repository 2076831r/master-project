require 'nn'
require 'nngraph'
require 'grgr_additions.LinearNB'
require 'grgr_additions.model.dqn'
require 'grgr_additions.model.drqn'
require 'grgr_additions.model.mqn'
require 'grgr_additions.model.rmqn'
require 'grgr_additions.model.frmqn'

function g_create_network(args)
    args.hist_len           = args.hist_len or 10
    args.name               = "dqn"
    args.n_actions          = args.n_actions or 6
    args.ncols              = args.ncols or 3
    args.image_dims         = args.image_dims or {3, 84, 84}
    args.input_dims         = args.input_dims or {args.hist_len * args.ncols, 84, 84}
    args.n_units            = args.n_units or {32, 64}
    args.pad                = args.pad or {1, 1}
    args.n_hid_enc          = args.n_hid_enc or 256
    args.edim               = args.edim or 256
    args.memsize            = args.memsize or (args.hist_len - 1)
    args.lindim             = args.lindim or args.edim / 2
    args.lstm_dim           = args.edim or 256
    args.gpu                = args.gpu or -1
    args.conv_dim           = args.n_units[#args.n_units] * 8 * 8
    args.Linear             = nn.LinearNB

    if args.name == "dqn" then
        return DQN.new(args)
    elseif args.name == "drqn" then
        return DRQN.new(args)
    elseif args.name == "mqn" then
        return MQN.new(args)
    elseif args.name == "rmqn" then
        return RMQN.new(args)
    elseif args.name == "frmqn" then
        return FRMQN.new(args)
    else
        error("Invalid model name:" .. args.name)
    end
end
