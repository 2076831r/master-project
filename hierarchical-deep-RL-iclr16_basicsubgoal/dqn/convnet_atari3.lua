--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require 'grgr_additions.model.init'

return function(args)
    args.n_units        = {32, 64, 64}
    args.filter_size    = {8, 4, 3}
    args.filter_stride  = {4, 2, 1}
    args.n_hid          = {512}
    args.nl             = nn.Rectifier
    args.name = "mqn"

    return g_create_network(args)
end
