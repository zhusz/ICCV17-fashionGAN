local config = {}

-- learning batch
config.batchSize = 4
config.win_size = 128
config.lr_win_size = 8

-- specific size
config.n_map_all = 7
config.n_condition = 4

config.nz = 80
config.nt_input = 100
config.nt = 20
config.lambda_fake = 0.9
config.lambda_mismatch = 1 - config.lambda_fake

-- Displaying and logging
config.resume_iter = 0
config.disp_win_id = 0

return config

