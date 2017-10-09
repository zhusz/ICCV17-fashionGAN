local config = {}

-- learning batch
config.batchSize = 1 -- zhuzhu
config.test_batchSize = 10
config.win_size = 128

-- specific size
config.n_map_all = 7
config.n_condition = 3
config.n_z = 100
config.nz = config.n_z
config.n_c = 3
config.nc = config.nc

-- Resume or finetune
config.resume_iter = 0

return config

