require 'nngraph'
require 'cunn'
require 'cudnn'
require 'hdf5'
require 'image'
require 'optim'
require 'paths'
local vis = dofile('../codes_lua/vis.lua')
local matio = require 'matio'
train_ind = matio.load('../data_release/benchmark/ind.mat','train_ind'):view(-1)

local dispSurrogate = dofile('../codes_lua/dispSurrogate.lua')
local disp = require 'display'
local getNet = dofile('../codes_lua/getNet.lua')
torch.setdefaulttensortype('torch.FloatTensor')

local theme = 'sr1'
assert(theme == 'sr1')

local config = dofile('./config_sr1.lua')
local G, D
G,D = dofile('./net_graph_sr1.lua')

local h5file = hdf5.open('../data_release/supervision_signals/G1.h5', 'r');
local b_ = h5file:read('/b_'):all();
h5file:close();
local n_file = b_:size(1)

local lr = matio.load('../data_release/test_phase_inputs/sr1_8.mat','d')
lr = lr:permute(4,3,1,2)

----------------------------------------------------------
local text = matio.load('../data_release/test_phase_inputs/encode_hn2_rnn_100_2_full.mat', 'hn2')
text = text:contiguous()
----------------------------------------------------------

config.lr = 0.0002
config.beta1 = 0.5

local criterion = nn.BCECriterion()
local cri_seg = cudnn.SpatialCrossEntropyCriterion()
local optimStateG = {
    learningRate = config.lr,
    beta1 = config.beta1,
}
local optimStateD = {
    learningRate = config.lr,
    beta1 = config.beta1,
}

local nz = config.nz
local input = torch.Tensor(config.batchSize, config.n_map_all, config.win_size, config.win_size)
local condition = torch.Tensor(config.batchSize, config.n_condition, config.lr_win_size, config.lr_win_size)

print(nz)
local noise = torch.Tensor(config.batchSize, nz, 1, 1)
local label = torch.Tensor(config.batchSize)
local encode = torch.Tensor(config.batchSize, config.nt_input, 1, 1)
local seg_target = torch.Tensor(config.batchSize, config.win_size,config.win_size)
local errD, errG
cutorch.setDevice(1)
input = input:cuda();  noise = noise:cuda();  label = label:cuda();  condition = condition:cuda();  encode = encode:cuda();  seg_target = seg_target:cuda()
local input_record

----------------------------------------------------------
local input_wrong = torch.Tensor(config.batchSize, config.n_map_all, config.win_size, config.win_size)
local condition_wrong = torch.Tensor(config.batchSize, config.n_condition, config.lr_win_size, config.lr_win_size)
input_wrong = input_wrong:cuda();     condition_wrong = condition_wrong:cuda()
----------------------------------------------------------

if pcall(require, 'cudnn') then
    require 'cudnn'
    cudnn.benchmark = true
    cudnn.convert(G, cudnn)
    cudnn.convert(D, cudnn)
end
D:cuda();           G:cuda();           criterion:cuda();      cri_seg:cuda()

local parametersD, gradParametersD = D:getParameters()
local parametersG, gradParametersG = G:getParameters()

local normal_holder = torch.Tensor(config.batchSize, config.n_map_all, config.win_size, config.win_size):cuda()
local simple_sample = function()
    local ind = torch.randperm(train_ind:size(1)):narrow(1,1,config.batchSize)
    local ind_wrong = torch.Tensor(config.batchSize)
    for i = 1,config.batchSize do ind_wrong[i] = (ind[i] + math.random(train_ind:size(1)-1) - 1) % train_ind:size(1) + 1; end
    for i = 1,config.batchSize do ind[i] = train_ind[ind[i]] end
    for i = 1,config.batchSize do ind_wrong[i] = train_ind[ind_wrong[i]] end
    noise:normal(0,1)

    for i = 1,config.batchSize do
        local t = b_[{{ind[i]},{1},{},{}}]
        for j = 1,config.n_map_all do
            local u = input[{{i},{j},{},{}}]:zero()
            u[t:eq(j%config.n_map_all)] = 1
            input[{{i},{j},{},{}}] = u
        end
        condition[{{i},{},{},{}}] = lr[{{ind[i]},{},{},{}}]
        encode[{{i},{},{},{}}] = text[{{ind[i]},{}}]:view(1, config.nt_input, 1, 1)
    end

    for i = 1,config.batchSize do
        local t = b_[{{ind_wrong[i]},{1},{},{}}]
        for j = 1,config.n_map_all do
            local u = input_wrong[{{i},{j},{},{}}]:zero()
            u[t:eq(j%config.n_map_all)] = 1
            input_wrong[{{i},{j},{},{}}] = u
        end
        condition_wrong[{{i},{},{},{}}] = lr[{{ind_wrong[i]},{},{},{}}]
    end

    for i = 1,config.batchSize do
        seg_target[{{i},{},{}}] = b_[{{ind[i]},{1},{},{}}]:view(1,config.win_size,config.win_size)
    end
    seg_target[seg_target:eq(0)] = config.n_map_all

end

local real_label = 1
local fake_label = 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

local errD_real, errD_wrong, errD_fake
local errSeg
local fDx = function(x)
   gradParametersD:zero()

   simple_sample()
   input_record = input:clone()
   label:fill(real_label)

   local output = D:forward{input, encode, condition}
   errD_real = criterion:forward(output, label)
   local de_do = criterion:backward(output, label)
   D:backward({input,encode,condition}, de_do)

   ---------------------------------------------------------

   label:fill(fake_label)
   output = D:forward{input_wrong, encode, condition_wrong}
   errD_wrong = config.lambda_mismatch * criterion:forward(output, label)
   de_do = criterion:backward(output, label)
   D:backward({input_wrong, encode, condition_wrong}, de_do)

   ---------------------------------------------------------

   local fake = G:forward{noise, encode, condition}
   input:copy(fake)

   output = D:forward{input,encode,condition}
   errD_fake = config.lambda_fake * criterion:forward(output, label)
   de_do = criterion:backward(output, label)
   D:backward({input,encode,condition}, de_do)

   errD = errD_real + errD_wrong + errD_fake

   return errD, gradParametersD
end

local fGx = function(x)
   gradParametersG:zero()

   local output = D:forward{input, encode, condition} -- this input is actually the fake --
   label:fill(real_label)
   errG = criterion:forward(output, label)
   local de_do = criterion:backward(output, label)
   local de_dg = D:updateGradInput({input,condition}, de_do)
   errSeg = cri_seg:forward(input, seg_target)
   local df_do_seg = cri_seg:backward(input, seg_target)
   G:backward({noise,encode,condition}, de_dg[1] + df_do_seg:mul(100))

   return errG, gradParametersG
end

-- training
local bsz_vis = 16
local noise_vis = torch.Tensor(bsz_vis, nz, 1,1)
noise_vis = noise_vis:cuda()
noise_vis:normal(0,1)
local condition_vis = torch.Tensor(bsz_vis, config.n_condition, config.lr_win_size, config.lr_win_size)
condition_vis = condition_vis:cuda()
local encode_vis = torch.Tensor(bsz_vis, config.nt_input, 1, 1)
encode_vis = encode_vis:cuda()

for iter = 1, 25000 do
    optim.adam(fDx, parametersD, optimStateD)
    optim.adam(fGx, parametersG, optimStateG)

    if iter == 1 then
        for i = 1, bsz_vis do condition_vis[{{i},{},{},{}}] = condition[{{1},{},{},{}}] end
        for i = 1, bsz_vis do encode_vis[{{i},{},{},{}}] = text[{{(i-1)%4+1},{}}]:view(1,100,1,1) end
    end
    if iter % 20 == 0 then
        print(('Iter %d: ErrD %.5f (ErrD_real %.5f, ErrD_wrong %.5f, ErrD_fake %.5f), ErrG %.5f, ErrSeg %.5f'):format(iter,errD,errD_real,errD_wrong,errD_fake,errG,errSeg))
        local base = config.disp_win_id
        local fake = G:forward{noise_vis, encode_vis, condition_vis} --

        dispSurrogate(fake:type('torch.FloatTensor'), 3+base, 'fake', 'm2c')
        dispSurrogate(seg_target:view(4,1,128,128):float(), 8+base, 'seg_target')
    end

    if iter % 1000 == 0 then
        local net = {}
        net.G = G
        net.D = D
        torch.save('./' .. theme .. '/' .. theme .. '.t7', net)
    end
end
