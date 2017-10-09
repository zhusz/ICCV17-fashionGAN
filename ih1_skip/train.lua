require 'nngraph'
require 'cunn'
require 'cudnn'
require 'hdf5'
require 'image'
require 'optim'
require 'paths'
local matio = require 'matio'
train_ind = matio.load('../data_release/benchmark/ind.mat','train_ind'):view(-1)

local dispSurrogate = dofile('../codes_lua/dispSurrogate.lua')
local disp = require 'display'
local getNet = dofile('../codes_lua/getNet.lua')
torch.setdefaulttensortype('torch.FloatTensor')

local theme = 'ih1'
assert(theme == 'ih1')

local config = dofile('./config_ih1.lua')
local lambda = 100
local G,D = dofile('./net_graph_ih1.lua')

local h5file = hdf5.open('../data_release/supervision_signals/G2.h5', 'r')
local ih = h5file:read('/ih'):all()
ih = ih:permute(1,2,4,3)
local ih_mean = h5file:read('/ih_mean'):all()
ih_mean = ih_mean:view(1,3,128,128)
ih_mean = ih_mean:permute(1,2,4,3)
local b_ = h5file:read('/b_'):all()
b_ = b_:permute(1,2,4,3)
local n_file = ih:size(1)

----------------------------------------------------------
local text = matio.load('../data_release/test_phase_inputs/encode_hn2_rnn_100_2_full.mat', 'hn2')
text = text:contiguous()
----------------------------------------------------------

config.lr = 0.0002 -- * 1e2 * 2.5
config.beta1 = 0.5

local criterion = nn.BCECriterion()
local criterionAE = nn.AbsCriterion()
local optimStateG = {
    learningRate = config.lr,
    beta1 = config.beta1,
}
local optimStateD = {
    learningRate = config.lr,
    beta1 = config.beta1,
}

local nz = config.nz
local input = torch.Tensor(config.batchSize, config.n_c, config.win_size, config.win_size)
local condition = torch.Tensor(config.batchSize, config.n_condition, config.win_size, config.win_size) -- zhuzhu

print(nz)
local noise = torch.Tensor(config.batchSize, nz, 1, 1) -- zhuzhunoise
local label = torch.Tensor(config.batchSize, 1, 14, 14)
local encode = torch.Tensor(config.batchSize, config.nt_input, 1, 1)
local errD, errG, errL1
cutorch.setDevice(1)
input = input:cuda();  noise = noise:cuda();  label = label:cuda();  condition = condition:cuda(); encode = encode:cuda() -- zhuzhunoise
local input_record, condition_record

----------------------------------------------------------
local input_wrong = torch.Tensor(config.batchSize, config.n_c, config.win_size, config.win_size)
local condition_wrong = torch.Tensor(config.batchSize, config.n_condition, config.win_size, config.win_size)
input_wrong = input_wrong:cuda();     condition_wrong = condition_wrong:cuda()
----------------------------------------------------------

if pcall(require, 'cudnn') then
    require 'cudnn'
    cudnn.benchmark = true
    cudnn.convert(G, cudnn)
    cudnn.convert(D, cudnn)
end
D:cuda();           G:cuda();           criterion:cuda();      criterionAE:cuda()

local parametersD, gradParametersD = D:getParameters()
local parametersG, gradParametersG = G:getParameters()

----------- zhuzhu ------------
local cb = {torch.Tensor{3,2,1,1,2,3,2}, torch.Tensor{2,3,3,2,1,1,2}, torch.Tensor{1,1,2,3,3,2,2}}
for i = 1,config.n_condition do cb[i] = cb[i] * 0.25 end
local H = torch.Tensor{0.0030,0.0133,0.0219,0.0133,0.0030,0.0133,0.0596,0.0983,0.0596,0.0133,0.0219,0.0983,0.1621,0.0983,0.0219,0.0133,0.0596,0.0983,0.0596,0.0133,0.0030,0.0133,0.0219,0.0133,0.0030}:view(5,5)
-------------------------------

local train_size = train_ind:size(1)
local simple_sample = function()
    local ind = torch.randperm(train_size):narrow(1,1,config.batchSize)
    local ind_wrong = torch.Tensor(config.batchSize)
    for i = 1,config.batchSize do ind_wrong[i] = (ind[i] + math.random(train_size-1) - 1) % train_size + 1; end
    for i = 1,config.batchSize do ind[i] = train_ind[ind[i]] end
    for i = 1,config.batchSize do ind_wrong[i] = train_ind[ind_wrong[i]] end
    noise:normal(0,1) -- zhuzhunoise

    for i = 1,config.batchSize do
        input[{{i},{},{},{}}] = ih[{{ind[i]},{},{},{}}]

        local t = b_[{{ind[i]},{1},{},{}}]
        for j = 1,config.n_condition do
            local u = torch.Tensor(1,1,config.win_size,config.win_size):zero()
            for k = 1,config.n_map_all do
                u[t:eq(k%config.n_map_all)] = cb[j][k]
            end
            -- do blurring toward u
            local v = image.convolve(u:squeeze():float(), H:float(), 'same'):contiguous()
            condition[{{i},{j},{},{}}] = v:view(1,1,config.win_size,config.win_size)
        end
        encode[{{i},{},{},{}}] = text[{{ind[i]},{}}]:view(1, config.nt_input, 1, 1)
    end
    condition = condition - 0.5; -- zero mean

    for i = 1,config.batchSize do
        input_wrong[{{i},{},{},{}}] = ih[{{ind_wrong[i]},{},{},{}}]

        local t = b_[{{ind_wrong[i]},{1},{},{}}]
        for j = 1,config.n_condition do
            local u = torch.Tensor(1,1,config.win_size,config.win_size):zero()
            for k = 1,config.n_map_all do
                u[t:eq(k%config.n_map_all)] = cb[j][k]
            end
            -- do blurring torward u
            local v = image.convolve(u:squeeze():float(), H:float(), 'same'):contiguous()
            condition_wrong[{{i},{j},{},{}}] = v:view(1,1,config.win_size,config.win_size)
        end
    end
    condition_wrong = condition_wrong - 0.5

    do return end
end

local real_label = 1
local fake_label = 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

local errD_real, errD_wrong, errD_fake
local fDx = function(x)
    D:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    G:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end) -- zhuzhu
   gradParametersD:zero()

   -- train with real
   simple_sample()
   local fake = G:forward{noise, encode, condition} -- zhuzhuf2A
   input_record = input:clone()
   condition_record = condition:clone()
   -- local real = data:getBatch()
   -- input:copy(real)
   label:fill(real_label)

   local output = D:forward{input, encode, condition} --
   errD_real = criterion:forward(output, label)
   local de_do = criterion:backward(output, label)
   D:backward({input, encode, condition}, de_do) --

   ------------------------------------------------------

   label:fill(fake_label)
   output = D:forward{input_wrong, encode, condition_wrong}
   errD_wrong = config.lambda_mismatch * criterion:forward(output, label)
   de_do = criterion:backward(output, label)
   D:backward({input_wrong, encode, condition_wrong}, de_do)

   ------------------------------------------------------

   -- local fake = G:forward{noise, condition} -- -- zhuzhunoise
   input:copy(fake)
   label:fill(fake_label)

   output = D:forward{input, encode, condition} --
   errD_fake = config.lambda_fake * criterion:forward(output, label)
   de_do = criterion:backward(output, label)
   D:backward({input, encode, condition}, de_do) --

   errD = (errD_real + errD_wrong + errD_fake) / 2 -- zhuzhu

   return errD, gradParametersD
end

local fGx = function(x)
    D:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    G:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end) -- zhuzhu
   gradParametersG:zero()

   local output = D.output -- zhuzhuf2A
   label:fill(real_label)
   errG = criterion:forward(output, label)
   local de_do = criterion:backward(output, label)
   local de_dg = D:updateGradInput({input, encode, condition}, de_do) --
   -------- zhuzhu --------
   errL1 = criterionAE:forward(input, input_record:cuda())
   local df_do_AE = criterionAE:backward(input, input_record:cuda())
   ------------------------
   G:backward({noise,encode,condition}, de_dg[1] + df_do_AE:mul(lambda)) -- -- zhuzhunoise

   return errG, gradParametersG
end

-- training
-- zhuzhunoise
local vis_factor = math.ceil(16 / config.batchSize)
local vis_size = config.batchSize * vis_factor
local noise_vis = torch.Tensor(vis_size, nz, 1,1)
noise_vis = noise_vis:cuda()
noise_vis:normal(0,1)
local condition_vis = torch.Tensor(vis_size, config.n_condition, config.win_size, config.win_size)
condition_vis = condition_vis:cuda()
local encode_vis = torch.Tensor(vis_size, config.nt_input, 1, 1)
encode_vis = encode_vis:cuda()

for iter = 1, 1e9 do
    optim.adam(fDx, parametersD, optimStateD)
    optim.adam(fGx, parametersG, optimStateG)
    if iter % 20 == 0 then
        print(('Iter %d: ErrD %.5f (ErrD_real %.5f, ErrD_wrong %.5f, ErrD_fake %.5f), ErrG %.5f, ErrL1 %.5f.'):format(iter,errD,errD_real,errD_wrong,errD_fake,errG, errL1))
    end

    if iter == 1 then
        for i = 1, vis_size do condition_vis[{{i},{},{},{}}] = condition[{{1},{},{},{}}] end
        for i = 1, vis_size do encode_vis[{{i},{},{},{}}] = text[{{(i-1)%4+1},{}}]:view(1,100,1,1) end
    end
    local base = 700
    local fake_vis = G:forward{noise_vis, encode_vis, condition_vis}

    if iter % 20 == 0 then
        dispSurrogate(fake_vis:type('torch.FloatTensor') + ih_mean:repeatTensor(vis_size, 1,1,1), 3+base, 'fake')
        dispSurrogate(input_record:type('torch.FloatTensor') + ih_mean:repeatTensor(config.batchSize, 1,1,1), 4+base, 'real')
        dispSurrogate(condition_record:type('torch.FloatTensor'), 5+base, 'condition')
    end

    if iter == 1 or iter % 10000 == 0 then
        local net = {}
        net.G = G
        net.D = D
        torch.save('./' .. theme .. '/' .. theme .. '_'.. tostring(iter) .. '.t7', net)
    end
end
