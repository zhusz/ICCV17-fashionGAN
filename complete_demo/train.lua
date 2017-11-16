require 'cunn'
require 'cudnn'
require 'image'
local matio = require 'matio'
local optim = require 'optim'
local d = dofile('/home/szzhu/link/alper/ga/codes_lua/dispSurrogate.lua')

local debug_data_check = false

local config = {}
config.n_map_all = 18
config.batch_size = tonumber(os.getenv('B'))
config.win_size = 224

-- loading nameList
local namesFile = io.open('atr_nameList.txt')
local idx = 1
local nameList = {}
for line in namesFile:lines() do
    nameList[idx] = line
    idx = idx + 1
end
namesFile:close()
local m = idx - 1

-- model initialization
vgg_model = torch.load('./caffemodel/vgg16_torch_cudnn.t7')
model = nn.Sequential()
for i = 1, 23 do
    model:add(vgg_model:get(i))
end
model:add(nn.SpatialDilatedConvolution(512, 512, 3,3,1,1,2,2,2,2))
model:add(nn.ReLU())
model:add(nn.SpatialDilatedConvolution(512, 512, 3,3,1,1,2,2,2,2))
model:add(nn.ReLU())
model:add(nn.SpatialDilatedConvolution(512, 512, 3,3,1,1,2,2,2,2))
model:add(nn.ReLU())
model:add(nn.SpatialDilatedConvolution(512, 4096, 7,7,1,1,12,12,4,4))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.SpatialConvolution(4096,4096,1,1,1,1,0,0))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.SpatialConvolution(4096,18,1,1,1,1,0,0))
model:add(nn.SpatialUpSamplingBilinear(8))

model.modules[24].weight = vgg_model.modules[25].weight:clone()
model.modules[26].weight = vgg_model.modules[27].weight:clone()
model.modules[28].weight = vgg_model.modules[29].weight:clone()
model.modules[30].weight = vgg_model.modules[33].weight:clone():view(4096,512,7,7)
model.modules[24].bias = vgg_model.modules[25].bias:clone()
model.modules[26].bias = vgg_model.modules[27].bias:clone()
model.modules[28].bias = vgg_model.modules[29].bias:clone()
model.modules[30].bias = vgg_model.modules[33].bias:clone() -- no need to view, 4096
-- bias copy for them

cudnn.convert(model, cudnn)
model:cuda()

-- construct criterion
local criterion = cudnn.SpatialCrossEntropyCriterion()
criterion:cuda()

-- place holder
local input = torch.zeros(config.batch_size, 3, config.win_size, config.win_size):float()
local label = torch.zeros(config.batch_size, config.win_size, config.win_size):float()
input = input:cuda()
label = label:cuda()

-- sample function
print(m)
local img_root = '/home/szzhu/data_trial/humanparsing/'
local trim_head_tail_index = function(vec)
    local v = vec:view(-1)
    local x1, x2
    x1 = 0/0; x2 = 0/0;
    for i = 1, v:size(1) do
        if v[i] > 0 then x1 = i; break; end
    end
    for i = v:size(1), 1, -1 do
        if v[i] > 0 then x2 = i; break; end
    end
    return x1, x2
end
local swap = torch.Tensor{9,10,12,13,14,15}:view(3,2)
local sample = function()
    local ind = torch.randperm(m):narrow(1,1,config.batch_size)
    for i = 1, config.batch_size do

        local img = image.load(img_root .. 'JPEGImages/' .. nameList[ind[i]] .. '.jpg', 3, 'float')
        local lb = image.load(img_root .. 'SegmentationClassAug/' .. nameList[ind[i]] .. '.png', 1, 'byte')
        -- 1. padding to be a square
        local L -- to get the length of the big square
        local img_aug
        local lb_aug
        if img:size(2) >= img:size(3) then -- which is the most comon case
            L = img:size(2)
            img_aug = torch.zeros(3,L,L):float()
            lb_aug = torch.zeros(1,L,L):byte()
            local start = torch.floor((L - img:size(3)) / 2 + 1)
            local finish = start + img:size(3) - 1
            img_aug[{{},{},{start,finish}}] = img
            lb_aug[{{},{},{start,finish}}] = lb
        else
            L = img:size(3)
            img_aug = torch.zeros(3,L,L):float()
            lb_aug = torch.zeros(1,L,L):byte()
            local start = torch.floor((L - img:size(2)) / 2 + 1)
            local finish = start + img:size(2) - 1
            img_aug[{{},{start,finish},{}}] = img
            lb_aug[{{},{start,finish},{}}] = lb
        end

        -- 1B. pad to 3L
        local img_aug_temp = img_aug
        local lb_aug_temp = lb_aug
        img_aug = torch.zeros(3,3*L,3*L):float()
        img_aug[{{},{L+1,2*L},{L+1,2*L}}] = img_aug_temp
        lb_aug = torch.zeros(1,3*L,3*L):byte()
        lb_aug[{{},{L+1,2*L},{L+1,2*L}}] = lb_aug_temp
        L = 3*L

        -- 2. Get the containing boundary
        local x1, x2, y1, y2
        local relax = 5
        local k = torch.max(lb_aug, 2)
        x1, x2 = trim_head_tail_index(k)
        k = torch.max(lb_aug, 3)
        y1, y2 = trim_head_tail_index(k)
        x1 = x1 + relax; y1 = y1 + relax; x2 = x2 - relax; y2 = y2 - relax;

        -- 3. define croping area and crop
        local magic = L / 18
        local margin = 1
        local rand_left = (1 + x2) / 2 + margin
        local rand_right = (x1 + L) / 2 - margin
        assert(rand_left < rand_right)
        if rand_right - rand_left > magic*4 then
            local l = rand_left
            local r = rand_right
            rand_left = (l+r)/2 - magic*2
            rand_right = (l+r)/2 + magic*2
        end
        local rand_up = (1 + y2) / 2 + margin
        local rand_down = (y1 + L) / 2 - margin
        assert(rand_up < rand_down)
        if rand_down - rand_up > magic then
            local u = rand_up
            local d = rand_down
            rand_up = (u+d)/2 - magic/2
            rand_down = (u+d)/2 + magic/2
        end
        local cx = math.random() * (rand_right - rand_left) + rand_left
        local cy = math.random() * (rand_down - rand_up) + rand_up
        local rand_short = 2 * math.max(x2-cx, cx-x1, y2-cy, cy-y1)
        local rand_long = 2 * math.min(L-cx, cx-1, L-cy, cy-1)
        assert(rand_short < rand_long)
        if rand_long - rand_short > magic*2 then
            rand_long = rand_short + magic*2
        end
        local cs = math.random() * (rand_long - rand_short) + rand_short
        cs = math.floor(cs)
        local left = math.floor(cx - cs/2)
        local right = left + cs - 1
        local up = math.floor(cy - cs/2)
        local down = up + cs - 1
        local img_crop = img_aug[{{},{up,down},{left,right}}]
        local lb_crop = lb_aug[{{},{up,down},{left,right}}]

        -- 4. Scale to 224
        local img_scaled = image.scale(img_crop, config.win_size, config.win_size, 'bilinear')
        local lb_scaled = image.scale(lb_crop, config.win_size, config.win_size, 'simple')
        -- 5. Randomly flip
        if math.random() > 0.5 then
            img_scaled = image.hflip(img_scaled)
            lb_scaled = image.hflip(lb_scaled)
            for j = 1, swap:size(1) do
                local t1 = lb_scaled:eq(swap[j][1])
                local t2 = lb_scaled:eq(swap[j][2])
                lb_scaled[t1] = swap[j][2]
                lb_scaled[t2] = swap[j][1]
            end
        end
        -- 6. Add pixel-level noise
        img_scaled = img_scaled + torch.randn(img_scaled:size()):float() * 0.01

        -- 7. write to the place holder
        input[{{i},{},{},{}}] = img_scaled:view(1, 3, config.win_size, config.win_size) - 0.5
        lb_scaled[lb_scaled:eq(0)] = config.n_map_all
        label[{{i},{},{}}] = lb_scaled:view(1, config.win_size, config.win_size)
    end
    if debug_data_check then
        matio.save("debug_data_check.mat", {input = input:float():permute(3,4,2,1), label = label:float():permute(2,3,1)})
        error("debug_data_check_exit_point")
    end
end

local optimState = {learningRate = 0.001, beta1 = 0.5}
local params
local gradParmas
params, gradParams = model:getParameters()
local err
local output
local feval = function()
    model:zeroGradParameters()
    output = model:forward(input)
    err = criterion:forward(output, label)
    local df_do = criterion:backward(output,label)
    model:backward(input,df_do)
    return err, gradParams
end
local tm = torch.Timer()
for iter = 1, 1e9 do
    tm:reset(); tm:resume()
    sample()
    tm:stop()
    local time_data = tm:time().real
    tm:reset(); tm:resume()
    optim.adam(feval, params, optimState)
    tm:stop()
    local time_back = tm:time().real
    local base = 100 * config.batch_size
    if iter % 20 == 0 then
        d(input:float(), 1 + base, 'input')
        d(output:float(), 2 + base, 'output', 'm2c')
        d(label:float():view(config.batch_size, 1, config.win_size, config.win_size), 3 + base, 'label')
    end
    if iter % 1 == 0 then
        print(string.format('Iter %d: Err %.5f. Data %.5fs, Back %.5fs.', iter, err, time_data, time_back))
    end
    if iter % 10000 == 0 then
        torch.save('latest_' .. tostring(config.batch_size) .. '_' .. tostring(base) .. '.t7', model)
    end
end

