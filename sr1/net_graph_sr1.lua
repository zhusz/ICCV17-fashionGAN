require 'nngraph'
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02 / 16)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local config = dofile('./config_sr1.lua')

local nc = config.n_map_all
local ncondition = config.n_condition

local nz = config.nz
local nt_input = config.nt_input
local nt = config.nt

local ndf = 64
local ngf = 64
local inplace = true

local bn4 = nn.SpatialBatchNormalization
local conv = nn.SpatialConvolution
local deconv = nn.SpatialFullConvolution
local relu = nn.ReLU
local lerelu = nn.LeakyReLU

local input_data = nn.Identity()()
---------------------------------------------------
local input_encode = nn.Identity()()
local h1 = input_encode - conv(nt_input, nt, 1, 1) - lerelu(0.2, inplace)
---------------------------------------------------
local input_data_encode = nn.JoinTable(2)({input_data, h1})
local g1 = input_data_encode - deconv(nz+nt, ngf*16, 4,4) - bn4(ngf*16) - relu(inplace)
local g_extra = g1 - deconv(ngf*16,ngf*8,4,4,2,2,1,1) - bn4(ngf*8) - relu(inplace)

local input_condition = nn.Identity()()
local f1 = input_condition - conv(ncondition, ngf, 3,3,1,1,1,1) - bn4(ngf) - lerelu(0.2, inplace)
local f_extra = f1 - conv(ngf*1, ngf*2, 3,3,1,1,1,1) - bn4(ngf*2) - lerelu(0.2, inplace)

local g2 = nn.JoinTable(2)({g_extra, f_extra}) - deconv(ngf*10, ngf*4, 4,4,2,2,1,1) - bn4(ngf*4) - relu(inplace)

local m1 = g2 - conv(ngf*4, ngf*8, 3,3,1,1,1,1) - bn4(ngf*8) - lerelu(0.2, inplace)
local m2 = m1 - conv(ngf*8, ngf*8, 3,3,1,1,1,1) - bn4(ngf*8) - lerelu(0.2, inplace)
local m3 = m2 - conv(ngf*8, ngf*4, 3,3,1,1,1,1) - bn4(ngf*4) - lerelu(0.2, inplace)

local g3 = m3 - deconv(ngf*4, ngf*2, 4,4,2,2,1,1) - bn4(ngf*2) - relu(inplace)
local g4 = g3 - deconv(ngf*2, ngf*1, 4,4,2,2,1,1) - bn4(ngf*1) - relu(inplace)
local g5 = g4 - deconv(ngf, nc, 4,4,2,2,1,1)

local netG = nn.gModule({input_data, input_encode, input_condition},{g5})
netG:apply(weights_init)

local output_data = nn.Identity()()
--
local output_data_softmax = output_data - cudnn.SpatialSoftMax()
--
local d1 = output_data_softmax - conv(nc,ndf,4,4,2,2,1,1) - lerelu(0.2, inplace)
local d2 = d1 - conv(ndf, ndf*2, 4,4,2,2,1,1) - bn4(ndf*2) - lerelu(0.2, inplace)
local d3 = d2 - conv(ndf*2, ndf*4, 4,4,2,2,1,1) - bn4(ndf*4) - lerelu(0.2, inplace)

local mid1 = d3 - conv(ndf*4, ndf*8, 3,3,1,1,1,1) - bn4(ndf*8) - lerelu(0.2, inplace)
local mid2 = mid1-conv(ndf*8, ndf*8, 3,3,1,1,1,1) - bn4(ndf*8) - lerelu(0.2, inplace)
local mid3 = mid2-conv(ndf*8, ndf*4, 3,3,1,1,1,1) - bn4(ndf*4) - lerelu(0.2, inplace)

local d4 = mid3 - conv(ndf*4, ndf*8, 4,4,2,2,1,1) - bn4(ndf*8) - lerelu(0.2, inplace)

local output_condition = nn.Identity()()
local c1 = output_condition - conv(ncondition,ndf,3,3,1,1,1,1) - lerelu(0.2, inplace)
local c2 = c1 - conv(ndf, ndf*2, 3,3,1,1,1,1) - bn4(ndf*2) - lerelu(0.2, inplace)

local d_extra = nn.JoinTable(2)({d4,c2}) - conv(ndf*10, ndf*8, 4,4,2,2,1,1) - bn4(ndf*8) - lerelu(0.2, inplace)
---------------------------------------------------
local output_encode = nn.Identity()()
local b1 = output_encode - conv(nt_input, nt, 1, 1) - bn4(nt) - lerelu(0.2, inplace) - nn.Replicate(4,3) - nn.Replicate(4,4)
local d_extra_b1 = nn.JoinTable(2)({d_extra, b1}) - conv(ndf*8+nt,ndf*8,1,1) - bn4(ndf*8) - lerelu(0.2, inplace)
---------------------------------------------------
local d5 = d_extra_b1 - conv(ndf*8, 1, 4, 4) - nn.Sigmoid()

local netD = nn.gModule({output_data, output_encode, output_condition},{d5})
netD:apply(weights_init)

return netG, netD
