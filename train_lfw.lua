------------------------------------------------------------
--- This code is based on the eyescream code released at
--- https://github.com/facebook/eyescream
--- If you find it usefull consider citing
--- http://arxiv.org/abs/1506.05751
------------------------------------------------------------

require 'hdf5'
require 'nngraph'
require 'cudnn'
require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
adversarial = require 'lfw_adverserial'


----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "logs512_lfw64")      subdirectory to save logs
  --saveFreq         (default 1)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.001)        learning rate
  -b,--batchSize     (default 128)         batch size
  -m,--momentum      (default 0)           momentum, for SGD only
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)           L2 penalty on the weights
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default 0)           gpu to run on (default cpu)
  -d,--noiseDim      (default 512)         dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           windsow id of sample image
  --scale            (default 64)          scale of images to train on
]]


if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = false end

print(opt)

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.gpu then
  cutorch.setDevice(opt.gpu + 1)
  print('<gpu> using device ' .. opt.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end

opt.geometry = {3, opt.scale, opt.scale}

local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]

if opt.network == '' then
  ----------------------------------------------------------------------
  -- define D network to train
  model_D = nn.Sequential()
  model_D:add(cudnn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
  model_D:add(cudnn.SpatialMaxPooling(2,2))
  model_D:add(cudnn.ReLU(true))
  model_D:add(nn.SpatialDropout(0.2))
  model_D:add(cudnn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
  model_D:add(cudnn.SpatialMaxPooling(2,2))
  model_D:add(cudnn.ReLU(true))
  model_D:add(nn.SpatialDropout(0.2))
  model_D:add(cudnn.SpatialConvolution(64, 96, 5, 5, 1, 1, 2, 2))
  model_D:add(cudnn.ReLU(true))
  model_D:add(cudnn.SpatialMaxPooling(2,2))
  model_D:add(nn.SpatialDropout(0.2))
  model_D:add(nn.Reshape(8*8*96))
  model_D:add(nn.Linear(8*8*96, 1024))
  model_D:add(cudnn.ReLU(true))
  model_D:add(nn.Dropout())
  model_D:add(nn.Linear(1024,1))
  model_D:add(nn.Sigmoid())

  x_input = nn.Identity()()
  lg = nn.Linear(opt.noiseDim, 128*8*8)(x_input)
  lg = nn.Reshape(128, 8, 8)(lg)
  lg = cudnn.ReLU(true)(lg)
  lg = nn.SpatialUpSamplingNearest(2)(lg)
  lg = cudnn.SpatialConvolution(128, 256, 5, 5, 1, 1, 2, 2)(lg)
  lg = nn.SpatialBatchNormalization(256)(lg)
  lg = cudnn.ReLU(true)(lg)
  lg = nn.SpatialUpSamplingNearest(2)(lg)
  lg = cudnn.SpatialConvolution(256, 256, 5, 5, 1, 1, 2, 2)(lg)
  lg = nn.SpatialBatchNormalization(256)(lg)
  lg = cudnn.ReLU(true)(lg)
  lg = nn.SpatialUpSamplingNearest(2)(lg)
  lg = cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2)(lg)
  lg = nn.SpatialBatchNormalization(128)(lg)
  lg = cudnn.ReLU(true)(lg)
  lg = cudnn.SpatialConvolution(128, 3, 3, 3, 1, 1, 1, 1)(lg)
  model_G = nn.gModule({x_input}, {lg})

else
  print('<trainer> reloading previously trained network: ' .. opt.network)
  tmp = torch.load(opt.network)
  model_D = tmp.D
  model_G = tmp.G
end

-- loss function: negative log-likelihood
criterion = nn.BCECriterion()

-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()

-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)


local lfwHd5 = hdf5.open('datasets/lfw.hdf5', 'r')
local data = lfwHd5:read('lfw'):all()
data:mul(2):add(-1)
lfwHd5:close()


ntrain = 13000
nval = 233
trainData = data[{{1, ntrain}}]
valData = data[{{ntrain, nval+ntrain}}]


-- this matrix records the current confusion across classes
classes = {'0','1'}
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

if opt.gpu then
  print('Copy model to gpu')
  model_D:cuda()
  model_G:cuda()
end

-- Training parameters
sgdState_D = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  optimize=true,
  numUpdates = 0
}

sgdState_G = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  optimize=true,
  numUpdates=0
}

-- Get examples to plot
function getSamples(dataset, N)
  local numperclass = numperclass or 10
  local N = N or 8
  local noise_inputs = torch.Tensor(N, opt.noiseDim)

  -- Generate samples
  noise_inputs:normal(0, 1)
  local samples = model_G:forward(noise_inputs)
  samples = nn.HardTanh():forward(samples)
  local to_plot = {}
  for i=1,N do
    to_plot[#to_plot+1] = samples[i]:float()
  end

  return to_plot
end


-- training loop
while true do
  local to_plot = getSamples(valData, 100)
  torch.setdefaulttensortype('torch.FloatTensor')

  trainLogger:style{['% mean class accuracy (train set)'] = '-'}
  testLogger:style{['% mean class accuracy (test set)'] = '-'}
  trainLogger:plot()
  testLogger:plot()

  local formatted = image.toDisplayTensor({input=to_plot, nrow=10})
  formatted:float()
  image.save(opt.save .."/lfw_example_v1_"..(epoch or 0)..'.png', formatted)
  if opt.gpu then
    torch.setdefaulttensortype('torch.CudaTensor')
  else
    torch.setdefaulttensortype('torch.FloatTensor')
  end


  -- train/test
  adversarial.train(trainData)
  adversarial.test(valData)

  sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
  sgdState_D.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)
  sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
  sgdState_G.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)


end
