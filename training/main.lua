#!/usr/bin/env th

require 'nn'
require 'sys'
require 'xlua'
-- require 'dpnn'
require 'torch'
require 'optim'
require 'paths'
require 'image'
require 'paths'
require 'torch'
require 'torchx'

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)
print(opt)
torch.save(paths.concat(opt.save, 'opts.t7'), opt, 'ascii')
print('Saving everything to: ' .. opt.save)

torch.setdefaulttensortype('torch.FloatTensor')
if opt.cuda then
   print('\n<==================== CUDA support ====================>\n')
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.device)
end

torch.setnumthreads(opt.threads)
torch.manualSeed(opt.manualSeed)

print('<==================== execute all files ====================>')

local trainData,trainGT,testData,testGT = paths.dofile('data.lua')
local model = require 'models'
local criterion = paths.dofile('loss.lua')
local train = paths.dofile('train.lua')

local epoch = 1

local lr = opt.learningRate

local mini_loss = 100000.0
local model_number = 0

print('<====================> train network ====================>')
for _=1, opt.nEpochs do
	local lr_index = math.floor((epoch / 20))
	local optimState = {
		learningRate = lr[lrIndex],
		weightDecay = opt.weightDecay,
		momentum = opt.momentum,
		learningRateDecay = 0.1
	   	}

	local trained_model, loss = train.process(model, criterion, trainData, trainGT, testData, testGT, opt, optimState, epoch)
	if loss < mini_loss then
		mini_loss = loss
		model_path = paths.concat(opt.save, 'model_' .. epoch .. '.t7')
		optimState_path = paths.concat(opt.save, 'optimState_' .. epoch .. '.t7')
		torch.save(model_path, trained_model:float():clearState())
		torch.save(optimState_path, optimState)

		if epoch ~= 1 then
			for j = 1, epoch-1 do
				local old_model_path = paths.concat(opt.save, 'model_' .. j .. '.t7')
				local old_optimState_path = paths.concat(opt.sva, 'optimState_' .. j .. '.t7')

				os.execute('rm -f ' .. old_model_path)
				os.execute('rm -f ' .. old_optimState_path)
			end
		end
	end
	epoch = epoch + 1
end

-- if opt.test do
-- 	paths.dofile('test.lua')
-- 	test()
-- end
