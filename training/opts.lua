#!/usr/bin/env th
require 'nn'
require 'sys'
require 'xlua'
-- require 'dpnn'
require 'torch'
require 'optim'
require 'image'
require 'paths'
require 'image'
require 'paths'
require 'torch'
require 'torchx'

local M = { }

-- http://stackoverflow.com/questions/6380820/get-containing-path-of-lua-file
function script_path()
   local str = debug.getinfo(2, "S").source:sub(2)
   return str:match("(.*/)")
end

function M.parse(arg)

   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Deep Homography')
   cmd:text()
   cmd:text('Options:')

   ------------ General options --------------------
   -- cmd:option('-cache', paths.concat(script_path(), 'work'), 
      -- 'Directory to cache experiments and data.')
   cmd:option('-save', '', 'Directory to save experiment.')
   cmd:option('-cache', paths.concat(script_path(), 'work'), 'Directory to cache experiments and data.')
   cmd:option('-data', paths.concat(os.getenv('HOME'), 'DeepHomography', 'data'), 'Directory of training dataset')
   cmd:option('-nEpochs', 100, 'Number of total epochs to run')
   cmd:option('-manualSeed', 1, 'manually set RNG seed')
   cmd:option('-threads', 2, 'Number of threads')
   cmd:option('-optimization', 'SGD', 'optimization method')
   cmd:option('-learningRate', {0.01, 0.005, 0.001, 0.0005, 0.0001}, 'learning rate')
   cmd:option('-weightDecay', 0, 'weight Decay' )
   cmd:option('-learningRateDecay', 0.1, 'learning rate decay')
   cmd:option('-batchSize', 10, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-momentum', 0.9, 'momentum (SGD only)')
   cmd:option('-maxIteration', 20000, 'maximum nb of iterations for parameters update')
   
   cmd:option('-cuda', true, 'Use cuda.')
   cmd:option('-device', 1, 'Cuda device to use.')
   cmd:option('-nGPU',   1,  'Number of GPUs to use by default')
   cmd:option('-cudnn', true, 'Convert the model to cudnn.')
   cmd:option('-cudnn_bench', false, 'Run cudnn to choose fastest option. Increase memory usage')

   cmd:option('-test256', false, 'Run testing with 256*256 images')
   cmd:text()

   local opt = cmd:parse(arg or {})
   os.execute('mkdir -p ' .. opt.cache)

   if opt.save == '' then
      opt.save = paths.concat(opt.cache, os.date("%Y-%m-%d_%H-%M-%S"))
   end
   os.execute('mkdir -p ' .. opt.save)

   return opt
end

return M
