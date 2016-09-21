#!/usr/bin/env th
require 'nn'
require 'sys'
-- require 'dpnn'
require 'torch'
require 'optim'
require 'image'
require 'paths'
require 'xlua'
require 'image'
require 'paths'
require 'torch'
require 'torchx'


print('<==================== define loss function ====================>')

criterion = nn.MSECriterion()
-- criterion.sizeAverage = false
print(criterion)

return criterion