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

local train = {}
imgSize = {128,128}
outputSize = 8
-- save log file
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
function train.process(model, criterion, trainData, trainGT, testData, testGT, opt, optimS, epoch)

	local epochLoss = 0.0

	if opt.cuda then
		model:cuda()
		criterion:cuda()
	end

	if model then
		parameters,gradParameters = model:getParameters()
	end

	local batchSize = opt.batchSize
	local dataSize = #trainData
	-- configure optimization
	optimState = optimS
	optimMethod = optim.sgd

	print('==========> taining procedure')
	print('==========> doing epoch on training data')
	print("==========> epoch # " .. epoch)

	model:training()

	local tm = torch.Timer()

	for i = 1, dataSize, batchSize do
		-- disp progress
		xlua.progress(i, dataSize)

		-- create mini-batch
		local inputData = torch.Tensor(batchSize, 2, imgSize[1], imgSize[2])
		local groudTruth = torch.Tensor(batchSize, outputSize)

		local k = 1
		for j = i, math.min(i+batchSize-1, dataSize) do
			inputData[k] = torch.Tensor(2, 128,128):copy(trainData[j])
			groudTruth[k] = torch.Tensor(8):copy(trainGT[j])
			k = k + 1
		end
		
		if opt.cuda then
			inputData = inputData:cuda()
			groudTruth = groudTruth:cuda()
		end
		
		collectgarbage()

		local feval = function(x)
			if x ~= parameters then
				parameters:copy(x)
			end

			--reset gradients
			gradParameters:zero()

			local outputs = model:forward(inputData)
			local loss = criterion:forward(outputs,groudTruth)

			local gradients = criterion:backward(outputs,groudTruth)
			model:backward(inputData, gradients)

			gradParameters:div(inputData:size()[1])

			epochLoss = epochLoss + loss

			return loss,gradParameters
		end
		-- print(inputData[1])
		-- print(groudTruth[1])

		optimMethod(feval, parameters, optimState)

	end

	epochLoss = epochLoss / dataSize
	trainLogger:add{['loss'] = epochLoss,}
	print('==========> training loss: ', epochLoss)
	print('\n')
	time = tm:time().real
	timeSample = time / dataSize
	print('Time is: ', time)
	print('Time per sample is: ', timeSample)
	torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model:float():clearState())
	torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)

	-- test dataset

	local testDataSize = #testData
	model:evaluate()
	local testError = 0.0
	for iT = 1, testDataSize, batchSize do
		-- disp progress
		xlua.progress(iT, testDataSize)

		-- create mini-batch
		local testInputData = torch.Tensor(batchSize, 2, imgSize[1], imgSize[2])
		local testGroudTruth = torch.Tensor(batchSize, outputSize)

		local kT = 1
		for jT = iT, math.min(i+batchSize-1, testDataSize) do
			testInputData[kT] = torch.Tensor(2, 128,128):copy(testData[jT])
			testGroudTruth[kT] = torch.Tensor(8):copy(testGT[jT])
			kT = kT + 1
		end
		
		if opt.cuda then
			testInputData = testInputData:cuda()
			testGroudTruth = testGroudTruth:cuda()
		end

		testOutputs = model:forward(testInputData)
		local batchError = 0.0
		for mO = 1,batchSize do
			local meanError = torch.Tensor(8):fill(0)
			for nO = 1, outputSize do
				meanError[nO] = torch.pow((testOutputs[mO][nO] - testGroudTruth[mO][nO]), 2)
			end
			batchError = batchError + torch.sqrt(torch.cumsum(meanError))
		end
		testError = testError + batchError / batchSize
		testLogger:add{['error'] = testError,}
		print('==========> testing error: ', testError)
		print('\n')
end

return train
















