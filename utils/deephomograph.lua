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

-- initialization

-- fix seed 
torch.manualSeed(1)

-- threads
-- torch.setnumberThreads(4)
-- print('====> set nb of threads to 4.')

-- use floats for SGD
torch.setdefaulttensortype('torch.FloatTensor')

-- batch size

-- load dataset
-- image data
dataPath = '/Users/bragi/Computer_Vision/DeepHomography/graffiti/homographyData.t7'
trainingData = torch.load(dataPath)
-- for key, value in pairs(trainingData) do
-- 	-- print(key)
-- 	for k,v in pairs(value) do
-- 		print(v)
-- 	end
-- end

-- ground truth
homography = {}
for key,value in pairs(trainingData) do
	local h = {}
	for k,v in pairs(value) do
		local hh = torch.Tensor(8):fill(1.0)
		table.insert(h, hh)
	end
	homography[key] = h
end

for key,value in pairs(homography) do
	-- print(key)
	for k, v in pairs(value) do
		-- print(v)
	end
end

-- define the neural network

function creatModel()
	local net = nn.Sequential()
	    
	-- layer 1
	-- (input, output, kernal width, kernal height, stride width, stride height, padding one, padding two )
	net:add(nn.SpatialConvolution(2,64,3,3,1,1,1,1))
	-- (batch size)
	-- net:add(nn.SpatialBatchNormalization(64))
	net:add(nn.ReLU())

	-- layer 2 
	net:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
	-- net:add(nn.SpatialBatchNormalization(64))
	net:add(nn.ReLU())
	-- max pooling 
	-- (pooling width, pooling height, stride width, stide height, padding one, padding two)
	net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

	-- layer 3
	net:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
	-- net:add(nn.SpatialBatchNormalization(64))
	net:add(nn.ReLU())

	-- layer 4
	net:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
	-- net:add(nn.SpatialBatchNormalization(64))
	net:add(nn.ReLU())
	-- max pooling 
	net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

	-- layer 5
	net:add(nn.SpatialConvolution(64,128,3,3,1,1,1,1))
	-- net:add(nn.SpatialBatchNormalization(128))
	net:add(nn.ReLU())

	-- layer 6
	net:add(nn.SpatialConvolution(128,128,3,3,1,1,1,1))
	-- net:add(nn.SpatialBatchNormalization(128))
	net:add(nn.ReLU())
	-- max pooling 
	net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

	-- layer 7
	net:add(nn.SpatialConvolution(128,128,3,3,1,1,1,1))
	-- net:add(nn.SpatialBatchNormalization(128))
	net:add(nn.ReLU())

	-- layer 8
	net:add(nn.SpatialConvolution(128,128,3,3,1,1,1,1))
	-- net:add(nn.SpatialBatchNormalization(128))
	net:add(nn.ReLU())
	-- dropout with 0.5
	net:add(nn.Dropout(0.500000))
	    
	-- first fully connected layer
	-- (16*16*128)
	net:add(nn.View(32768))
	net:add(nn.Linear(32768,1024))
	-- dropout with 0.5
	net:add(nn.Dropout(0.500000))
	    
	-- final fully connected layer 
	net:add(nn.Linear(1024, 8))
	net:add(nn.LogSoftMax())
	return net
end

model = creatModel()
-- retrieve parameters and gradients
-- parameters,gradParameters = model:gradParameters()
-- neural netowrk structure
-- print('Deep Homography Network\n' .. model:__tostring());

-- define the loss function
criterion = nn.MSECriterion()
criterion.sizeAverage = false

-- train the neural network
model:zeroGradParameters()
function train(trainingData)
	epoch = 1
	local time = sys.clock()
	print('====> doing epoch on training data:')
	print("====> online epoch # " .. epoch)

	for class, images in pairs(trainingData) do
		print(class)
		for index, image in pairs(images) do
			local output = model:forward(image)
			print(output)
			print(homography[class][index])
			local err = criterion:forward(output, homography[class][index])
			local gradients = criterion:backward(output, homography[class][index])
			model:backward(image, gradients)
			model:updateParameters(0.005)
		end
	end
end
train(trainingData)
-- print(model:forward(trainingData['bark'][1]))



