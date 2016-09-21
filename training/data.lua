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

local D = {}
function D.trainDataset(continue)

	torch.setdefaulttensortype('torch.FloatTensor')
	local dataPath = opt.data
	local trainDataPath = paths.concat(dataPath, 'trainData')

	if not paths.filep(paths.concat(opt.cache, 'trainData.t7')) then
		print('<==================== process raw traininig data ====================>')
		-- local imagesData = {}
		local imagesGT = {}
		local imagesTensor = {}
		for dir in paths.iterdirs(trainDataPath) do
			if dir ~= '.DS_Store' then
				local dataDirPath = paths.concat(trainDataPath,dir)
				-- print(dataDirPath)
			   	local images = {}
			   	local gt = {}
			   	local file_table = paths.dir(dataDirPath)
			   	-- print(#file_table)

				for file in paths.iterfiles(dataDirPath) do
					-- load images and generate image paris
			   		if file ~= '.DS_Store' and string.find(file, '.txt') == nil then
			   			imagePath = paths.concat(dataDirPath, file)
			   			-- print(imagePath)
			   			local img = image.scale(image.load(imagePath,1,'float'),128,128)
			   			-- print(img)
			   			table.insert(images,img)
				   	end	

				   	if file ~= '.DS_Store' and string.find(file, '.txt') ~= nil then
				   		homography_path = paths.concat(dataDirPath, file)
				   		local f = io.open(homography_path, 'rb')
				   		-- local data = f:read('*all')
				   		-- print(type(data))
				   		local h = {}
				   		local l = io.lines(homography_path)
				   		local i = 1
				   		local j = 1
				   		for line in l do
					   		if i % 9 ~= 0 then
					   			h[j] = line 
					   			j = j + 1
					   		end
					   		i = i + 1
						end

						local pairs_number = j / 8
						local h_align = torch.Tensor(pairs_number,8)
						local p = 1
						for m = 1, pairs_number do
							for n = 1, 8 do
								-- print(m)
								h_align[m][n] = h[p]
								p = p + 1
							end
						end
						for p = 1,pairs_number do
							table.insert(imagesGT,h_align[p])
						end
				   	end
		 		end

			   	image_number = #images
			   	image_size = images[1]:size()
			   	-- local imagesTensor = {}
			   	local image_number = 1
			   	while image_number < #images do
			   		local imagePairs = torch.Tensor(2,128,128):fill(1)
			   		imagePairs[1] = torch.Tensor(image_size):copy(images[image_number])
			   		image_number = image_number + 1
			   		imagePairs[2] = torch.Tensor(image_size):copy(images[image_number])
			   		image_number = image_number + 1
			   		table.insert(imagesTensor, imagePairs)
			   	end
		   	end
		end
		print(#imagesTensor)
		print(#imagesGT)

		torch.save(paths.concat(opt.cache, 'trainData.t7'), imagesTensor)
		torch.save(paths.concat(opt.cache, 'trainGT.t7'), imagesGT)
	end
end

funtion D.testDataset(continue)
	
	torch.setdefaulttensortype('torch.FloatTensor')
	local dataPath = opt.data
	local testDataPath = paths.concat(dataPath, 'testData')

	if not paths.filep(paths.concat(opt.cache, 'testData.t7')) then
		print('<==================== process raw testing data ====================>')
		-- local imagesData = {}
		local imagesGT = {}
		local imagesTensor = {}
		for dir in paths.iterdirs(testDataPath) do
			if dir ~= '.DS_Store' then
				dataDirPath = paths.concat(testDataPath,dir)
				-- print(dataDirPath)
			   	local images = {}
			   	local gt = {}
			   	local file_table = paths.dir(dataDirPath)
			   	-- print(#file_table)

				for file in paths.iterfiles(dataDirPath) do
					-- load images and generate image paris
			   		if file ~= '.DS_Store' and string.find(file, '.txt') == nil then
			   			imagePath = paths.concat(dataDirPath, file)
			   			-- print(imagePath)
			   			local img = image.scale(image.load(imagePath,1,'float'),128,128)
			   			-- print(img)
			   			table.insert(images,img)
				   	end	

				   	if file ~= '.DS_Store' and string.find(file, '.txt') ~= nil then
				   		homography_path = paths.concat(dataDirPath, file)
				   		local f = io.open(homography_path, 'rb')
				   		-- local data = f:read('*all')
				   		-- print(type(data))
				   		local h = {}
				   		local l = io.lines(homography_path)
				   		local i = 1
				   		local j = 1
				   		for line in l do
					   		if i % 9 ~= 0 then
					   			h[j] = line 
					   			j = j + 1
					   		end
					   		i = i + 1
						end

						local pairs_number = j / 8
						local h_align = torch.Tensor(pairs_number,8)
						local p = 1
						for m = 1, pairs_number do
							for n = 1, 8 do
								-- print(m)
								h_align[m][n] = h[p]
								p = p + 1
							end
						end
						for p = 1,pairs_number do
							table.insert(imagesGT,h_align[p])
						end
				   	end
		 		end

			   	image_number = #images
			   	image_size = images[1]:size()
			   	-- local imagesTensor = {}
			   	local image_number = 1
			   	while image_number < #images do
			   		local imagePairs = torch.Tensor(2,128,128):fill(1)
			   		imagePairs[1] = torch.Tensor(image_size):copy(images[image_number])
			   		image_number = image_number + 1
			   		imagePairs[2] = torch.Tensor(image_size):copy(images[image_number])
			   		image_number = image_number + 1
			   		table.insert(imagesTensor, imagePairs)
			   	end
		   	end
		end
		print(#imagesTensor)
		print(#imagesGT)

		torch.save(paths.concat(opt.cache, 'testData.t7'), imagesTensor)
		torch.save(paths.concat(opt.cache, 'testGT.t7'), imagesGT)
	end

	-- load data
	print('<==================== loading training data ====================>')

	trainCacheDataCachePath = paths.concat(opt.cache, 'trainData.t7')
	trainCacheGTPath = paths.concat(opt.cache, 'trainGT.t7')

	trainingData = torch.load(trainCacheDataCachePath)
	trainGroundTruth = torch.load(trainCacheGTPath)
	return trainingData, trainGroundTruth
end

function D.testDataset256(continue)

	torch.setdefaulttensortype('torch.FloatTensor')
	local dataPath = opt.data
	local testDataPath = paths.concat(dataPath, 'testData256')

	if not paths.filep(paths.concat(opt.cache, 'testData256.t7')) then
		print('<==================== process raw testing data ====================>')
		-- local imagesData = {}
		local imagesGT = {}
		local imagesTensor = {}
		for dir in paths.iterdirs(testDataPath) do
			if dir ~= '.DS_Store' then
				dataDirPath = paths.concat(testDataPath,dir)
				-- print(dataDirPath)
			   	local images = {}
			   	local gt = {}
			   	local file_table = paths.dir(dataDirPath)
			   	-- print(#file_table)

				for file in paths.iterfiles(dataDirPath) do
					-- load images and generate image paris
			   		if file ~= '.DS_Store' and string.find(file, '.txt') == nil then
			   			imagePath = paths.concat(dataDirPath, file)
			   			-- print(imagePath)
			   			local img = image.scale(image.load(imagePath,1,'float'),128,128)
			   			-- print(img)
			   			table.insert(images,img)
				   	end	

				   	if file ~= '.DS_Store' and string.find(file, '.txt') ~= nil then
				   		homography_path = paths.concat(dataDirPath, file)
				   		local f = io.open(homography_path, 'rb')
				   		-- local data = f:read('*all')
				   		-- print(type(data))
				   		local h = {}
				   		local l = io.lines(homography_path)
				   		local i = 1
				   		local j = 1
				   		for line in l do
					   		if i % 9 ~= 0 then
					   			h[j] = line 
					   			j = j + 1
					   		end
					   		i = i + 1
						end

						local pairs_number = j / 8
						local h_align = torch.Tensor(pairs_number,8)
						local p = 1
						for m = 1, pairs_number do
							for n = 1, 8 do
								-- print(m)
								h_align[m][n] = h[p]
								p = p + 1
							end
						end
						for p = 1,pairs_number do
							table.insert(imagesGT,h_align[p])
						end
				   	end
		 		end

			   	image_number = #images
			   	image_size = images[1]:size()
			   	-- local imagesTensor = {}
			   	local image_number = 1
			   	while image_number < #images do
			   		local imagePairs = torch.Tensor(2,128,128):fill(1)
			   		imagePairs[1] = torch.Tensor(image_size):copy(images[image_number])
			   		image_number = image_number + 1
			   		imagePairs[2] = torch.Tensor(image_size):copy(images[image_number])
			   		image_number = image_number + 1
			   		table.insert(imagesTensor, imagePairs)
			   	end
		   	end
		end
		print(#imagesTensor)
		print(#imagesGT)

		torch.save(paths.concat(opt.cache, 'testData256.t7'), imagesTensor)
		torch.save(paths.concat(opt.cache, 'testGT256.t7'), imagesGT)
	end
	print('<==================== loading testing data ====================>')

	testCacheDataPath = paths.concat(opt.cache, 'testData.t7')
	testCacheGTPath = paths.concat(opt.cache, 'testGT.t7')

	testingData = torch.load(testCacheDataPath)
	testingGroundTruth = torch.load(testCacheGTPath)
	return testingData, testingGroundTruth
end
if opt.test256 then
	return D.trainDataset, D.testDataset256
end
return D.trainDataset, D.testDataset
