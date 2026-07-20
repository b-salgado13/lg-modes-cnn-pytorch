imgPath ='/Volumes/INGFISICA/Red Neuronal Modos Espaciales de Luz/Fotos 128x192';
imds = imageDatastore(imgPath,'IncludeSubfolders',true, ...
    'FileExtensions','.jpg','LabelSource','foldernames');
numTrainFiles = 0.7;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
img = readimage(imds,1);
[ren,col] = size(img);

layers = [
    imageInputLayer([ren col 1])

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(19)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',20,...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',20, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
save('CNN-8-8-8.mat', 'net');
