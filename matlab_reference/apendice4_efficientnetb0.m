imgPath ='/Volumes/INGFISICA/Red Neuronal Modos Espaciales de Luz/Fotos 224x224x3';
imds = imageDatastore(imgPath,'IncludeSubfolders',true, ...
    'FileExtensions','.jpg','LabelSource','foldernames');
numTrainFiles = 0.7;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
[ren,col] = size(readimage(imds,1));
% Cargar la red preentrenada
net= efficientnetb0;
%Sustituir capas finales
lgraph = layerGraph(net);
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
%Se cambia las capas finales para coincidir con el número de clases
numClasses = numel(categories(imdsTrain.Labels));
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
%% Congelar las capas iniciales
layers = lgraph.Layers;
connections = lgraph.Connections;
flayer=286;
layers(1:flayer) = freezeWeights(layers(1:flayer));
lgraph = createLgraphUsingConnections(layers,connections);
%% Establecer las opciones de entrenamiento
Minibatch=70;
Valfreq=floor(numel(imdsTrain.Files)/Minibatch);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',Minibatch,'MaxEpochs', 10,...
    'InitialLearnRate',1e-3,'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation,'ValidationFrequency',Valfreq, ...
    'Verbose',false, 'Plots','training-progress');
net = trainNetwork(imdsTrain,lgraph,options);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = mean(YPred == YValidation)
