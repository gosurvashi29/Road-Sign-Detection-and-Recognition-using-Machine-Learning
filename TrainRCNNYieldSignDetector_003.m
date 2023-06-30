
%% Train R-CNN Stop Sign Detector
% 


%%
clc, clear, close all

% Load training data and network layers.
%%%%load('rcnnStopSigns.mat', 'stopSigns', 'layers')


% Load network layers:
load('myRCNN_Layers.mat')




% Yield_Sign
% Load training data:
load('YIELD2\yield2_label_definations.mat');
roadSignYield =  table(gTruth.DataSource.Source, table2array(gTruth.LabelData) );


%%   
% Add the image directory to the MATLAB path.
%%%%%imDir = fullfile(matlabroot, 'toolbox', 'vision', 'visiondata','stopSignImages');
imDir = '.\YIELD2';
addpath(imDir);
%%
% Set network training options to use mini-batch size of 32 to reduce GPU
% memory usage. Lower the InitialLearnRate to reduce the rate at which
% network parameters are changed. This is beneficial when fine-tuning a
% pre-trained network and prevents the network from changing too rapidly. 
options = trainingOptions('sgdm', ...
  'MiniBatchSize', 32, ...
  'InitialLearnRate', 1e-6, ...
  'MaxEpochs', 10);

%%
% Train the R-CNN detector. Training can take a few minutes to complete.
rcnn = trainRCNNObjectDetector(roadSignYield, layers, options, 'NegativeOverlapRange', [0 0.3]);

%%
% Test the R-CNN detector on a test image.
img = imread('YIELD2\27.jpeg'); 
[bbox, score, label] = detect(rcnn, img, 'MiniBatchSize', 32);
[score, idx] = max(score); % Display strongest detection result.
bbox = bbox(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);
detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation);
figure
imshow(detectedImg)
%%
% Remove the image directory from the path.
rmpath(imDir); 
