%% 🧭 SETUP & PREPROCESSING
% Set paths
coverFolder = 'C:\Users\YourUsername\Desktop\cover_dataset\';
secretFolder = 'C:\Users\YourUsername\Desktop\secret_dataset\';
outputFolder = 'C:\Users\YourUsername\Desktop\stego_images_output\';
mkdir(outputFolder);

% Standard image size
imageSize = [256 256 3];

% Load and resize datasets
coverDS = imageDatastore(coverFolder);
secretDS = imageDatastore(secretFolder);

coverDS.ReadFcn = @(filename) imresize(im2double(imread(filename)), imageSize(1:2));
secretDS.ReadFcn = @(filename) imresize(im2double(imread(filename)), imageSize(1:2));

%% 🧠 EMBEDDING NETWORK DEFINITION
inputCover = imageInputLayer(imageSize,'Name','cover_input');
inputSecret = imageInputLayer(imageSize,'Name','secret_input');

coverLayers = [
    convolution2dLayer(3,16,'Padding','same','Name','conv1_cover')
    reluLayer('Name','relu1_cover')];

secretLayers = [
    convolution2dLayer(3,16,'Padding','same','Name','conv1_secret')
    reluLayer('Name','relu1_secret')];

merge = depthConcatenationLayer(2,'Name','concat');

mergeLayers = [
    convolution2dLayer(3,32,'Padding','same','Name','conv2')
    reluLayer('Name','relu2')
    convolution2dLayer(3,3,'Padding','same','Name','stego_output')
    regressionLayer('Name','reg_output')];

lgraph = layerGraph();
lgraph = addLayers(lgraph, [inputCover; coverLayers]);
lgraph = addLayers(lgraph, [inputSecret; secretLayers]);
lgraph = addLayers(lgraph, merge);
lgraph = addLayers(lgraph, mergeLayers);

lgraph = connectLayers(lgraph,'relu1_cover','concat/in1');
lgraph = connectLayers(lgraph,'relu1_secret','concat/in2');
lgraph = connectLayers(lgraph,'concat','conv2');

%% 🏋 TRAIN EMBEDDING NETWORK
% Create combined training datastore
combinedDS = combine(coverDS, secretDS);
trainDS = transform(combinedDS, @(data) struct( ...
    'cover_input',data{1}, ...
    'secret_input',data{2}, ...
    'reg_output',data{1}));  % Using cover image as dummy target

options = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false);

netEmbed = trainNetwork(trainDS, lgraph, options);

%% 💾 GENERATE & SAVE STEGO IMAGES
reset(coverDS); reset(secretDS);
for i = 1:min(numel(coverDS.Files), numel(secretDS.Files))
    cover = read(coverDS);
    secret = read(secretDS);
    stego = predict(netEmbed, {'cover_input', cover, 'secret_input', secret});
    imwrite(stego, fullfile(outputFolder, sprintf('stego_%04d.png', i)));
end

%% 🔓 EXTRACTION NETWORK DEFINITION
inputStego = imageInputLayer(imageSize,'Name','stego_input');
extractLayers = [
    convolution2dLayer(3,16,'Padding','same','Name','conv1')
    reluLayer('Name','relu1')
    convolution2dLayer(3,32,'Padding','same','Name','conv2')
    reluLayer('Name','relu2')
    convolution2dLayer(3,3,'Padding','same','Name','reconstructed')
    regressionLayer('Name','reg_output')];

lgraphExtract = layerGraph([inputStego; extractLayers]);

%% 🏋 TRAIN EXTRACTION NETWORK
stegoDS = imageDatastore(outputFolder);
stegoDS.ReadFcn = @(filename) imresize(im2double(imread(filename)), imageSize(1:2));
reset(secretDS);

extractTrainDS = combine(stegoDS, secretDS);
trainExtractDS = transform(extractTrainDS, @(data) struct( ...
    'stego_input', data{1}, ...
    'reg_output', data{2}));

netExtract = trainNetwork(trainExtractDS, lgraphExtract, options);

%% 📊 EVALUATE USING PSNR AND SSIM
reset(stegoDS); reset(secretDS);
total_psnr = 0; total_ssim = 0;
N = numel(stegoDS.Files);

for i = 1:N
    stego = im2double(imread(stegoDS.Files{i}));
    secret = read(secretDS);
    extracted = predict(netExtract, stego);

    total_psnr = total_psnr + psnr(extracted, secret);
    total_ssim = total_ssim + ssim(extracted, secret);
end

avg_psnr = total_psnr / N;
avg_ssim = total_ssim / N;

fprintf('\n✅ Average PSNR: %.2f dB\n', avg_psnr);
fprintf('✅ Average SSIM: %.4f\n', avg_ssim);