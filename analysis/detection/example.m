clearvars;

% -- GETTING STARTED WITH THE PERSON DETECTION DATASET -- %
% IMPORTANT: Make sure you downloaded Piotr's toolbox: http://vision.ucsd.edu/~pdollar/toolbox/doc/
%            and add it to the path with
%            addpath(genpath('where/the/toolbox/is'))
%
%    And make sure you downloaded the train_feats.mat and train_imgs.mat
%    files we provided you with.

% add path to piotr's toolbox

% ---------------------------------------------
% ====>   FIX TO YOUR ACTUAL TOOLBOX PATH <====
% ---------------------------------------------
addpath(genpath('../toolbox/'));

% Load both features and training images
load train_feats;
load train_imgs;

%% --browse through the images, and show the feature visualization beside
%  -- You should explore the features for the positive and negative
%  examples and understand how they resemble the original image.
for i=1:10
    clf();
    
    subplot(121);
    imshow(imgs{i}); % image itself
    
    subplot(122);
    im( hogDraw(feats{i}) ); colormap gray;
    axis off; colorbar off;
    
    pause;  % wait for keydo that then, 
end

%% -- Generate feature vectors (so each one is a row of X)
fprintf('Generating feature vectors..\n');
D = numel(feats{1});  % feature dimensionality
X = zeros([length(imgs) D]);

for i=1:length(imgs)
    X(i,:) = feats{i}(:);  % convert to a vector of D dimensions
end

%% -- Example: split half and half into train/test
fprintf('Splitting into train/test..\n');
% NOTE: you should do this randomly! and k-fold!
Tr.idxs = 1:2:size(X,1);
Tr.X = X(Tr.idxs,:);
Tr.y = labels(Tr.idxs);

Te.idxs = 2:2:size(X,1);
Te.X = X(Te.idxs,:);
Te.y = labels(Te.idxs);

%% Train simple neural network with matlab's toolbox
fprintf('Training simple neural network..\n');
rng(8339);  % fix seed, this    NN is very sensitive to initialization
net = patternnet([3 3]);

% train the neural network on the training set
net = train(net,Tr.X', 1.0*[(Tr.y' > 0); (Tr.y' < 0)]);

% predict on the test set
nnPred = net(Te.X');

% just keep the one for the positive class
nnPred = nnPred(1,:)';


%% See prediction performance
fprintf('Plotting performance..\n');
% let's also see how random predicition does
randPred = rand(size(Te.y));

% and plot all together, and get the performance of each
methodNames = {'Neural Network', 'Random'}; % this is to show it in the legend
avgTPRList = evaluateMultipleMethods( Te.y > 0, [nnPred,randPred], true, methodNames );

% now you can see that the performance of each method
% is in avgTPRList. You can see that random is doing very bad.
avgTPRList

%% visualize samples and their predictions (test set)
figure;
for i=1:10
    clf();
    
    subplot(121);
    imshow(imgs{Te.idxs(i)}); % image itself
    
    subplot(122);
    im( hogDraw(feats{Te.idxs(i)}) ); colormap gray;
    axis off; colorbar off;
    
    % show if it is classified as pos or neg, and true label
    title(sprintf('Label: %d, Pred: %d', labels(Te.idxs(i)), 2*(nnPred(i)>0.5) - 1));
    
    pause;  % wait for keydo that then, 
end
