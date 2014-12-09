function nnPred = neuralNetworkPredict(Tr, Te, plot_flag, learningRate, activationFunction, dropoutFraction, weightPenaltyL2, numepochs, batchsize)
% Binary classification using Neural Network provided by the deepLearning toolbox
% This function takes train and test data and apply NN model with given
% parameters in order to test different parameters tuning instead of boilerplate code.
%
% Outputs:
%   - nnPred: predictions obtained from the NN
% Inputs:
%   - Tr: Tr.X training input data, Tr.y training output data
%   - Te: Te.X test input data, Te.y test output data
%   - learningRate: Note from toolbox: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
%   - activationFunction: Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
%   - dropoutFraction: Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
%   - weightPenaltyL2: L2 regularization
%   - plot_flag: if == 1 => plots trainin error as the NN is trained 
%   - numepochs: Number of full sweeps through data
%   - batchsize: Take a mean gradient step over this many samples

% TODO: for now check non-normalize data. Put normalization outside?

    % Default parameter settings
    
    if (nargin < 3)
        plot_flag = 0;
    end
    
    if (nargin < 4)
        learningRate = 2;
    end

    if (nargin < 5)
        activationFunction = 'tanh_opt';
    end

    if (nargin < 6)
        dropoutFraction = 0;
    end
    
    if (nargin < 7)
        weightPenaltyL2 = 0;
    end

    if (nargin < 8)
        numepochs = 50;
    end
    
    if (nargin < 9)
        batchsize = 100;
    end

    rng('default');
    rng(8339); % fix seed, this NN is very sensitive to initialization

    % setup NN. The first layer needs to have number of features neurons,
    %  and the last layer the number of classes (here two).
    nn = nnsetup([size(Tr.X,2) 2]);
    opts.numepochs =  numepochs;        %  Number of full sweeps through data
    opts.batchsize = batchsize;         %  Take a mean gradient step over this many samples

    % if == 1 => plots trainin error as the NN is trained    
    if (plot_flag == 1)
        opts.plot = 1;
    end

    % Parameter settings
    nn.activation_function = activationFunction;
    nn.learningRate = learningRate;
    nn.dropoutFraction = dropoutFraction;
    nn.weightPenaltyL2 = weightPenaltyL2;

    % this neural network implementation requires number of samples to be a
    % multiple of batchsize, so we remove some for this to be true.
    numSampToUse = opts.batchsize * floor( size(Tr.X) / opts.batchsize);
    Tr.X = Tr.X(1:numSampToUse,:);
    Tr.y = Tr.y(1:numSampToUse);

    % normalize data
    [Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std

    % prepare labels for NN
    LL = [1*(Tr.y>0)  1*(Tr.y<0)];  % first column, p(y=1)
                                    % second column, p(y=-1)

    [nn, L] = nntrain(nn, Tr.normX, LL, opts);

    Te.normX = normalize(Te.X, mu, sigma);  % normalize test data

    % to get the scores we need to do nnff (feed-forward) 
    % which returns an neural network structure with updated 
    % layer activations, error and loss (nn.a, nn.e and nn.L)
    % See for example nnpredict().
    % (This is a weird thing of this toolbox)
    nn.testing = 1;
    nn = nnff(nn, Te.normX, zeros(size(Te.normX,1), nn.size(end)));
    nn.testing = 0;

    % predict on the test set
    nnPred = nn.a{end};

    % we want a single score, subtract the output sigmoids
    nnPred = nnPred(:,1) - nnPred(:,2);
    
    % prediction scaled on 0 to 1 probabilities to use a [0:1] threshold
    % TODO: check if correct to do so
    nnPred = (nnPred + 1) / 2;
    
end