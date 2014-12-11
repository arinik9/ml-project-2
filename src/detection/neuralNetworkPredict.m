function nnPred = neuralNetworkPredict(yTr, XTr, XTe, plot_flag, learningRate, activationFunction, dropoutFraction, weightPenaltyL2, nnArchitecture, numepochs, batchsize)
% Binary classification using Neural Network provided by the deepLearning toolbox
% This function takes train and test data and apply NN model with given
% parameters in order to test different parameters tuning instead of boilerplate code.
%
% Outputs:
%   - nnPred: predictions obtained from the NN
% Inputs:
%   - yTr: training output data
%   - XTr: normalized training input data
%   - XTe: normalized test input data
%   - learningRate: Note from toolbox: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
%   - activationFunction: Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
%   - dropoutFraction: Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
%   - weightPenaltyL2: L2 regularization
%   - plot_flag: if == 1 => plots trainin error as the NN is trained 
%   - nnSetup: setup for Neural Network. The first layer needs to have
%   number of features neurons, and the last layer the number of classes
%   (here two). One can add as many layer as he wants by indicating the
%   number of activation functions in that layer.
%   - numepochs: Number of full sweeps through data
%   - batchsize: Take a mean gradient step over this many samples

    % Default parameter settings
    
    if (nargin < 4)
        plot_flag = 0;
    end
    
    if (nargin < 5)
        learningRate = 2;
    end

    if (nargin < 6)
        activationFunction = 'tanh_opt';
    end

    if (nargin < 7)
        dropoutFraction = 0;
    end
    
    if (nargin < 8)
        weightPenaltyL2 = 0;
    end
    
    if (nargin < 9)
       % setup NN with 2 layers for binary classification
       nnArchitecture = [size(XTr,2) 10 2]; 
    end

    if (nargin < 10)
        numepochs = 50;
    end
    
    if (nargin < 11)
        batchsize = 100;
    end

    rng('default');
    rng(8339); % fix seed, this NN is very sensitive to initialization

    % setup NN. The first layer needs to have number of features neurons,
    %  and the last layer the number of classes (here two).
    nn = nnsetup(nnArchitecture);
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
    numSampToUse = opts.batchsize * floor( size(XTr) / opts.batchsize);
    XTr = XTr(1:numSampToUse,:);
    yTr = yTr(1:numSampToUse);

    % prepare labels for NN
    LL = [1*(yTr>0)  1*(yTr<0)];  % first column, p(y=1)
                                    % second column, p(y=-1)

    [nn, ~] = nntrain(nn, XTr, LL, opts);

    % to get the scores we need to do nnff (feed-forward) 
    % which returns an neural network structure with updated 
    % layer activations, error and loss (nn.a, nn.e and nn.L)
    % See for example nnpredict().
    % (This is a weird thing of this toolbox)
    nn.testing = 1;
    nn = nnff(nn, XTe, zeros(size(XTe,1), nn.size(end)));
    nn.testing = 0;

    % predict on the test set
    nnPred = nn.a{end};

    nnPred(:,1)
    nnPred(:,2)
    % we want a single score, subtract the output sigmoids
    nnPred = nnPred(:,1) - nnPred(:,2);
    
    nnPred = (nnPred + 1) / 2;
    
end