function nnPred = predictNeuralNetwork(nn, xPred)
% Predict output labels from the given Neural Network model for the xPred
% data examples

    % to get the scores we need to do nnff (feed-forward) 
    % which returns an neural network structure with updated 
    % layer activations, error and loss (nn.a, nn.e and nn.L)
    % See for example nnpredict().
    % (This is a weird thing of this toolbox)
    nn.testing = 1;
    nn = nnff(nn, xPred, zeros(size(xPred,1), nn.size(end)));
    nn.testing = 0;

    % predict on the test set
    nnPred = nn.a{end};

    % we want a single score, subtract the output sigmoids
    nnPred = nnPred(:,1) - nnPred(:,2);
    
    nnPred = (nnPred + 1) / 2;

end