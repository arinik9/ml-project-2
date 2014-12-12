function [rfPred, labels] = predictRandomForest(rfModel, Xpred)
% Takes a random forest and predict outputs for given Xpred values
% Returns probabilities and associated labels computed by the model

    [labels, scores] = rfModel.predict(Xpred);
    
    % Labels is a char. We want it to be a number.
    labels = str2double(labels);
    
    % We take as our score the probability of predicting label 1
    rfPred = scores(:,2);
    
end