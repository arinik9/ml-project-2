function yhat = randomPrediction(Nyhat, threshold)
% Produce random predictions labels
    randPred = rand(Nyhat);
    yhat = outputLabels(randPred, threshold);

end