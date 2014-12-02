function yhat = randomPrediction(Nyhat, threshold)

    randPred = rand(Nyhat);
    yhat = outputLabels(randPred, threshold);

end