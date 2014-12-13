function saveDetectionPredictions(probabilities)
    Ytest_score = probabilities;
    save('./results/personPred.mat', 'Ytest_score');
end
