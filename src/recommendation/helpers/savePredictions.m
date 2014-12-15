function saveRecommendationPredictions(Yweak, Ystrong)
    Ytest_strong_pred = Yweak;
    Ytest_weak_pred = Ystrong;

    output = './results/songPred.mat';
    save(output, 'Ytest_strong_pred', 'Ytest_weak_pred');
    fprintf('Predictions saved to %s.\n', output);
end
