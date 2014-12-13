function saveRecommendationPredictions(Yweak, Ystrong)
    Ytest_strong_pred = Yweak;
    Ytest_weak_pred = Ystrong;
    save('./results/songPred.mat', 'Ytest_strong_pred', 'Ytest_weak_pred');
end
