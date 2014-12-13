function predictor = learnConstantPredictor(Y, Ytest, userDV, artistDV)
% LEARNCONSTANTPREDICTOR This predictor always returns the overall mean

    overallMean = mean(nonzeros(Y));
    predictor = @(user, artist) overallMean;

end
