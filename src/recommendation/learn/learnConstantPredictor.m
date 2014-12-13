function predictor = learnConstantPredictor(Y, Ytrain, userDV, artistDV)
% LEARNCONSTANTPREDICTOR This predictor always returns the overall mean

    overallMean = mean(nonzeros(Ytrain));
    predictor = @(user, artist) overallMean;

end
