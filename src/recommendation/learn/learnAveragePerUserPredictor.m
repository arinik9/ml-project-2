function predictor = learnAveragePerUserPredictor(~, ~, userDV, ~)
% LEARNAVERAGEPERUSERPREDICTOR This predictor returns the mean of the user

    predictor = @(user, artist) userDV(user, 1);

end
