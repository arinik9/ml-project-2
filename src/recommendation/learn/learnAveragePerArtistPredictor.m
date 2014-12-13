function predictor = learnAveragePerArtistPredictor(Y, Ytest, userDV, artistDV)
% LEARNAVERAGEPERARTISTPREDICTOR This predictor returns the mean of the artist

    predictor = @(user, artist) artistDV(artist, 1);

end
