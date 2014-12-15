function predictor = learnArtistBasedPredictor(Y, ~, userDV, artistDV)
% LEARNARTISTBASEDPREDICTOR Very simple: user's mean + artist likeability

    predictor = @(user, artist) predict(user, artist, userDV, artistDV);
end

function prediction = predict(user, artist, userDV, artistDV)
    expectedMean = median(userDV(:, 1));
    
    prediction = expectedMean;
    if(artistDV(artist, 2) > 5)
        prediction = prediction + artistDV(artist, 3);
    end;
end
