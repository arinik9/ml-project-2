function predictor = learnSimilarityBasedPredictor(Ytrain, Ytest, userDV, artistDV)
% LEARNSIMILARITYBASEDPREDICTOR Predict based on inter-user similarity measures

    S = computeSimilarityMatrix(Ytrain, Ytest, userDV);

    predictor = @(user, artist) predict(user, artist, Y, userDV, S);
end

function prediction = predict(user, artist, Y, userDV, S)
    % Participants are all users having listened to this artist,
    % regardless of their proximity. Their vote will be weighted by the
    % similarity afterwards.
    participants = find(Y(:, artist));
    predictVotesWeightedBySimilarity(user, artist, Y, participants, userDV, S)
end
