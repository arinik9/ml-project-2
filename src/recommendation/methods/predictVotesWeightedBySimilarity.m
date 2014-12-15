function prediction = predictVotesWeightedBySimilarity(user, artist, Y, participants, userDV, S)
% prediction = deviation to participant's average + this user's average
%              * normalized similarity measure as a way to indicate trust

    votes = full(Y(participants, artist) - userDV(participants, 1));
    % Prediction is centered around this user's mean count
    prediction = userDV(user, 1);

    % Weight vote of each user by its similarity
    similarities = S(participants, user);
    if(sum(abs(similarities)) > eps)
        % Normalize the similarity to use them as weights
        similarities = similarities ./ sum(abs(similarities));

        prediction = prediction + sum(similarities .* votes);
    end;
end
