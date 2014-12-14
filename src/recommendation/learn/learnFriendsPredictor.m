function predictor = learnFriendsPredictor(Y, Gstrong, userDV, artistDV)
% LEARNFRIENDSPREDICTOR Predict counts based on friends' votes
% If there are no friends who listened to the target artist, fallback on
% the artist's mean.
% A "vote" is computed from the deviation to the user's mean.
    predictor = @(user, artist) predictFromFriends(user, artist, Y, Gstrong, userDV, artistDV);
end

function prediction = predictFromFriends(user, artist, Y, Gstrong, userDV, artistDV)
    nStrong = size(Gstrong, 1);

    % Hypothetic volume of this unknown user
    volumeEstimate = median(userDV(:, 1));
    % Safe fallback
    prediction = volumeEstimate + artistDV(artist, 3);

    % We ignore the unknown to unknown friendships
    friends = find(Gstrong(user, 1:(end-nStrong)));

    if(~isempty(friends))
        Ysub = Y(friends, artist);
        if(nnz(Ysub) > 1)
            [participants, ~] = find(Ysub);
            % TODO: only take positive information? (ignore the user's mean?)
            votes = Y(participants, artist); % - userDV(participants, 1)
            prediction = mean(votes);
        end;
    end;
end


