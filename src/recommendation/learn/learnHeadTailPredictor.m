function predictor = learnHeadTailPredictor(Y, Ytest, userDV, artistDV, headThreshold)
% LEARNHEADTAILPREDICTOR This predictor separates the dataset into "head" and "tail" artist
% Head contains artists for which we have many observations,
% while Tail contains artists which we do not know well.
% Both parts are handled separately.
%
% INPUT
%   headThreshold: Minimum number of observations required for an artist
%                  to be part of the head
%
% SEE ALSO
%   Park, Yoon-Joo, and Alexander Tuzhilin.
%   "The long tail of recommender systems and how to leverage it."


    % TODO: take the social network in input!
    Gtrain = sparse(size(Y, 1), size(Y, 1));

    % Each prediction uses a model learnt from all users having listened to the artist
    betas = learnEachArtist(Y, Gtrain, headThreshold, userDV, artistDV);
    getFeatures = @(user, artist) generateFeatures(artist, user, Gtrain, userDV, artistDV);

    % TODO: handle the tail! This takes care of the head only
    predictor = @(user, artist) getFeatures(user, artist) * betas(:, artist);
end
