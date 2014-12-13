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
    G = sparse(size(Y, 1), size(Y, 1));

    headPredictor = learnHeadPredictor(Y, G, Ytest, userDV, artistDV, headThreshold);
    tailPredictor = learnTailPredictor(Y, G, Ytest, userDV, artistDV);
    predictor = @(user, artist) predict(user, artist, artistDV(:, 2), headThreshold, headPredictor, tailPredictor);
end

function prediction = predict(user, artist, popularity, headThreshold, head, tail)
% At prediction time, choose either head or tail predictor
    if(popularity(artist) > headThreshold)
        prediction = head(user, artist);
    else
        prediction = tail(user, artist);
    end;
end

function predictor = learnHeadPredictor(Y, G, Ytest, userDV, artistDV, headThreshold)
    % Each prediction uses a model learnt from all users having listened to the artist
    % TODO: `modelEachArtist` shouldn't be aware of `headThreshold`
    betas = modelEachArtist(Y, G, headThreshold, userDV, artistDV);
    getFeatures = @(user, artist) generateFeatures(artist, user, G, userDV, artistDV);

    predictor = @(user, artist) getFeatures(user, artist) * betas(:, artist);
end

function predictor = learnTailPredictor(Y, G, Ytest, userDV, artistDV)
    % TODO: handle the tail properly!
    predictor = learnAveragePerArtistPredictor(Y, Ytest, userDV, artistDV);
end
