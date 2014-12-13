function predictor = learnEachArtistPredictor(Y, Ytest, userDV, artistDV)
% LEARNEACHARTISTPREDICTOR This predictor trains a single linear regression model per artist
%
% It can be seen as a particular case of the Head / Tail method
% with a threshold set to 1 (i.e. we allow every artist to be part of the head)
%
% SEE ALSO
%   Park, Yoon-Joo, and Alexander Tuzhilin.
%   "The long tail of recommender systems and how to leverage it."

    predictor = learnHeadTailPredictor(Y, Ytest, userDV, artistDV, 1);
end
