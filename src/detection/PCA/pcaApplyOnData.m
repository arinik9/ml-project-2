function [pcaX, pcaXhat, pcaAvsq] = pcaApplyOnData(X, coeff, mu, kPC)
% Wrapper for Piotr's pcaApply function to handle matrix transpositions
% Results have to be normalized
    [pcaXRes, pcaXhatRes, pcaAvsqRes] = pcaApply(X', coeff, mu, kPC);
    pcaX = pcaXRes'; pcaXhat = pcaXhatRes'; pcaAvsq = pcaAvsqRes;

end