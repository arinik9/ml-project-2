function [coeff, mu, latent] = pcaCompute(X)
% Wrapper function for Piotr's PCA to handle matrice transposition
% Piotr's pca() needs to be given a DxN matrix with N # data example
    [coeff, mu, latent] = pca(X');
end
