function [U, M] = alswr(R, k, lambda)
% ALSWR Alternating Least Squares with Weighted lambda-regulaziation
% Low-rank matrix factorization using the technique described by:
%   Zhou, Y., Wilkinson, D., Schreiber, R., & Pan, R. (2008).
%   Large-scale parallel collaborative filtering for the netflix prize.
%
% INPUT:
%   R: The sparse features matrix (N x D) to be factorized (e.g. ratings
%      of D movies by N users).
%   k: Target (reduced) dimensionality
%   lambda: Regulaziation parameter
% OUTPUT:
%   

    if(~exist('lambda', 'var'))
        lambda = 0;
    end;
    
    [N, D] = size(R);
    % Indices of nonzeros elements
    [rIdx, cIdx] = find(R);
    rowsIdx = unique(rIdx);
    columnsIdx = unique(cIdx);

    % Notation:
    %   U: (k x N)
    %   M: (k x D)
    % An approximation of the initial R can then be reconstructed using:
    %   Rapprox = U' * M

    % Initialization: average over the features or small random numbers
    M = rand(k, D);
    U = zeros(k, N);
    for i = 1:k
        if(nnz(R(:, i)) > 0)
            M(1, i) = mean(nonzeros(R(:, i)));
        end;
    end;
    
    % Until convergence, make ALS steps
    % Convergence criterion: the U and M matrices stop changing much
    maxIterations = 5;
    it = 0;
    
    % TODO: plot learning curves (train and test reconstruction error vs
    % iterations). Should look like ridge regression (because after all,
    % this is just a regularized learning process).
    % TODO: convergence criterion: stop when test error starts going up
    % epsilon = 1e-2;
    % movement = -1;
    
    while (it < maxIterations) % (abs(movement) > epsilon)
        it = it + 1;
        
        % Fix M, solve for U
        % (For each user)
        for i = 1:length(rowsIdx)
            row = rowsIdx(i);
            % Columns for which we have data from this user
            columns = cIdx(rIdx == row);
            subM = M(:, columns);
            % Number of data points available for this user
            nObserved = nnz(R(row, :));
            
            A = (subM * subM') + lambda * nObserved * eye(k);
            V = subM * R(row, columns)';
            
            % Least-squares solve
            U(:, i) = A \ V;
        end;
        
        % Fix U, solve for M
        % (For each movie)
        for j = 1:length(columnsIdx)
            column = columnsIdx(j);
            % Users for which we have data about this column
            rows = rIdx(cIdx == column);
            subU = U(:, rows);
            % Number of data points available for this user
            nObserved = nnz(R(:, column));
            
            A = (subU * subU') + lambda * nObserved * eye(k);
            V = subU * R(rows, column);
            
            % Least-squares solve
            M(:, j) = A \ V;
        end;
        
        % Estimate the quality of our approximation
        Rapprox = U' * M;
        error = computeRmse(R, Rapprox);
        
        fprintf('At iteration %d, can reconstruct R with error %f\n', it, error);
    end;
end