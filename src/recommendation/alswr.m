function [U, M] = alswr(R, Rtest, k, lambda, plotLearningCurve)
% ALSWR Alternating Least Squares with Weighted lambda-regulaziation
% Low-rank matrix factorization using the technique described by:
%   Zhou, Y., Wilkinson, D., Schreiber, R., & Pan, R. (2008).
%   Large-scale parallel collaborative filtering for the netflix prize.
%
% Warning: with very small values of k (e.g. 3),
% the algorithm may not converge.
%
% INPUT:
%   R: The sparse features matrix (N x D) to be factorized (e.g. ratings
%      of D movies by N users).
%   Rtest: Test set on which to test the reconstruction error. Iterations
%           will stop when train error starts going up ("early stopping").
%   k: Target (reduced) dimensionality
%   lambda: Regulaziation parameter
%   plotLearningCurve: Boolean flag to plot the learning curve
% OUTPUT:
%   U: (k x N) The dimensionality-reduced matrix representing the
%      individuals (e.g. users).
%   M: (k x D) The dimensionality-reduced matrix representing the items
%      (e.g. movies).
% An approximation of the initial R can then be reconstructed using:
%   Rapprox = U' * M

    if(~exist('lambda', 'var'))
        lambda = 0;
    end;
    if(~exist('plotLearningCurve', 'var'))
        plotLearningCurve = 0;
    end;
    
    [N, D] = size(R);
    % Indices of nonzeros elements
    [rIdx, cIdx] = find(R);

    % Initialization: average over the features or small random numbers
    U = zeros(k, N);
    M = rand(k, D);
    for j = 1:D
        if(nnz(R(:, j)) > 0)
            M(1, j) = mean(nonzeros(R(:, j)));
        else
            M(1, j) = 0;
        end;
    end;
    
    % Until convergence, make ALS steps
    % Convergence criterion: the U and M matrices stop changing much
    maxIterations = 10;
    it = 0;
    
    % TODO: plot learning curves (train and test reconstruction error vs
    % iterations). Should look like ridge regression (because after all,
    % this is just a regularized learning process).
    % TODO: convergence criterion: stop when test error starts going up
    % epsilon = 1e-2;
    % movement = -1;
    trErrors = [];
    teErrors = [];
    while (it < maxIterations) % (abs(movement) > epsilon)
        it = it + 1;
        
        % Fix M, solve for U
        % For each row of R (e.g. users)
        for i = 1:N
            if(nnz(R(i, :)) > 0)
                % Columns for which we have data from this user
                columns = cIdx(rIdx == i);
                subM = M(:, columns);
                % Number of data points available for this user
                nObserved = nnz(R(i, :));

                A = (subM * subM') + lambda * nObserved * eye(k);
                V = subM * R(i, columns)';

                % Least-squares solve
                U(:, i) = A \ V;
            else
                U(:, i) = zeros(k, 1);
            end;
        end;
        
        % Fix U, solve for M
        % For each column of R (e.g. movies)
        for j = 1:D
            if(nnz(R(:, j)) > 0)
                % Users for which we have data about this column
                rows = rIdx(cIdx == j);
                subU = U(:, rows);
                % Number of data points available for this user
                nObserved = nnz(R(:, j));

                A = (subU * subU') + lambda * nObserved * eye(k);
                V = subU * R(rows, j);

                % Least-squares solve
                M(:, j) = A \ V;
            else
                M(:, j) = zeros(k, 1);
            end;
        end;
        
        % Estimate the quality of our approximation
        Rapprox = U' * M;
        trError = computeRmse(R, Rapprox);
        trErrors = [trErrors; trError];
        teError = computeRmse(Rtest, Rapprox);
        teErrors = [teErrors; teError];
        
        if(plotLearningCurve)
            fprintf('Iteration %d: reconstruction RMSE %f | %f\n', it, trError, teError);
        end;
    end;
    
    if(plotLearningCurve)
        plot(1:it, trErrors, 'b.-');
        hold on;
        plot(1:it, teErrors, 'r.-');
        legend('Reconstruction test error', 'Reconstruction train error');
    end;
end
