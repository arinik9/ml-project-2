function [U, M] = alswr(R, Rtest, k, lambda, plotLearningCurve)
% ALSWR Alternating Least Squares with Weighted lambda-regulaziation
% Low-rank matrix factorization using the technique described by:
%   Zhou, Y., Wilkinson, D., Schreiber, R., & Pan, R. (2008).
%   Large-scale parallel collaborative filtering for the netflix prize.
% 
% Input matrices are expected to be normalized.
% Warning: for very small values of k (e.g. 3),
% the algorithm may not converge.
%
% INPUT:
%   R: The sparse features matrix (N x D) to be factorized (e.g. ratings
%      of D movies by N users). Should be normalized.
%   Rtest: Test set on which to test the reconstruction error. Iterations
%          will stop when train error starts going up ("early stopping").
%          Should *not* be normalized.
%   k: Target (reduced) dimensionality
%   lambda: Regulaziation parameter
%   plotLearningCurve: Boolean flag to plot the learning curve
% OUTPUT:
%   U: (k x N) The dimensionality-reduced matrix representing the
%      individuals (e.g. users).
%   M: (k x D) The dimensionality-reduced matrix representing the items
%      (e.g. movies).
% An approximation of the initial R can then be reconstructed using:
%   Rapprox = denormalize(U' * M)

    if(~exist('lambda', 'var'))
        lambda = 0;
    end;
    if(~exist('plotLearningCurve', 'var'))
        plotLearningCurve = 0;
    end
    
    % TODO: smaller epsilon
    epsilon = 1;
    
    % TODO: use getRelevantIndices
    Roriginal = R;
    R = normalizedSparse(R);
    [idx, sz] = getRelevantIndices(Roriginal);
    [testIdx, testSz] = getRelevantIndices(Rtest);
    
    % Initialization: average over the features or small random numbers
    U = zeros(k, sz.u);
    M = rand(k, sz.a);
    for j = 1:sz.a
        jj = (idx.a == j);
        if(nnz(jj) > 0)
            M(1, j) = mean(R(idx.u(jj), j));
        else
            M(1, j) = 0;
        end;
    end;
    
    if(plotLearningCurve)
        fprintf('Starting ALS-WR with k = %d...\n', k);
    end;
    
    % Until convergence, make ALS steps
    % Convergence criterion: test RMSE reduction becomes insignificant
    % or even negative (i.e. we start overfitting)
    trErrors = [];
    teErrors = [];

    maxIterations = 100;
    it = 0;
    while true
        it = it + 1;
        
        % Fix M, solve for U
        % For each row of R (e.g. users)
        for i = 1:sz.u
            ii = (idx.u == i);
            if(nnz(ii) > 0)
                % Columns for which we have data from this user
                columns = idx.a(ii);
                subM = M(:, columns);
                % Number of data points available for this user
                nObserved = nnz(Roriginal(i, columns));

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
        for j = 1:sz.a
            jj = (idx.a == j);
            if(nnz(jj) > 0)
                % Users for which we have data about this column
                rows = idx.u(jj);
                subU = U(:, rows);
                % Number of data points available for this user
                nObserved = nnz(Roriginal(rows, j));

                A = (subU * subU') + lambda * nObserved * eye(k);
                V = subU * R(rows, j);

                % Least-squares solve
                M(:, j) = A \ V;
            else
                M(:, j) = zeros(k, 1);
            end;
        end;
        
        % Estimate the quality of our approximation
        trError = computeRmse(Roriginal, denormalize(reconstructFromLowRank(U, M, idx, sz), idx));
        teError = computeRmse(Rtest, denormalize(reconstructFromLowRank(U, M, testIdx, testSz), testIdx));
        
        if(plotLearningCurve)
            trErrors = [trErrors; trError];
            teErrors = [teErrors; teError];
            fprintf('Iteration %d: reconstruction RMSE %f | %f\n', it, trError, teError);
        end;
        
        % Stopping criterion
        if (it > 5 && previousError - teError < epsilon) || (it > maxIterations)
            break;
        end;
        previousError = teError;
    end;
    
    if(plotLearningCurve)
        fprintf('Done!\n');
        plot(1:it, trErrors, 'b.-');
        hold on;
        plot(1:it, teErrors, 'r.-');
        legend('Reconstruction train error', 'Reconstruction test error');
    end;
end
