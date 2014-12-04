function betaStar = penLogisticRegression(y, tX, alpha, lambda)
% Learn model parameters beta using Penalized Logistic Regression
% alpha: gradient descent step size
% lambda: regularization parameter
    if(nargin < 4)
        lambda = 0;
    end;

	% Stopping criterion
	maxIters = 100;

	% Initialization
	beta = zeros(size(tX, 2), 1);
    betaStar = beta;
    
	% Convergence criterion
	epsilon = 0.5;
	k = 0;
    err = -1; bestError = -1;
    errors = 0; % Can be used to plot error VS iteration number
	progress = 10;

	
	while ((k < maxIters) && (abs(progress) > epsilon))
		k = k + 1;
		
		% Newton's method step
        oldErr = err;
		[err, g, H] = penalizedLogisticRegressionLoss(y, tX, beta, lambda);
		
        descentDirection = H \ g;
		beta = beta - alpha .* descentDirection;
		
        errors(k) = err;
		progress = abs(err - oldErr);
        
		% Retain the best parameter fitted
		if (err < bestError || bestError == -1)
			betaStar = beta;
			bestError = err;
		end;
        
        % Status
		fprintf('%d| L = %.4f,  norm(beta) = %.4f\n', k, err, norm(beta, 2));
	end;

    
    %{
    figure;
    plot(1:k, errors, '.r');
    xlabel('Iteration');
    ylabel('Logistic regression error');
    %}
    
    %{
    if(k < maxIters)
        fprintf('Newton''s method converged after %d iterations.\n', k);
    else
        fprintf('Newton''s method stopped after %d iterations.\n', k);
    end;
    %}
end

