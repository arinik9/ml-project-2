function lSigmoid = logSigmoid(A)
% Compute the log of the sigmoid function
% Useful to avoid approaching machine number representation boundaries
% Computational trick:
% sigma(X) = exp(X) / (1 + exp(X))
%          = 1 / (1 + exp(-X))
% If X is large and positive, it's useful to use:
% log(sigma(X)) = - log(1 + exp(-X))
% If X is large and negative, it's useful to use:
% log(sigma(X)) = X - log(1 + exp(X))
% And THEN take the exp of that when you need to use sigma(X).
	
	aPlus = A(A > 0);
	aMinus = A(A <= 0);
	% Log of the sigmoid function
	
	lSigmoid = zeros(size(A));
    
	lSigmoid(A > 0) = - log(1 + exp(-aPlus));
	lSigmoid(A <= 0) = aMinus - log(1 + exp(aMinus));
end