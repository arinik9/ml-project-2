function setSeed(seed)
% INPUT
%   seed: Positive integer to obtain reproducible random numbers generation
%         Or -1 to switch back to unpredictable results
%
% SEE ALSO
%   http://fr.mathworks.com/help/matlab/math/updating-your-random-number-generator-syntax.html
    if(seed > 0)
        rng(seed);
    else
        rng('shuffle');
    end;
end