function Sfisher = applyFisherTransform(S)
% APPLYFISHERTRANSFORM
%
% INPUT
%   S: (users x users) Similarity matrix (can be sparse)
% OUTPUT
%   Sfisher
% SEE ALSO
%   http://en.wikipedia.org/wiki/Fisher_transformation

    % We shrink the values of S slightly to avoid infinite values of atanh(S)
    epsilon = 1e-5;
    
    if(issparse(S))
        [ii, jj] = find(S);
        [n, d] = size(S);
        values = nonzeros(S);
        Sfisher = sparse(ii, jj, atanh(values * (1 - epsilon)), n, d);
    else
        Sfisher = atanh(S * (1 - epsilon));
    end;
end
