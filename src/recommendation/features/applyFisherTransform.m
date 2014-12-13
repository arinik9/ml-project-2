function Sfisher = applyFisherTransform(S)
% APPLYFISHERTRANSFORM
%
% INPUT
%   S: (users x users) Similarity matrix (can be sparse)
% OUTPUT
%   Sfisher
% SEE ALSO
%   http://en.wikipedia.org/wiki/Fisher_transformation

    if(issparse(S))
        [ii, jj] = find(S);
        [n, ~] = size(S);
        values = nonzeros(S);
        Sfisher = sparse(ii, jj, atanh(values), n, n);
    else
        Sfisher = atanh(S);
    end;
end
