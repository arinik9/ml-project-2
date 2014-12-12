%% Searching (and verifying) the appropriate way to generate predictions
clearvars;
loadDataset;

%% Any kind of processing
[Ynorm, newMean] = normalizedSparse(Ytrain);

%% Output predictions
% For each user
trValues = zeros(sz.tr.nnz, 1);
teValues = zeros(sz.te.nnz, 1);
for i = 1:sz.u
    ii = (idx.tr.u == i);
    if(nnz(ii) > 0)
        jj = idx.tr.a(ii);
        trValues(ii) = Ynorm(i, jj) + 0.05;
    end
    
    ii = (idx.te.u == i);
    if(nnz(ii) > 0)
        jj = idx.te.a(ii);
        teValues(ii) = newMean * rand(1, length(jj));
    end
end

trYhatFamous = sparse(idx.tr.u, idx.tr.a, trValues, sz.u, sz.a);
teYhatFamous = sparse(idx.te.u, idx.te.a, teValues, sz.u, sz.a);

% Compute error on denormalized data
e.tr.famous = computeRmse(Ytrain, denormalize(trYhatFamous, idx));
e.te.famous = computeRmse(Ytest, denormalize(teYhatFamous, testIdx));

fprintf('RMSE on popular artists only : %f | %f\n', e.tr.famous, e.te.famous);
diagnoseError(Ytrain, denormalize(trYhatFamous, idx));
diagnoseError(Ytest, denormalize(teYhatFamous, testIdx));


