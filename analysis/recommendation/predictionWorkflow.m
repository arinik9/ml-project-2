%% Searching (and verifying) the appropriate way to generate predictions
clearvars;
loadDataset;

%% Any kind of processing
normalizedMean = mean(nonzeros(Ytrain));

%% Output predictions
% For each user
trValues = zeros(sz.tr.nnz, 1);
teValues = zeros(sz.te.nnz, 1);
for i = 1:sz.u
    ii = (idx.tr.u == i);
    if(nnz(ii) > 0)
        jj = idx.tr.a(ii);
        trValues(ii) = Ytrain(i, jj) + 0.05;
    end;

    ii = (idx.te.u == i);
    if(nnz(ii) > 0)
        jj = idx.te.a(ii);
        teValues(ii) = normalizedMean * rand(1, length(jj));
    end;
end;

trYhatFamous = sparse(idx.tr.u, idx.tr.a, trValues, sz.u, sz.a);
teYhatFamous = sparse(idx.te.u, idx.te.a, teValues, sz.u, sz.a);

% Compute error on normalized data
e.tr.famous = computeRmse(Ytrain, trYhatFamous);
e.te.famous = computeRmse(Ytest, teYhatFamous);

fprintf('RMSE on popular artists only : %f | %f\n', e.tr.famous, e.te.famous);
diagnoseError(Ytrain, trYhatFamous);
diagnoseError(Ytest, teYhatFamous);


