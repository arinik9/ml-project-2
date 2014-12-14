function [trErrors, teErrors] = diagnoseError(y, yHat, yTest, yTestHat)
% DIAGNOSEERROR Look at the error made VS the quantity of available data
% INPUT:
%   y: Real results (each data example can have several outputs)
%   yHat: Predicted result
%   [yTest]
%   [yTestHat]
% OUTPUT:
%   errors: [number of available listening count, average RMSE error made] (sorted)


    % Training set
    nRows = 1;
    name = ' ';
    trErrors = computeErrorMeasures(y, yHat);

    % Test set
    if(exist('yTest', 'var'))
        nRows = 2;
        teErrors = computeErrorMeasures(yTest, yTestHat, y);
    end;

    figure;
    makeSubplots(name,trErrors, nRows, 0);
    if(exist('yTest', 'var'))
        makeSubplots(' test ', teErrors, nRows, 3);
    end;
end

function errors = computeErrorMeasures(y, yHat, yRef)
    if(~exist('yRef', 'var'))
        yRef = y;
    end;

    errors = {};
    baseline = sparse(size(y, 1), size(y, 2));
    [errors.baseline, ~] = computeErrorByCount(y, baseline, yRef);

    [errors.averaged, errors.byCount] = computeErrorByCount(y, yHat, yRef);
    errors.byUser = computeErrorByUser(y, yHat);
end

function makeSubplots(name, errors, nRows, offset)
    counts = unique(errors.byCount(:, 1));

    % Plot error for each user
    subplot(nRows, 3, offset + 1);
    semilogy(1:size(errors.byUser, 1), errors.byUser, '.');
    xlim([1 size(errors.byUser, 1)+1]);
    xlabel('Users');
    ylabel('Euclidean norm of nonzero entries (log scale)');
    legend('True norm', 'Predicted norm');
    title([name, 'error over each user']);

    % Plot repartition of log(RMSE)
    subplot(nRows, 3, offset + 2);
    hist(errors.byCount(:, 2), 20);
    title(['Repartition of', name, 'RMSE']);

    % Plot error made VS quantity of data available
    subplot(nRows, 3, offset + 3);
    hold on;
    %errorbar(1:n, averaged(:, 1), averaged(:, 2));
    semilogx(counts, errors.averaged(:, 1), 'b.');
    semilogx(counts, errors.baseline(:, 1), 'r+');

    set(gca,'Xdir','reverse')
    xlabel('Number of available listening counts');
    ylabel(['Average', name, 'RMSE error']);
end
