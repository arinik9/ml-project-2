function [errors, teErrors] = diagnoseError(y, yHat, yTest, yTestHat)
% DIAGNOSEERROR Look at the error made VS the quantity of available data
% INPUT:
%   y: Real results (each data example can have several outputs)
%   yHat: Predicted result
%   [yTest]
%   [yTestHat]
% OUTPUT:
%   errors: [number of available listening count, average RMSE error made] (sorted)

    % Baseline
    baseline = sparse(size(y, 1), size(y, 2));
    [averagedBaseline, ~] = computeErrorByCount(y, baseline);

    % Training set
    nRows = 1;
    name = ' ';
    [averaged, errors] = computeErrorByCount(y, yHat);

    % Test set
    if(exist('yTest', 'var'))
        nRows = 2;
        name = ' train ';
        [teAveraged, teErrors] = computeErrorByCount(yTest, yTestHat, y);
        [teAveragedBaseline, ~] = computeErrorByCount(yTest, baseline, y);
    end;

    figure;
    makeSubplots(name, errors, averaged, averagedBaseline, nRows, 0);

    if(exist('yTest', 'var'))
        makeSubplots(' test ', teErrors, teAveraged, teAveragedBaseline, nRows, 2);
    end;
end

function makeSubplots(name, errors, averaged, averagedBaseline, nRows, offset)
    nCounts = unique(errors(:, 1));

    % Plot repartition of log(RMSE)
    subplot(nRows, 2, offset + 1);
    hist(errors(:, 2), 20);
    title(['Repartition of', name, 'RMSE']);

    % Plot error made VS quantity of data available
    subplot(nRows, 2, offset + 2);
    hold on;
    %errorbar(1:n, averaged(:, 1), averaged(:, 2));
    semilogx(nCounts, averaged(:, 1), 'b.');
    semilogx(nCounts, averagedBaseline(:, 1), 'r+');

    set(gca,'Xdir','reverse')
    xlabel('Number of available listening counts');
    ylabel(['Average', name, 'RMSE error']);
end
