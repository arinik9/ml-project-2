function errors = diagnoseError(y, yHat)
% DIAGNOSEERROR Look at the error made VS the quantity of available data
% INPUT:
%   y: Real results (each data example can have several outputs)
%   yHat: Predicted result
% OUTPUT:
%   errors: [number of available listening count, average RMSE error made] (sorted)

    [averaged, errors] = computeErrorByCount(y, yHat);
    nCounts = unique(errors(:, 1));

    % Baseline
    nullMatrix = sparse(size(y, 1), size(y, 2));
    [averagedNull, ~] = computeErrorByCount(y, nullMatrix);

    rmse = computeRmse(y, yHat);
    baselineRmse = computeRmse(y, nullMatrix);
    %fprintf('RMSE: %f (compared to constant predictor: %f)\n', rmse, baselineRmse);

    figure;
    % Plot reparition of log(RMSE)
    subplot(1, 2, 1);
    hist(errors(:, 2), 20);
    title('Repartition of RMSE');

    % Plot error made VS quantity of data available
    subplot(1, 2, 2);
    hold on;
    %errorbar(1:n, averaged(:, 1), averaged(:, 2));
    semilogx(nCounts, averaged(:, 1), 'b.');
    semilogx(nCounts, averagedNull(:, 1), 'r+');

    set(gca,'Xdir','reverse')
    title('Average error made over artists with a given number of observations');
    xlabel('Number of available listening counts');
    ylabel('Average RMSE error');

    % Spot the largest errors
    %{
    largeRmse = 2;
    largeErrorIdx = (errors(:, 2) > largeRmse);
    maxAvailable = max(errors(largeErrorIdx, 1));
    medianAvailable = median(errors(largeErrorIdx, 1));
    fprintf('RMSE > %f occurs over artists having\nat most %d ratings available (median: %d rating available).\n', largeRmse, maxAvailable, medianAvailable);
    %}
end
