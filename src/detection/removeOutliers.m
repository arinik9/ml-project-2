function [Xnew, ynew] = removeOutliers(X,y, nDeviations)

    mediansX = ones(size(X,1), 1) * median(X);
    deviationsX = ones(size(X,1), 1) * std(X);
    % Find indices of the outliers
    outlier = abs(X - mediansX) > nDeviations * deviationsX;
    t = sum(outlier,2) > 0;
    Xnew = X(~t,:);
    ynew = y(~t);

end