function [tprAtWP,auc,fprAvg,tprAvg] = kCVfastROC(allLabels, allScores, plot_flag, plotStyle)

    if ~exist('plot_flag','var')
        plot_flag = 0;
    end

    if ~exist('plotStyle','var')
        plotStyle = 'b';
    end
    
    % Compute the fpr and tpr means to draw average curve
    n = size(allLabels,2);
    nPoints = size(allLabels,1);
    tprAtWP = zeros(n,1);
    auc = zeros(n,1);
    fpr = zeros(nPoints, n);
    tpr = zeros(nPoints, n);
    for i=1:n
        [tprAtWP(i),auc(i),fpr(:,i),tpr(:,i)] = fastROC(allLabels(:,i) > 0, allScores(:,i), 0);
    end
    fprAvg = mean(fpr,2);
    tprAvg = mean(tpr,2);

    %Plot the ROC curve
    if plot_flag==1
        figure();
        semilogx(fprAvg,tprAvg,plotStyle,'LineWidth',2); hold on;
        %plot(fprAvg,tprAvg,plotStyle,'LineWidth',2);
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
    end

end