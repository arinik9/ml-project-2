function [tprAtWPAvg,aucAvg,fprAvg,tprAvg] = kCVfastROC(allLabels, allScores, plot_flag, info_flag, plot_title, plotStyle)

    if (~exist('plot_flag','var') && plot_flag ~= 0 && plot_flat ~= 1 )
        plot_flag = 0;
    end
    
    if (~exist('info_flag','var') && info_flag ~= 0 && info_flag ~= 1 )
        info_flag = 0;
    end
    
    if ~exist('plotStyle','var')
        plotStyle = 'b';
    end
    
    if ~exist('plot_title','var')
        plot_title = '';
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
    
    % Compute average area under curve and tprAtWP
    aucAvg = mean(auc,2);
    tprAtWPAvg = mean(tprAtWP,2);
    
    % Compute average false positive and true positive rate
    fprAvg = mean(fpr,2);
    tprAvg = mean(tpr,2);
    
    % Compute standard deviation
    uncertScore = std(tpr,0, 2);
    
    %Plot the ROC curve
    if plot_flag==1
        uncert = 2*uncertScore; % +/- 2 sigma => 95% confidence interval
        %figure();
        semilogx(fprAvg, tprAvg, plotStyle, 'LineWidth',2); hold on;
        jbfill(fprAvg, tprAvg + uncert, tprAvg - uncert, plotStyle, plotStyle, 1, 0.2);
        %plot(fprAvg,tprAvg,plotStyle,'LineWidth',2);
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        if info_flag == 1
          title( ...
                {plot_title, ...
                sprintf('Average ROC Curve with %d cross validation (95%% interval)', size(allLabels,2))}...
                );
            legendName = sprintf('avgTPR = %.3f', mean(tprAtWPAvg));
            legend( legendName, 'Location', 'NorthWest' );
        end
       
    end

end