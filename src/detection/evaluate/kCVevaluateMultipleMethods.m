function avgTprAtWP = kCVevaluateMultipleMethods( labels, predictions, ...
                                            showPlot, legendNames )                                   
% Automatically calls kCVfastROC() to show multiple curves, one for each
% prediction vector provided.
%
% INPUTS
%   labels          NxDxM vector, D number of folds, M being the number of
%                   predictions
%                   should be passed as cat(3, labelModel1, labelModel2, ...
%                   labelModelN)
%   predictions     NxDxM vector, D number of folds, M being the number of predictions to show
%                   should be passed as cat(3, predModel1, predModel2, ...
%                   predModelN)
%   if showPlot == true => single plot with multiple curves is shown.
%   legendNames is a cell list (optional) with the name to show for each
%   prediction in the legend.
%
% Returns avgTprAtWP where each element is the avgTprAtWP of each prediction
% vector given as input over its folds, plot average ROC curves for each method 
% and associated boxplot to visualize variance
                                        
    if nargin < 3
        showPlot = false;
    end

    if nargin < 4
        legendNames = [];
    end

    if size(labels,2) ~= size(predictions,2)
        error('labels and predictions must have same number of folds');
    end

    if size(labels,1) ~= size(predictions,1)
        error('labels and predictions must have same number of rows');
    end
    
    M = size(predictions,3);

    % list of plotting styles
    styles = ['r','b','k','m','g', 'y', 'c','r--', 'b--', 'k--','m--','g--'];

    if showPlot && (M > length(styles))
        error('Number of lines to show exceeds possible styles');
    end

    % tprAtWP for each fold
    tprAtWP = zeros(size(predictions,2),M);
    
    % Average tprAtWP over fold
    avgTprAtWP = zeros(M,1);

    if showPlot
        figure;
    end

    % Plot averaged ROC curves
    for i=1:M
        tprAtWP(:,i) = kCVfastROC( labels(:,:,i), predictions(:,:,i), 1, 0, 0, '', styles(i) );
        avgTprAtWP(i) = mean(tprAtWP(:,i));
        fprintf('avgTprAtWP: %d \n', avgTprAtWP(i));
        if showPlot
            hold on;
        end
    end
    
    if showPlot && ~isempty(legendNames)
        % add tprAtWP to legend names
        for i=1:M
            legendROC{i} = sprintf('%s: %.3f', legendNames{i}, avgTprAtWP(i));
        end

        legend( legendROC, 'Location', 'SouthEast' );
        %savePlot('./report/figures/detection/pcaselection-curve4.pdf','False Positive Rate','True Positive Rate');
    end

    % Boxplot to show variance over all methods    
    if showPlot
        figure;
        boxplot(tprAtWP, 'colors', styles(1:M), 'labels', legendNames, 'whisker', 1);
        title('TPR at WP of different methods');
        xlabel('Applied methods');
        ylabel('Average TPR');
        %savePlot('./report/figures/detection/pcaselection-boxplots4.pdf','Applied methods','Average TPR');
    end
    

end
