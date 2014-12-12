% Automatically calls fastROC() to show multiple curves, one for each
% prediction vector provided.
%
% labels        Nx1 vector
% predictions   NxM vector, M being the number of predictions to show
%
% if showPlot == true => single plot with multiple curves is shown.
% legendNames is a cell list (optional) with the name to show for each
% prediction in the legend.
%
% Returns tprAtWP where each element is the tprAtWP of each prediction
% vector given as input.
%
function tprAtWP = kCVEvaluateMultipleMethods( labels, predictions, ...
                                            showPlot, legendNames )                                   
                                        
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
    styles = {'r','b','k','m','g','r--', 'b--', 'k--','m--','g--'};

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

    for i=1:M
        tprAtWP(:,i) = kCVfastROC( labels(:,:,i), predictions(:,:,i), 1, 0, '', styles{i} );
        avgTprAtWP(i) = mean(tprAtWP(:,i));
        fprintf('avgTprAtWP: %d \n', avgTprAtWP(i));
        if showPlot
            hold on;
        end
    end

    if showPlot && ~isempty(legendNames)
        % add tprAtWP to legend names
        for i=1:M
            legendNames{i} = sprintf('%s: %.3f', legendNames{i}, tprAtWP(i));
        end
        legend( legendNames, 'Location', 'NorthWest' );
        title('Coucou');
    end

end
