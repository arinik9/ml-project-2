function [explained, cumExplained, nPC] = pcaExplainedVariance(latent, plot_flag, nPConplot, explainedPercent)
% Takes latent variables from pca() function and compute explained variance
% per principal component and cumulated explained variance per component.
% It also computes how many principal components (nPC) should be kept to have a
% given percentage of cumulated explained variance (explainedPercent).
% It plots explained variance per PC on a barplot and cumulated explained
% variance on a specified number of PC (nPConplot).

    if (nargin < 2)
        plot_flag = 1;
    end

    if (nargin < 3)
        nPConplot = floor(size(latent,1) / 4); % because we have a lot of PC
    end
    
    if (nargin < 4)
        explainedPercent = 0.95;
    end

    % Percentage of the total variance explained by each principal component
    explained = latent ./sum(latent);

    % Cumulative percentage of the total variance explained by each principal component
    cumExplained = (cumsum(latent)./sum(latent));

    % cumExplained is the cumulated explained variance on PC, finding the
    % index of the first cumExplained variable that is greater or equal to
    % the given explainedPercent will give the number of PC to keep
    idxs = find(cumExplained >= explainedPercent);
    nPC = idxs(1);
    
    if (plot_flag == 1)
        
        % Plot only on nPlot first Principal Components
        e1 = explained(1:nPConplot);
        e2 = 100 * cumExplained(1:nPConplot);
        
        figure();
        [ax,~,hLine] = plotyy(1:nPConplot, e1, 1:nPConplot, e2, 'bar', 'plot');
        title('PCA on Features')
        xlabel('Principal Component')
        ylabel(ax(1),'Variance Explained per PC')
        ylabel(ax(2),'Total Variance Explained (%)')
        hLine.LineWidth = 3;
        hLine.Color = [0,0.7,0.7];
        ylim(ax(2),[1 100]);
    end

end