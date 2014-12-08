function [coeff, score, latent, tsquared, explained, explCum] = applyPCA(X, plot_flag)
% Applies PCA on given parameter and plots variance explained by Principal
% Component if the plot_flag is set to 1
% WARNING: This function uses Matlab's pca function and not pca from Piotr
% toolbox which shadows matlab one when the toolbox is loaded

    if (nargin < 2)
        plot_flag = 0;
    end

    % Apply PCA
    [coeff, score, latent, tsquared, explained] = pca(X);
    
    % Compute cumulative variance explained on principal component
    explCum = cumsum(explained);

    % Plot variance explained per Principal Component (PC)
    if (plot_flag ~= 0)
        
        % Plot only on the first half of the PC
        n = size(explained,1)/2;

        e1 = explained(1:n);
        e2 = explCum(1:n);

        figure();
        [ax,hBar,hLine] = plotyy(1:n, e1, 1:n, e2, 'bar', 'plot');
        title('PCA on Features')
        xlabel('Principal Component')
        ylabel(ax(1),'Variance Explained per PC')
        ylabel(ax(2),'Cumulative Variance Explained (%)')
        hLine.LineWidth = 3;
        hLine.Color = [0,0.7,0.7];
        ylim(ax(2),[1 100]);
        
    end

end