function prettifyPlot(xLabel, yLabel)
% Makes the current figure look nice (font size, etc)
    if(nargin < 2)
        yLabel = '';
    end;
    if(nargin < 1)
        xLabel = '';
    end;
    
    hx = xlabel(xLabel);
    hy = ylabel(yLabel);

    set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
    set([hx; hy],'fontsize',18,'fontname','Helvetica','color',[.3 .3 .3]);
    grid on;
end