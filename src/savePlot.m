function savePlot(filename, xLabel, yLabel)
% Helper function provided by Emti
% Save the latest figure to PDF with filename `name`
    if(nargin < 3)
        yLabel = '';
    end;
    if(nargin < 2)
        xLabel = '';
    end;
    
    prettifyPlot(xLabel, yLabel);

    % Print the file to pdf
    print('-dpdf', filename);

    % Next you should CROP PDF using pdfcrop in linux and mac
end