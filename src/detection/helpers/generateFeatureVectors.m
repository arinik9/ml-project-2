function X = generateFeatureVectors(feats)
    D = numel(feats{1});  % feature dimensionality
    X = zeros([size(feats, 1) D]);

    % convert features to a vectors of D dimensions
    for i=1:length(feats)
        X(i,:) = feats{i}(:); 
    end;
end