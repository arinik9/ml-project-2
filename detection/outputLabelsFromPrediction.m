function yHat = outputLabelsFromPrediction(probabilities, threshold)
% Takes the probabilities and returns 1/-1 labels according to the given threshold    
    yHat = 2*(probabilities > threshold) - 1;

end