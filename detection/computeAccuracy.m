function acc = computeAccuracy(trueLabels, predictedLabels)
% Accuracy = (true positives + true negatives) / number of examples
    predictedPositive = predictedLabels == 1;

    tp = sum(trueLabels(predictedPositive) == 1);
    tn = sum(trueLabels(~predictedPositive) == -1);

    acc = (tp + tn) / length(trueLabels);

end