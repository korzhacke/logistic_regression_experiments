function [weights] = logistic_train(data, labels, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
% data = n * (d+1) matrix withn samples and d features, where
% column d+1 is all ones (corresponding to the intercept term)
% labels = n * 1 vector of class labels (taking values 0 or 1)
% epsilon = optional argument specifying the convergence
% criterion - if the change in the absolute difference in
% predictions, from one iteration to the next, averaged across
% input features, is less than epsilon, then halt
% (if unspecified, use a default value of 1e-5)
% maxiter = optional argument that specifies the maximum number of
% iterations to execute (useful when debugging in case your
% code is not converging correctly!)
% (if unspecified can be set to 1000)
%
% OUTPUT:
% weights = (d+1) * 1 vector of weights where the weights correspond to
% the columns of "data"
%
    %handle optional args
    switch nargin
        case 2
            maxiter = 1000;
            epsilon = 1e-5;
        case 3
            maxiter = 1000;
    end
    
    % initialize weights
    weights = zeros(size(data,2), 1);
    
    for m = 1:maxiter
        %compute predictions
        pred = sigmoid(data * weights);
        
        %subtract truths
        b = pred - labels;
        
        %compute gradient
        gradient = data.'*b / size(labels, 1);
        
        %update weights
        weights = weights - gradient;
        
        %check change in abs value of predictions
        if mean(abs(pred - sigmoid(data*weights))) < epsilon
            break;
        end
    end
    
end
