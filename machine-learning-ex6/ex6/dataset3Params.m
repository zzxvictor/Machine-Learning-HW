function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
cArray = [0.01 0.03 0.1 0.3 1 3 10 30];
sigArray = [0.01 0.03 0.1 0.3 1 3 10 30];
size = 8;
error = zeros(size,size);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
for i = 1 : size
    for j = 1:size
        model= svmTrain(X, y, cArray(i), @(x1, x2) gaussianKernel(x1, x2, sigArray(j) ));
        pred = svmPredict(model, Xval);
        error(i,j) = mean(double(pred ~= yval));
    end
end

%find minimum
miniError = error(1,1);
miniI = 1;
miniJ = 1;
for i = 1:size
    for j = 1:size
        if(miniError >error(i,j))
            miniError = error(i,j);
            miniI = i;
            miniJ = j;
        end
    end
end
C = cArray(miniI);
sigma = sigArray(miniJ);

% =========================================================================

end

