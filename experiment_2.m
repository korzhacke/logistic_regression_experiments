%load data
load('ad_data.mat')

%parameters to test
par  = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];

%add bias
X_train = [X_train, ones(size(X_train,1),1)];
X_test = [X_test, ones(size(X_test,1),1)];

%AUCs and numbers of features selected to plot
AUCs = zeros(size(par,2),1);
num_features = zeros(size(par,2),1);

for m = 1:size(par,2)
    %calculate weights and bias from logistic l1 reg with m parameter
    [w, c] = logistic_l1_train(X_train, y_train, par(m));
    
    %calculate AUC for m param
    [~,~,~,AUCs(m)] = perfcurve(y_test, X_test*w + c,1);
    
    %calculate num features selected for m param
    num_features(m) = nnz(w);
end


%plot the AUC results
figure
plot(par, AUCs, '-o');
title('experiment 2 AUC experiments')
xlabel('regularization par')
ylabel('AUC')

%plot the number of selected features results
figure
plot(par, num_features, '-o');
title('experiment 2 number of features selected experiments')
xlabel('regularization par')
ylabel('features selected')


