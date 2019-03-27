%load data
data = load('data.txt');
labels = load('labels.txt');

%add bias
data = [ones(size(data,1),1), data];

%extract train/test
train_x = data(1:2000, :);
train_y = labels(1:2000, :);
test_x = data(2001:end, :);
test_y = labels(2001:end, :);

%sizes to train for experiment
indices = [200; 500; 800; 1000; 1500; 2000];

%accuracies of training sizes to record and plot
accuracies = zeros(6,1);

%foreach size, train and record accuracy
for m = 1:6
    %size of this experiment
    index = indices(m);

    %compute the weights from training on this size
    weights = logistic_train(train_x(1:index, :), train_y(1:index, :));

    %calculate predictions
    pred = round(sigmoid(test_x * weights));
    
    %record model accuracy
    accuracies(m) = sum(pred == test_y) / size(test_y, 1);
end

%plot the results
plot(indices, accuracies, '-o');
title('experiment 1 logistic regression results')
xlabel('training size (n)')
ylabel('accuracy (%)')
