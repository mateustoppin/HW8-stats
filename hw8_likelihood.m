%%MATLAB Likelihood Script for Homework 8


%  Choose which data set you will read 

dataset0 = readmatrix("syntheticH0.csv");
dataset1 = readmatrix("syntheticH1.csv");
%dataset0 = readmatrix("benignfull.csv");
%dataset1 = readmatrix("malignantfull.csv");
%[dataset0 dataset1] = read_cats_dogs;


[n0, d0] = size(dataset0);
[n1, d1] = size(dataset1);
if (d0 == d1)
    d = d0;
else
    error("dataset0 and dataset1 have a different number of columns.")
end

%Split dataset into training and test data.
train0 = dataset0(1:floor(n0/2),:);
test0 = dataset0(floor(n0/2)+1:n0,:);
train1 = dataset1(1:floor(n1/2),:);
test1 = dataset1(floor(n1/2)+1:n1,:);
n0train = size(train0,1);
n1train = size(train1,1);
n0test = size(test0,1);
n1test = size(test1,1);

%Estimate mean vectors and covariance matrices from training data.
mu0 = mean(train0);
mu1 = mean(train1);
sigma0 = cov(train0);
sigma1 = cov(train1);

%Apply decision rules.
H0guesses_idcov = zeros(n0test,1);
H1guesses_idcov = zeros(n1test,1);

for i = 1:n0test
    currentdata= test0(i,:);
    %%%Problem 8.4(a) code goes here. Compute the pdf of the gaussian model
    %%% at current data under hypothesis H0 (Gaussian(mu0,I)) and
    %%% under hypothesis H1 (Gaussian(mu1,I), where I is the identity
    %%% Use I = eye(d0) to generate an identity matrix with the same
    %%% dimension as the number of features in the data. 
    %%% Pick the one that is largest, and store the decision as 1 if H1 
    %%% is more likely, 0 if H0 is more likely in the array below.
    
    H0guesses_idcov(i) = mvnpdf(currentdata, mu0);
end

for i = 1:n1test
    currentdata = test1(i,:);
    %%%Problem 8.4(a) code goes here. Do the same thing as above for the
    %%%test data from class 1.
    
    
    
    H1guesses_idcov(i) = mvnpdf(currentdata, mu1);
end

Pe_idcov = proberror(H0guesses_idcov,H1guesses_idcov);
fprintf('Probability of error for identity covariance is %.2g.\n',Pe_idcov);

H0guesses_samecov = zeros(n0test,1);
H1guesses_samecov = zeros(n1test,1);

%%%Problem 8.4(b) code goes here %%%
%%% First, compute the pooled covariance matrix as indicated in the homework
%%% sheet.  
PooledVar = (1/(n0train + n1train - 2)) * (((n0test - 1) * sigma0) + ((n1train - 1) * sigma1));

for i = 1:n0test
    currentdata = test0(i,:);
    %%%Problem 8.3(b) code goes here. Compute the pdf of the gaussian model
    %%% at current data under hypothesis H0 (Gaussian(mu0,PooledVar)) and
    %%% under hypothesis H1 (Gaussian(mu1,PooledVar). 
    %%% Pick the one that is largest, and store the decision as 1 if H1 
    %%% is more likely, 0 if H0 is more likely in the array below.

    
    H0guesses_samecov(i) = mvnpdf(currentdata, mu0, sigma0);
end

for i = 1:n1test
    currentdata = test1(i,:);
        %%%Problem 8.4(b) code goes here. Repeat the above for data from
        %%%test1

    H1guesses_samecov(i) = mvnpdf(currentdata, mu1, sigma1);
end

Pe_samecov = proberror(H0guesses_samecov,H1guesses_samecov);
fprintf('Probability of error for same covariance is %.2g.\n',Pe_samecov);




H0guesses_diffcov = zeros(n0test,1);
H1guesses_diffcov = zeros(n1test,1);

for i = 1:n0test
    currentdata = test0(i,:);
    %%%Problem 8.4(c) code goes here. Compute the pdf of the gaussian model
    %%% at current data under hypothesis H0 (Gaussian(mu0,sigma0)) and
    %%% under hypothesis H1 (Gaussian(mu1,sigma1), where sigma0 and sigma1
    %%% are the covariances computed from the training data. 
    %%% Pick the one that is largest, and store the decision as 1 if H1 
    %%% is more likely, 0 if H0 is more likely in the array below.
    
    
    
    H0guesses_diffcov(i) = mvnpdf(currentdata, mu1, sigma1) > mvnpdf(currentdata, mu0, sigma0);
end

for i = 1:n1test
    currentdata = test1(i,:);
    
    %%%Problem 8.4(c) code goes here.  Repeat the
    %%% above for data from test1 %%%
    
    H1guesses_diffcov(i) = mvnpdf(currentdata, mu1, sigma1) > mvnpdf(currentdata, mu0, sigma0);
end

Pe_diffcov = proberror(H0guesses_diffcov,H1guesses_diffcov);
fprintf('Probability of error for different covariances is %.2g.\n',Pe_diffcov);


