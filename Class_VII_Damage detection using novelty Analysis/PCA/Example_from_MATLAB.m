
clc
clear

load hald


x = ingredients;

% Normalization

x = (x-repmat(mean(x),size(x,1),1))./repmat(std(x),size(x,1),1);


[coeff,score,latent,tsquared,explained,mu] = pca(x);


COV = x'*x;
[V,D] = eig(COV);

[latent2,b]=sort(diag(D),'descend');
coeff2 = V(:,b);
D = latent2;
explainedvar = 100*D./sum(D);
score2 = x*coeff2;



