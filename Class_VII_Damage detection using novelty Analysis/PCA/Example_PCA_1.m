
clc
clear
rng('default');  % Generate Random Numbers That Are Repeatable

%% PCA example


dt = 1/800;      % Time step
T = 20;          % Time length
t = 0:dt:T;      % Time vector
np = numel(t);   % Number of data points

% Generation of variables
signal = sin(t*2*pi*10);         
noise = mvnrnd([0,0],[1,0.05],np);
var1 = signal+noise(:,1)';          % Variable 1
var2 = 10+var1+noise(:,2)';         % Variable 2

figure(1)
hold on
plot(t,var1)
plot(t,var2)
hold off
box on
legend(['Var 1';'Var 2'],'FontSize',24)
xlabel('Time','FontSize',24)

figure(2)
plot(var1,var2,'x')
axis equal 
xlabel('Var 1','FontSize',24)
ylabel('Var 2','FontSize',24)

%% PCA analysis

% Normalization
x = [var1',var2'];
meanval = mean(x);
stadval = std(x);
x = (x-repmat(meanval,size(x,1),1))./repmat(stadval,size(x,1),1);

figure(3)
hold on
plot(t,x(:,1))
plot(t,x(:,2))
hold off
box on
legend(['Var 1';'Var 2'],'FontSize',24)
xlabel('Time','FontSize',24)

figure(4)
plot(x(:,1),x(:,2),'x')
axis equal 
xlabel('Var 1','FontSize',24)
ylabel('Var 2','FontSize',24)

[coeff,score,latent,tsquared,explained,mu] = pca(x);

COV = x'*x/(np-1);
COV = cov(x);
[V,D] = eig(COV);

[latent2,b]=sort(diag(D),'descend');
coeff2 = V(:,b);
D = latent2;
explainedvar = 100*D./sum(D);
score2 = x*coeff2;

figure(4)
hold on
plot(x(:,1),x(:,2),'bx')
plot([0,V(1,1)],[0,V(2,1)],'r','LineWidth',2)
plot([0,V(1,2)],[0,V(2,2)],'r','LineWidth',2)
hold off
axis equal
xlabel('Var 1','FontSize',24)
ylabel('Var 2','FontSize',24)
box on


figure(5);
bar(explainedvar)
xticks([1 2])
ylabel('Explained variance [%]')
set(gca,'XTickLabel',{'PC1','PC2'},'fontsize',24,'FontWeight','bold')


% Dimension reduction
ll = 1;
Z_vect=x*coeff2;
score2=Z_vect(:,1:ll);
reconstr=score2*coeff2(:,1:ll)'; 
reconstr = reconstr.*repmat(stadval,size(x,1),1)+repmat(meanval,size(x,1),1);

figure(6)
hold on
plot(t,var1,'b')
plot(t,var2,'g')
scatter(t,reconstr(:,1),12,'r','filled')
scatter(t,reconstr(:,2),12,'r','filled')
hold off
box on
legend(['Var 1        ';'Var 2        ';'Reconstructed'],'FontSize',24)
xlabel('Time','FontSize',24)


%% Residuals

Residuals1 = var1-reconstr(:,1)';
Residuals2 = var2-reconstr(:,2)';

figure(7)
subplot(2,2,1)
plot(t,Residuals1,'b')
xlabel('Time','FontSize',24)
ylabel('Residuals Var. 1','FontSize',24)
subplot(2,2,3)
plot(t,Residuals1,'b')
xlabel('Time','FontSize',24)
ylabel('Residuals Var. 2','FontSize',24)
subplot(2,2,2)
hist(Residuals1)
box on
set(gca,'view',[90 -90])
subplot(2,2,4)
hist(Residuals2)
box on
set(gca,'view',[90 -90])
