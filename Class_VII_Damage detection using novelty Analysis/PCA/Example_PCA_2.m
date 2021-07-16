

clear
clc

%% Read results

load('Monitoring_Data.mat')
expfreq = expfreq';
all_temp_TH = all_temp_TH';

figure(1)
hold on
colors = [1,0,0;0,1,0;0,0,1;1,1,0;0,1,1;0,0,0;0.5,0,0.5];
for i=1:7
plot(day_no_vect,expfreq(:,i),'MarkerFaceColor',colors(i,:),'MarkerEdgeColor',colors(i,:),'MarkerSize',5,'Marker','o','LineStyle','none')
end
datetick('x','dd/mmm/yy','keepticks');
ylabel('Resonant frequency [Hz]')
box on

figure(2)
hold on
for i=1:2
plot(day_no_vect,all_temp_TH(:,i),'MarkerFaceColor',colors(i,:),'MarkerEdgeColor',colors(i,:),'MarkerSize',5,'Marker','o','LineStyle','none')
end
hold off
datetick('x','dd/mmm/yy','keepticks');
ylabel('Temperature [Celsius degrees]')
box on

%% PCA

% Normalization
x = expfreq;
meanval = mean(x);
stadval = std(x);
x = (x-repmat(meanval,size(x,1),1))./repmat(stadval,size(x,1),1);


[coeff,score,latent,tsquared,explained,mu] = pca(x);


COV = x'*x;
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
xticks([1 2 3 4 5 6 7])
ylabel('Explained variance [%]')
set(gca,'XTickLabel',{'PC1','PC2','PC3','PC4','PC5','PC6','PC7'},'fontsize',24,'FontWeight','bold')

figure(6);
bar(cumsum(explainedvar))
xticks([1 2 3 4 5 6 7])
ylabel('Cumulated explained variance [%]')
set(gca,'XTickLabel',{'PC1','PC2','PC3','PC4','PC5','PC6','PC7'},'fontsize',24,'FontWeight','bold')



% Dimension reduction
ll = 2;
Z_vect=x*coeff2;
score2=Z_vect(:,1:ll);
reconstr=score2*coeff2(:,1:ll)'; 
reconstr = reconstr.*repmat(stadval,size(x,1),1)+repmat(meanval,size(x,1),1);

figure(7)
hold on
plot(day_no_vect,expfreq,'O','MarkerFaceColor',[0 0 1],'MarkerEdgeColor',[0 0 1],'MarkerSize',3)
plot(day_no_vect,reconstr,'O','MarkerFaceColor',[1 0 0],'MarkerEdgeColor',[1 0 0],'MarkerSize',3)
hold off
datetick('x','dd/mmm/yy','keepticks');
ylabel('Resonant frequency [Hz]')
box on
set(gca,'fontsize',24)


%% Residuals


R = reconstr-expfreq;

figure(8)
for i=1:7
subplot(7,2,2*i-1);
plot(day_no_vect,R(:,i),'MarkerFaceColor',[0,0,1],'MarkerEdgeColor',[0,0,1],'MarkerSize',2,'Marker','o','LineStyle','none')
datetick('x','dd/mmm/yy','keepticks');
ylabel(['R = ',int2str(i)])
box on
end
for i=1:7
subplot(7,2,2*i)
hist(R(:,i))
box on
set(gca,'view',[90 -90])
end
