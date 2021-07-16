clear
clc


load('Monitoring_data_Consoli_o.mat')

npoints = 36330;

Resonant_frequencies = Resonant_frequencies(1:npoints,:);
day_no_vect = day_no_vect(1:npoints);

i = 1;
freq = Resonant_frequencies(:,i);
poscero = find(freq==0);
freq(25000:end) = freq(25000:end)-nanmean(freq(1:25000-1))*1.5/100;
freq(poscero) = 0;
Resonant_frequencies(:,i) = freq;

save('Monitoring_data_Consoli','Resonant_frequencies','day_no_vect')


figure(1)
hold on
colors = [1,0,0;0,1,0;0,0,1;0,0,0;0.5,0,0.5];
for i=1:5
plot(day_no_vect,Resonant_frequencies(:,i),'MarkerFaceColor',colors(i,:),'MarkerEdgeColor',colors(i,:),'MarkerSize',3,'Marker','o','LineStyle','none')
end
datetick('x','dd/mmm/yy','keepticks');
ylabel('Resonant frequency [Hz]')
box on
set(gca,'fontsize',24)

%% Fill missing data

for i=1:5
poscero = find(Resonant_frequencies(:,i)==0);
Resonant_frequencies(poscero,i) = NaN;     
Resonant_frequencies(:,i) = fillmissing(Resonant_frequencies(:,i),'nearest');
end


figure(2)
hold on
colors = [1,0,0;0,1,0;0,0,1;0,0,0;0.5,0,0.5];
for i=1:5
plot(day_no_vect,Resonant_frequencies(:,i),'MarkerFaceColor',colors(i,:),'MarkerEdgeColor',colors(i,:),'MarkerSize',3,'Marker','o','LineStyle','none')
end
datetick('x','dd/mmm/yy','keepticks');
ylabel('Resonant frequency [Hz]')
box on
set(gca,'fontsize',24)

%% We set_up a training period

tp = 365*24*2;   % One year

trainingpop = Resonant_frequencies(1:tp,:);


%% Step 1 - Create statistical model


% 1.1. Normalization
x = trainingpop;
meanval = mean(x);
stadval = std(x);
x = (x-repmat(meanval,size(x,1),1))./repmat(stadval,size(x,1),1);

% 1.2. PCA dimension reduction

[coeff,score,latent,tsquared,explained,mu] = pca(x);

D = latent;
explainedvar = 100*D./sum(D);

figure(3);
subplot(1,2,1)
bar(explainedvar)
xticks([1 2 3 4 5 6 7])
ylabel('Explained variance [%]')
set(gca,'XTickLabel',{'PC1','PC2','PC3','PC4','PC5'},'fontsize',24,'FontWeight','bold')

subplot(1,2,2)
bar(cumsum(explainedvar))
xticks([1 2 3 4 5])
ylabel('Cumulated explained variance [%]')
set(gca,'XTickLabel',{'PC1','PC2','PC3','PC4','PC5'},'fontsize',24,'FontWeight','bold')


% 1.3. Dimension reduction

ll = 2;
Z_vect=x*coeff;
score2=Z_vect(:,1:ll);
reconstr=score2*coeff(:,1:ll)'; 
reconstr = reconstr.*repmat(stadval,size(x,1),1)+repmat(meanval,size(x,1),1);

figure(4)
hold on
plot(day_no_vect(1:tp),trainingpop,'O','MarkerFaceColor',[0 0 1],'MarkerEdgeColor',[0 0 1],'MarkerSize',3)
plot(day_no_vect(1:tp),reconstr,'O','MarkerFaceColor',[1 0 0],'MarkerEdgeColor',[1 0 0],'MarkerSize',3)
hold off
datetick('x','dd/mmm/yy','keepticks');
ylabel('Resonant frequency [Hz]')
box on
set(gca,'fontsize',24)

% 1.4 Residuals

R = reconstr-trainingpop;

figure(5)
for i=1:5
subplot(5,2,2*i-1);
plot(day_no_vect(1:tp),R(:,i),'MarkerFaceColor',[0,0,1],'MarkerEdgeColor',[0,0,1],'MarkerSize',2,'Marker','o','LineStyle','none')
datetick('x','dd/mmm/yy','keepticks');
ylabel(['R = ',int2str(i)])
box on
end
for i=1:5
subplot(5,2,2*i)
hist(R(:,i))
box on
set(gca,'view',[90 -90])
end

%% Now we use the statistical model to perform predictions

x = Resonant_frequencies;
x = (x-repmat(meanval,size(x,1),1))./repmat(stadval,size(x,1),1);

Z_vect=x*coeff;
score2=Z_vect(:,1:ll);
reconstr=score2*coeff(:,1:ll)'; 
reconstr = reconstr.*repmat(stadval,size(x,1),1)+repmat(meanval,size(x,1),1);

figure(6)
hold on
plot(day_no_vect,Resonant_frequencies,'O','MarkerFaceColor',[0 0 1],'MarkerEdgeColor',[0 0 1],'MarkerSize',3)
plot(day_no_vect,reconstr,'O','MarkerFaceColor',[1 0 0],'MarkerEdgeColor',[1 0 0],'MarkerSize',3)
plot([day_no_vect(tp),day_no_vect(tp)],[2,8],'--r')
hold off
datetick('x','dd/mmm/yy','keepticks');
ylabel('Resonant frequency [Hz]')
box on
set(gca,'fontsize',24)
ylim([2,8])


%% Hotelling's control chart

% Residuals
R = reconstr-Resonant_frequencies;

% Parameters
gr = 4; % Group number
UCL_lim = 0.95;

% Phase I
cov_mat_data=cov(R(1:tp,:));
inv_cov_mat_data = pinv(cov_mat_data);
mean_v=mean(R(1:tp,:));
              
% Phase II
s=floor(tp/gr);
t_quadro=0;


p = size(R,2); % Number of variables
n = gr; % Group size
m = length(R)/gr; % Number of observations

% T2-statistic
for jj=1:m
     dif = (mean(R(1+(jj-1)*gr:jj*gr,:),1) - mean_v);
     t_quadro(jj,1) = n*dif*inv_cov_mat_data*dif';
end
 
% Control chart
n_cl=tp;
xx=linspace(0,nanmax(t_quadro(1:s)),n_cl);
        
t_quadro_new=t_quadro(1:s);  % Only the part corresponding to the training period

% Upper Control Limit (UCL)
[F,X]=ecdf(t_quadro_new);

[aa pos]=min(abs(F-UCL_lim)/UCL_lim);
ucl=X(pos);

for kk=1:length(t_quadro)
    time_new(1,kk)=day_no_vect(kk*gr);
end



figure(7)
hold on
pos = find(t_quadro>=ucl);
neg = find(t_quadro<ucl);
plot(time_new(pos),t_quadro(pos),'O','MarkerFaceColor',[1 0 0],'MarkerEdgeColor',[1 0 0],'MarkerSize',3)
plot(time_new(neg),t_quadro(neg),'O','MarkerFaceColor',[0 1 0],'MarkerEdgeColor',[0 1 0],'MarkerSize',3)
plot([time_new(1),time_new(end)],[ucl,ucl],'--b', 'LineWidth',4)
plot([day_no_vect(tp),day_no_vect(tp)],[0,200],'--r', 'LineWidth',4)
hold off
datetick('x','dd/mmm/yy','keepticks');
ylabel('Hotellings T2')
box on
set(gca,'fontsize',24)
xlim([time_new(1),time_new(end)])
ylim([0,200])

