function [thresh, K_bag, K_pos, K_neg] = find_thresh(x_bags, neg_train, pos_train, validate, ...
                              bag_class, EMparam, Kmax, sub_inst)
thresh = [];
warning off all                          
                          
[n_inst, ~, dim] = size(x_bags);

% Fit the bag GMMs. Find minimum AIC. Make sure they come from both classes
find_Kneg = randsample(neg_train, 5);  
find_Kpos = randsample(pos_train, 5);  

find_K = [find_Kneg find_Kpos];

% maxK = round(n_inst/5)

k_AIC = zeros(1,10);
Kvec = 1:Kmax;
AIC = zeros(1,length(Kvec));
for j = 1:10
  obj = gaussian_mixture(Kvec, squeeze(x_bags(:,find_K(j),:)),EMparam);
  for k = 1: length(Kvec)
    if any(obj{k}.ComponentProportion == 0)
      AIC(k) = inf;
    else
      AIC(k) = obj{k}.AIC;
    end    
  end
  [~,kaic] = min(AIC);
  k_AIC(j) = Kvec(kaic);
end
 
K_bag = round(median(k_AIC));
% return
% inst_comp = round(n_inst/minK_bag)
  
tic   
% The neg (pos) bags in the training set

rand = sort(randsample(n_inst, round(sub_inst*n_inst)));
x_pos = x_bags(rand,pos_train,:);
x_pos_2D = reshape(x_pos,size(x_pos,1)*size(x_pos,2),dim);

% minK = minK_bag;
% maxK = Kmax; %200; %round(size(x_pos_2D,1)/(inst_comp))

% step = round((maxK-minK)/5)

tic
Kvec = K_bag:5:Kmax
AIC = zeros(1,length(Kvec));

obj = gaussian_mixture(Kvec, x_pos_2D, EMparam);

for k = 1: length(Kvec)
  if any(obj{k}.ComponentProportion == 0)
    AIC(k) = inf;
  else
    AIC(k) = obj{k}.AIC;
  end    
end
[~, kaic] = min(AIC)
K_pos = Kvec(kaic)
pos_distr = obj{kaic};
toc  

x_neg = x_bags(rand,neg_train,:);
x_neg_2D = reshape(x_neg,size(x_neg,1)*size(x_neg,2),dim);
% maxK = Kmax; %200; %round(size(x_neg_2D,1)/(inst_comp))
% Kvec = K_bag:Kmax;
AIC = zeros(1,length(Kvec));
obj = gaussian_mixture(Kvec, x_neg_2D, EMparam);
for k = 1: length(Kvec)
  if any(obj{k}.ComponentProportion == 0)
    AIC(k) = inf;
  else
    AIC(k) = obj{k}.AIC;
  end    
end
[~, kaic] = min(AIC);
K_neg = Kvec(kaic)
neg_distr = obj{kaic}; 

      
%% Fit the distribution to the classes
        
% Fit a Gaussian mixture model to all negative instances
% neg_distr = gaussian_mixture(minK_neg, x_neg_2D, EMparam);

      
% Fit a Gaussian mixture model to all positive instances
% pos_distr = gaussian_mixture(minK_pos,x_pos_2D,EMparam);


%% The bags in the test set
nf_bag = length(validate); % number of bags in the test set
bags_f = cell(1, nf_bag);
bags_distr = cell(nf_bag,1);
for j = 1: nf_bag
  bags_distr(j) = gaussian_mixture(K_bag,squeeze(x_bags(:,validate(j),:)),EMparam);
  bags_f{j} = bags_distr{j,1};
  if mod(j,150) == 0
  end
end

bag2class_div = bag_to_class_divergence(neg_distr,pos_distr,bags_f);
toc

rBH = bag2class_div(1,:);
rKL = bag2class_div(2,:);
cKL = bag2class_div(3,:);
    
% Classification. Simple threshold.
    
[AUC(1),~,ACC]= AUC_ROC(rBH,bag_class(validate));
[value(1), thresh(1)] = max(ACC);
[AUC(2),~,ACC] = AUC_ROC(rKL,bag_class(validate));
[value(2), thresh(2)] = max(ACC);
[AUC(3),~,ACC] = AUC_ROC(cKL,bag_class(validate));
[value(3),thresh(3)] = max(ACC);
value
