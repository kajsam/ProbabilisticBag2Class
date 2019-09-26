function [x_bags, acracy, K_bag, AUC, neg_distr, pos_distr, bags_test, thresh] = ...
    pftas_patfold(n_inst, x_bags, neg_distr,pos_distr, bags_test, thresh, fold, magn)
save_to_base(1)
% The pftas feature vector for each image patch has been extracted by a
% python program (my first!).

% Requires: find_thresh.m, gaussian_mixture.m, bag_to_class_divergence.m, 
%           AUC_ROC.m
% find_thresh:          Need to train the threshold for the classifier. 
%                       It's a bit time consuming.
% gaussian_mixture:     Fits a gaussian mixture model to the data
% bag_to_class diverge: Calulates the divergences.
% AUC_ROC:              Calculates the accuracy (and AUC)

% Input:    n_inst: Number of instance per image. Need to know this
%           magn:   Magnification, to load the data    
%           x_bags: So we don't have to do the PCA again

% Output:   x_bags: So we don't have to do the PCA again
%           acracy:    The accuracy for all repetitions and all divergences
acracy = zeros(1,3);
AUC = zeros(1,3);

% Load the training data
filename = strcat('/Users/kam025/ptn/pftas_train_fold',fold,'_',magn,'.mat')
load(filename,'pftas_train')
pftas_train = pftas_train';

size(pftas_train)

train_files = textread(strcat('/Users/kam025/ptn/fold',fold,'_',magn,'_train.txt'),'%s','delimiter',',');
train_class = 2*ones(size(train_files));

if strcmp(train_files{1}(52),'M')
  train_class(1) = 1;
elseif strcmp(train_files{1}(52),'B')
  train_class(1) = 0;
end
for i = 2: length(train_files)
  if strcmp(train_files{i}(51),'M')
    train_class(i) = 1;
  elseif strcmp(train_files{i}(51),'B')
    train_class(i) = 0;
  end
end

n_train = size(pftas_train,1)/n_inst;           % # train images
n_feat = size(pftas_train,2);                 % length of feature vector
pftas_train_3D = reshape(pftas_train,n_inst, n_train, n_feat);

% Load the test data
filename = strcat('/Users/kam025/ptn/pftas_test_fold',fold,'_',magn,'.mat');
load(filename,'pftas_test')
pftas_test = pftas_test';

test_files = textread(strcat('/Users/kam025/ptn/fold',fold,'_',magn,'_test.txt'),'%s','delimiter',',');
test_class = 2*ones(size(test_files));

if strcmp(test_files{1}(51),'M')
  test_class(1) = 1;
elseif strcmp(test_files{1}(51),'B')
  test_class(1) = 0;
end
for i = 2: length(test_files)
  if strcmp(test_files{i}(50),'M')
    test_class(i) = 1;
  elseif strcmp(test_files{i}(50),'B')
    test_class(i) = 0;
  end
end

n_test = size(pftas_test,1)/n_inst;           % # test images
pftas_test_3D = reshape(pftas_test,n_inst, n_test, n_feat);

%% Transform the data using PCA

% No doubt, some of the features are highly correlated, and we don't need
% all of them. PCA is the common approach. We need to set a parameter that
% controls the number of final dimensions, and we'll use the 'explained
% variance' for that.

expl_var = 90  %%%%%%%%%%%%%%%%%%%%%% PARAMETER %%%%%%%%%%%%%%%%%%%%%%%%

if isempty(x_bags)
  data = [pftas_train; pftas_test]; % (1000 instances x 162 feature values)
  % Normalisation, mean centering is included in the pca function
  data = data./sqrt(var(data)); 
  [~,score,latent,~,explained] = pca(data);

  % Have a look
  figure(2), subplot(2,1,1)
  plot(1:100,latent(1:100)) 
  xlabel(sum(explained(1:100)))
  title('Scree plots')
  subplot(2,1,2)
  plot(1:25,latent(1:25)) 
  xlabel(sum(explained(1:25))) 
  drawnow
  
  % Keep only those that contribute
  for dim = 1: n_feat
    if sum(explained(1:dim)) > expl_var
      break
    end
  end
  disp(dim) % Let's see what we got
    
  data = score(:,1:dim);
  % The bags are identified
  n_bag = n_train+n_test;
  x_bags = reshape(data, n_inst, n_bag, dim); 
  
  

dim = size(x_bags,3) % Let's see what we got

% Give them their class labels
bag_class = [train_class; test_class]';
% And identify the indexes
neg_idx = find(bag_class == 0);
pos_idx = find(bag_class == 1);
 
%% Parameters for the EM-algorithm:  
% There's nothing much going on here. Increasing maxiter (maximum number of
% iterations) and reps (number of repetitions) will give a better fit, but
% you'll pay by time consumption. The 'reg' parameter must be a small
% positive number, or else, the covariance matrices risk being
% non-invertible, and the whole procedure just stops. 

maxiter = 100; %1000;
reps = 5 % 10;
reg = 1e-3; % Avoiding non-invertible matrices
EMparam = [maxiter reps reg];
Kmax = 100; %%%%%%%%%%%%%%%%%%%%%% PARAMETER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning on all % This will tell you if MaxIter reaches its limit

%% Now for the actual method

% We need a validation set to find the threshold, so we split the
% training set in two.
neg_train = find(train_class == 0);
pos_train = find(train_class == 1);

validate = randsample(n_train, floor(n_train/2));  
validate_class = train_class(validate);
pos_validate = validate(validate_class == 1);
neg_validate = validate(validate_class == 0);

neg_train = setdiff(neg_train,neg_validate);
pos_train = setdiff(pos_train,pos_validate);     

sub_inst = 0.10;   %%%%%%%%%%%%%%%%%%%% PARAMATER %%%%%%%%%%%%%%%%%%%%


% Here we go, finding the threshold, and at the same time finding the
% number of components for the GMM. 
whos neg_train
if isempty(thresh)
  [thresh, K_bag, K_pos, K_neg] = ...
      find_thresh(x_bags, neg_train, pos_train, validate, bag_class, EMparam, Kmax, sub_inst); % 
  thresh = thresh/(length(validate)+2) 
end
  
[K_pos K_neg]
   
save_to_base(1)
% Fit the bag GMMs for the test set. 
nt_bag = length(test_class) % number of bags in the test set
bags_test = cell(1, nt_bag); 
bags_distr = cell(nt_bag,1);

test = setdiff(1:length(bag_class), 1:length(train_class));
    
for j = 1: nt_bag   
  
  bags_distr(j) = gaussian_mixture(K_bag,squeeze(x_bags(:,test(j),:)),EMparam);
  bags_test{j} = bags_distr{j,1};
end
      
  % The amount of instances in each class is enormous, so we'll do a
  % subsampling of instances from each image inthe training set. This is
  % necessary to keep time consumption down. 
  

  rand = sort(randsample(n_inst, round(sub_inst*n_inst)));
  x_neg = x_bags(rand,setdiff(neg_idx,test),:);
  x_pos = x_bags(rand,setdiff(pos_idx,test),:);
         
  %% Fit the distribution to the classes
            
  % Fit a Gaussian mixture model to all negative (positive) instances
  x_neg_2D = reshape(x_neg,size(x_neg,1)*size(x_neg,2),dim);
  % maxK = 200; %round(size(x_neg_2D,1)/(dim*5))
  
%   Kvec = K_bag:5:Kmax;
%   AIC = zeros(1,length(Kvec));
%   obj = gaussian_mixture(Kvec, x_neg_2D, EMparam);
%   for k = 1: length(Kvec)
%     if any(obj{k}.ComponentProportion == 0)
%       AIC(k) = inf;
%     else
%       AIC(k) = obj{k}.AIC;
%     end    
%   end
%   [~, kaic] = min(AIC);
%   Kvec(kaic)
%   neg_distr = obj{kaic}; 
 
   neg_distr = gaussian_mixture(K_neg, x_neg_2D, EMparam);
   neg_distr = neg_distr{1};
        
  x_pos_2D = reshape(x_pos,size(x_pos,1)*size(x_pos,2),dim);
  % maxK = 200; % round(size(x_pos_2D,1)/(dim*5))
  
%   AIC = zeros(1,length(Kvec));
%   obj = gaussian_mixture(Kvec, x_pos_2D, EMparam);
%   for k = 1: length(Kvec)
%     if any(obj{k}.ComponentProportion == 0)
%       AIC(k) = inf;
%     else
%       AIC(k) = obj{k}.AIC;
%     end    
%   end
%   [~, kaic] = min(AIC);
%   Kvec(kaic)
%   pos_distr = obj{kaic}; 
    
   pos_distr = gaussian_mixture(K_pos,x_pos_2D,EMparam);
   pos_distr = pos_distr{1};
   
   % return
end
  
  % Calculate the divergences
  bag2class_div = bag_to_class_divergence(neg_distr,pos_distr,bags_test);
      
  rBH = bag2class_div(1,:);
  rKL = bag2class_div(2,:);
  cKL = bag2class_div(3,:);
    
  % Classification. Simple threshold.
   
  [AUC(1),~,ACC(1,:)]= AUC_ROC(rBH,bag_class(test));
  acracy(1) = ACC(1,round(thresh(1)*length(test)+2));
  
  [AUC(2),~,ACC(2,:)] = AUC_ROC(rKL,bag_class(test));
  
  acracy(2) = ACC(2,round(thresh(2)*length(test)+2));
  
  [AUC(3),~,ACC(3,:)] = AUC_ROC(cKL,bag_class(test));
  acracy(3) = ACC(3,round(thresh(3)*length(test)+2));
  acracy
  toc
  save_to_base(1)


K_bag
 



