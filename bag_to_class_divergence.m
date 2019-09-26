function DIV = bag_to_class_divergence(neg_distr,pos_distr,bags_distr)

% Input:    neg_distr    - the pdf of the negative class
%           pos_distr    - the pdf of the positive class
%           bags_distr   - the pdf of the bags

% Calculate the bag-to-class conditional divergence, and the Kullbak-Leibler 
% and Bhattacharyya ratio.

n_bags = length(bags_distr);
cKL = zeros(1,n_bags); % conditioning 
KL_neg = zeros(1,n_bags); % from bag to negative class
KL_pos = zeros(1,n_bags); % from bag to positive class
BH_neg = zeros(1,n_bags); % from bag to negative class
BH_pos = zeros(1,n_bags); % from bag to positive class

% In case there are components of size 0, we need to delete them
neg_comp = neg_distr.ComponentProportion;
neg_idx = find(neg_comp>0); 
neg_mu = neg_distr.mu(neg_idx,:);
neg_Sigma = neg_distr.Sigma(:,:,neg_idx);
neg_Prop = neg_comp(neg_idx);

pos_comp = pos_distr.ComponentProportion;
pos_idx = find(pos_comp>0);
pos_mu = pos_distr.mu(pos_idx,:);
pos_Sigma = pos_distr.Sigma(:,:,pos_idx);
pos_Prop = pos_comp(pos_idx);

n = 1000; % Number of samples drawn for the importance sampling
for j = 1:n_bags
    
  obj = bags_distr{j};
  
  % In case there are components of size 0, we need to delete them
  obj_comp = obj.ComponentProportion;
  obj_idx = find(obj_comp>0);
  obj_mu = obj.mu(obj_idx,:);
  obj_Sigma = obj.Sigma(:,:,obj_idx);
  obj_Prop = obj_comp(obj_idx);
  
  % Create a distribution from the neg, the pos, and the bag
  Mu_imp = [neg_mu; pos_mu; obj_mu];
  Sigma_imp = cat(3,neg_Sigma, pos_Sigma, obj_Sigma);
  P_imp = [neg_Prop pos_Prop obj_Prop]; 
  imp_distr = gmdistribution(Mu_imp,Sigma_imp,P_imp);

  % Importance sampling
  X = random(imp_distr,n); 
  imp = pdf(imp_distr,X);
  imp(imp<eps) = eps;
  
  p_neg = pdf(neg_distr,X);
  p_pos = pdf(pos_distr,X);
  p_bag = pdf(obj,X);        % bag pdf
  
  p_neg(p_neg<eps) = eps;
  p_pos(p_pos<eps) = eps;
  p_bag(p_bag<eps) = eps;
    
  w = p_pos./p_neg; % 
      
  % Calculate each term
  bh_terms_neg = sqrt(p_bag.*p_neg)./imp;
  bh_terms_pos = sqrt(p_bag.*p_pos)./imp;
  kl_terms_neg = p_bag.*log(p_bag./p_neg)./imp;
  kl_terms_pos = p_bag.*log(p_bag./p_pos)./imp;
  
  kl_terms_neg(p_bag == eps) = eps; % 0log0 = 0
  kl_terms_pos(p_bag == eps) = eps;
  
  ckl_terms = w.*kl_terms_neg;
  ckl_terms(p_pos == eps) = eps; 
  
  cKL(j)  = sum(ckl_terms)/n;
  KL_neg(j) = sum(kl_terms_neg)/n;    
  KL_pos(j) = sum(kl_terms_pos)/n;
  BH_neg(j) = -log(sum(bh_terms_neg)/n);    
  BH_pos(j) = -log(sum(bh_terms_pos)/n);
end

rBH = BH_neg./BH_pos;
rKL = KL_neg./KL_pos;

DIV = [rBH; rKL; cKL];




