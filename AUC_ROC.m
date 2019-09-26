function [AUC, SP, ACC] = AUC_ROC(div,bag_class)

n_bag = length(bag_class);

% Sort the bag-to-class divergence values in descending order
[sort_div, idx] = sort(div,'descend');
sort_class = bag_class(idx);     % and the class labels
idx_pos = find(sort_class == 1); % identify the positive bags
n_pos = length(idx_pos);

% Set initial values for sensitivity and specificity
SE = zeros(1,n_bag+2); SE(end) = 1;
SP = ones(1,n_bag+2);  SP(end) = 0;
ACC = zeros(1,n_bag+2);
for j = 1: n_bag
  label = zeros(1,n_bag);      % All bags are negative
  % Define threshold at pos bag with jth largest value
  thresh = sort_div(j);
  label(sort_div>=thresh) = 1;   % All bags above threshold are positive
  
  % Calculate sensitivity and specificity
  CP = classperf(sort_class,label,'Positive', 1, 'Negative', 0); 
  SE(j+1) = CP.Sensitivity;
  SP(j+1) = CP.Specificity;
  ACC(j+1) = 1- CP.ErrorRate;
end

AUC = trapz(1-SP,SE); % Area under ROC curve

SP = SP(1:end-1);