function distr = gaussian_mixture(Kvec,x,EMparam)

maxiter = EMparam(1);
reps = EMparam(2);
reg  = EMparam(3);
if reps > 3
  start = 'randSample';
else
  start = 'plus';
end
options = statset('MaxIter',maxiter); 

fit_distr = cell(1,length(Kvec));
for k = 1:length(Kvec)
  fit_distr{k} = fitgmdist(x,Kvec(k),'Regularize',reg,'Options',options,...
      'Replicates',reps, 'Start',start);
end
  
distr = fit_distr;