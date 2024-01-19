

data {
  int<lower=0> nt;
  int<lower=0> ng; //number of games
  int<lower=0> ht[ng]; //home team index
  int<lower=0> at[ng]; //away team index
  int<lower=0> s1[ng]; //score home team
  int<lower=0> s2[ng]; //score away team
  int<lower=0> np; //number of predicted games
  int<lower=0> htnew[np]; //home team index for prediction
  int<lower=0> atnew[np]; //away team index for prediction
}

parameters {
  real home; //home advantage
  vector[nt] att; //attack ability of each team
  vector[nt] def; //defence ability of each team
  real<lower=0, upper=1> thetaa;
  real<lower=0, upper=1> thetab;
}

transformed parameters {
  vector[ng] theta1; //score probability of home team
  vector[ng] theta2; //score probability of away team
  
  theta1 = exp(home + att[ht] - def[at]);
  theta2 = exp(att[at] - def[ht]);
}

model {
  //priors
  att ~ normal(0, 2);
  def ~ normal(0, 2);
  home ~ normal(0.2,0.001);
  thetaa ~ normal(0.2263158, 0.001);
  thetab ~ normal(0.3447368, 0.001);
  
  //likelihood
  s1 ~ poisson(theta1);
  s2 ~ poisson(theta2);
  
  for (n in 1:ng) {
    if (s1[n] == 0)
      target += log_sum_exp(bernoulli_lpmf(1 | thetaa),
                            bernoulli_lpmf(0 | thetaa)
                              + poisson_lpmf(s1[n] | theta1[n]));
    else
      target += bernoulli_lpmf(0 | thetaa)
                  + poisson_lpmf(s1[n] | theta1[n]);
  }
  
  for (n in 1:ng) {
    if (s2[n] == 0)
      target += log_sum_exp(bernoulli_lpmf(1 | thetab),
                            bernoulli_lpmf(0 | thetab)
                              + poisson_lpmf(s2[n] | theta2[n]));
    else
      target += bernoulli_lpmf(0 | thetab)
                  + poisson_lpmf(s2[n] | theta2[n]);
  }
}

generated quantities {
  //generate predictions
  vector[np] theta1new; //score probability of home team
  vector[np] theta2new; //score probability of away team
  real s1new[np]; //predicted score
  real s2new[np]; //predicted score
  vector[np] log_lik;
  
  theta1new = exp(home + att[htnew] - def[atnew]);
  theta2new = exp(att[atnew] - def[htnew]);
  s1new = poisson_rng(theta1new);
  s2new = poisson_rng(theta2new);
  
  
  
  for (n in 1:np){
    log_lik[n] = poisson_lpmf(s1[n] | theta1[n]) + poisson_lpmf(s2[n] | theta2[n]);
  }
}


  
  


