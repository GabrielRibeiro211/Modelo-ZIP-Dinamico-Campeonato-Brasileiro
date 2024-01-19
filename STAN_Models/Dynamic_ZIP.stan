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
  vector[nt] elenco; // squad value of each team
  int t; // time index
}

parameters {
  vector[t] home; //home advantage
  matrix[nt,t] att; //attack ability of each team
  matrix[nt,t] def; //defence ability of each team
  real<lower=0, upper=1> thetaA; //probability of 0 goals for home team
  real<lower=0, upper=1> thetaB; //probability of 0 goals for away team
  real v;
}

transformed parameters {
  vector[ng] theta1; //score probability of home team
  vector[ng] theta2; //score probability of away team
  
  theta1 = exp(home[1] + att[ht,1] - def[at,1] + elenco[ht]/1000);
  theta2 = exp(att[at,1] - def[ht,1] + elenco[at]/1000);
}


model {
  //priors
  att[,1] ~ normal(0, 0.5); 
  def[,1] ~ normal(0, 0.5); 
  home[1] ~ normal(0.3, 0.15); 
  thetaA ~ normal(0.2263158, 0.001);
  thetaB ~ normal(0.3447368, 0.001);

  
  for (i in 2:t) {
    home[i] ~ normal(home[i-1], 0.1); 
    att[,i] ~ normal(att[,i-1], 0.1); 
    def[,i] ~ normal(def[,i-1], 0.1); 
  }
  
  //likelihood
  s1 ~ poisson(theta1);
  s2 ~ poisson(theta2);
  
  
  for (n in 1:ng) {
    if (s1[n] == 0)
      target += log_sum_exp(bernoulli_lpmf(1 | thetaA),
                            bernoulli_lpmf(0 | thetaA)
                              + poisson_lpmf(s1[n] | theta1[n]));
    else
      target += bernoulli_lpmf(0 | thetaA)
                  + poisson_lpmf(s1[n] | theta1[n]);
  }
  
  for (n in 1:ng) {
    if (s2[n] == 0)
      target += log_sum_exp(bernoulli_lpmf(1 | thetaB),
                            bernoulli_lpmf(0 | thetaB)
                              + poisson_lpmf(s2[n] | theta2[n]));
    else
      target += bernoulli_lpmf(0 | thetaB)
                  + poisson_lpmf(s2[n] | theta2[n]);
  }
  
  
  
}
  
  
generated quantities {
  //generate predictions
  matrix[np,t] theta1new; //score probability of home team
  matrix[np,t] theta2new; //score probability of away team
  matrix[np,t] s1new; //predicted score
  matrix[np,t] s2new; //predicted score
  vector[np] log_lik;
  
  for (i in 1:t){
    for (n in 1:np){
      theta1new[n,i] = exp(home[i] + att[htnew[n],i] - def[atnew[n],i] + elenco[htnew[n]]/1000);
      theta2new[n,i] = exp(att[atnew[n],i] - def[htnew[n],i] + elenco[atnew[n]]/1000);
    }
  }
  

  for (i in 1:t) {
    for (n in 1:np){
      s1new[n,i] = poisson_rng(theta1new[n,i]);
      s2new[n,i] = poisson_rng(theta2new[n,i]);
    }
  }
  
  
  
  for (n in 1:np){
    log_lik[n] = poisson_lpmf(s1[n] | theta1[n]) + poisson_lpmf(s2[n] | theta2[n]);
  }
}
