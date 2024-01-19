

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
}

parameters {
  real home; //home advantage
  vector[nt] att; //attack ability of each team
  vector[nt] def; //defence ability of each team
  
}

transformed parameters {
  vector[ng] theta1; //score probability of home team
  vector[ng] theta2; //score probability of away team
  
  theta1 = exp(home + att[ht] - def[at] + (elenco[ht]/1000));
  theta2 = exp(att[at] - def[ht] + (elenco[at]/1000));
}

model {
  //priors
  
  att[1] ~ normal(-0.1240216,0.1228637); // Santos 
  att[2] ~ normal(0.583343,0.1228637); // Flamengo
  att[3] ~ normal(0.01565242,0.1228637); // Ceara
  att[4] ~ normal(0,2); // Goias
  att[5] ~ normal(0.1470092,0.1228637); // Internacional
  att[6] ~ normal(0.04203038,0.1228637); // Corinthians
  att[7] ~ normal(-0.1577586,0.1228637); // Cuiaba
  att[8] ~ normal(0.01171879,0.1228637); // America MG
  att[9] ~ normal(0.04454694,0.1228637); // Athletico PR
  att[10] ~ normal(0.3547752,0.1228637); // Bragantino
  att[11] ~ normal(0.4017413,0.1228637); // Palmeiras
  att[12] ~ normal(-0.09024919,0.1228637); // Juventude
  att[13] ~ normal(0,2); // Avai
  att[14] ~ normal(-0.07614485,0.1228637); // Fluminense
  att[15] ~ normal(0,2); // Coritiba
  att[16] ~ normal(-0.219788,0.1228637); // Sao Paulo
  att[17] ~ normal(0.51187,0.1228637); // Atletico MG
  att[18] ~ normal(-0.2184971,0.1228637); // Atletico GO
  att[19] ~ normal(0.1036299,0.1228637); // Fortaleza
  att[20] ~ normal(0,2); // Botafogo

  
  def[1] ~ normal(0.05302772,0.12275108); // Santos
  def[2] ~ normal(0.1463656,0.12275108); // Flamengo
  def[3] ~ normal(0.08494726,0.12275108); // Ceara
  def[4] ~ normal(0,2); // Goias
  def[5] ~ normal(-0.02572887,0.12275108); // Internacional
  def[6] ~ normal(0.1471188,0.12275108); // Corinthians
  def[7] ~ normal(0.1343564,0.12275108); // Cuiaba
  def[8] ~ normal(0.1015079,0.12275108); // America MG
  def[9] ~ normal(-0.0665464,0.12275108); // Athletico PR
  def[10] ~ normal(-0.1446179,0.12275108); // Bragantino
  def[11] ~ normal(-0.07936131,0.12275108); // Palmeiras
  def[12] ~ normal(-0.07446501,0.12275108); // Juventude
  def[13] ~ normal(0,2); // Avai
  def[14] ~ normal(0.08395092,0.12275108); // Fluminense
  def[15] ~ normal(0,2); // Coritiba
  def[16] ~ normal(0.1059268,0.12275108); // Sao Paulo
  def[17] ~ normal(0.2689039,0.12275108); // Atletico MG
  def[18] ~ normal(0.1148711,0.12275108); // Atletico GO
  def[19] ~ normal(-0.08367916,0.12275108); // Fortaleza
  def[20] ~ normal(0,2); // Botafogo
  home ~ normal(-1.51675e-07, 9.975853e-05);
  
  //likelihood
  s1 ~ poisson(theta1);
  s2 ~ poisson(theta2);
}

generated quantities {
  //generate predictions
  vector[np] theta1new; //score probability of home team
  vector[np] theta2new; //score probability of away team
  real s1new[np]; //predicted score
  real s2new[np]; //predicted score
  vector[np] log_lik;
  
  theta1new = exp(home + att[htnew] - def[atnew] + (elenco[htnew]/1000));
  theta2new = exp(att[atnew] - def[htnew] + (elenco[atnew]/1000));
  
  s1new = poisson_rng(theta1new);
  s2new = poisson_rng(theta2new);
  
  for (n in 1:np){
    log_lik[n] = poisson_lpmf(s1[n] | theta1[n]) + poisson_lpmf(s2[n] | theta2[n]);
  }
}

