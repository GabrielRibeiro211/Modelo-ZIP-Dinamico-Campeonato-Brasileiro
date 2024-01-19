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
  matrix[nt,t] home; //home advantage
  matrix[nt,t] att; //attack ability of each team
  matrix[nt,t] def; //defence ability of each team
  real<lower=0> Watt;
  real<lower=0> Wdef;
  real<lower=0> Whome;
}


transformed parameters {
  vector[ng] theta1; //score probability of home team
  vector[ng] theta2; //score probability of away team
  
  theta1 = exp(home[ht,1] + att[ht,1] - def[at,1] + (elenco[ht]/1000));
  theta2 = exp(att[at,1] - def[ht,1] + (elenco[at]/1000));
  
}

model {
  //priors
  
  att[1,1] ~ normal(-0.1240216,0.03); // Santos 
  att[2,1] ~ normal(0.583343,0.03); // Flamengo
  att[3,1] ~ normal(0.01565242,0.03); // Ceara
  att[4,1] ~ normal(0,0.5); // Goias
  att[5,1] ~ normal(0.1470092,0.03); // Internacional
  att[6,1] ~ normal(0.04203038,0.03); // Corinthians
  att[7,1] ~ normal(-0.1577586,0.03); // Cuiaba
  att[8,1] ~ normal(0.01171879,0.03); // America MG
  att[9,1] ~ normal(0.04454694,0.03); // Athletico PR
  att[10,1] ~ normal(0.3547752,0.03); // Bragantino
  att[11,1] ~ normal(0.4017413,0.03); // Palmeiras
  att[12,1] ~ normal(-0.09024919,0.03); // Juventude
  att[13,1] ~ normal(0,0.5); // Avai
  att[14,1] ~ normal(-0.07614485,0.03); // Fluminense
  att[15,1] ~ normal(0,0.5); // Coritiba
  att[16,1] ~ normal(-0.219788,0.03); // Sao Paulo
  att[17,1] ~ normal(0.51187,0.03); // Atletico MG
  att[18,1] ~ normal(-0.2184971,0.03); // Atletico GO
  att[19,1] ~ normal(0.1036299,0.03); // Fortaleza
  att[20,1] ~ normal(0,0.5); // Botafogo

  
  def[1,1] ~ normal(0.05302772,0.03); // Santos 
  def[2,1] ~ normal(0.1463656,0.03); // Flamengo
  def[3,1] ~ normal(0.08494726,0.03); // Ceara
  def[4,1] ~ normal(0,0.5); // Goias
  def[5,1] ~ normal(-0.02572887,0.03); // Internacional
  def[6,1] ~ normal(0.1471188,0.03); // Corinthians
  def[7,1] ~ normal(0.1343564,0.03); // Cuiaba
  def[8,1] ~ normal(0.1015079,0.03); // America MG
  def[9,1] ~ normal(-0.0665464,0.03); // Athletico PR
  def[10,1] ~ normal(-0.1446179,0.03); // Bragantino
  def[11,1] ~ normal(-0.07936131,0.03); // Palmeiras
  def[12,1] ~ normal(-0.07446501,0.03); // Juventude
  def[13,1] ~ normal(0,0.5); // Avai
  def[14,1] ~ normal(0.08395092,0.03); // Fluminense
  def[15,1] ~ normal(0,0.5); // Coritiba
  def[16,1] ~ normal(0.1059268,0.03); // Sao Paulo
  def[17,1] ~ normal(0.2689039,0.03); // Atletico MG
  def[18,1] ~ normal(0.1148711,0.03); // Atletico GO
  def[19,1] ~ normal(-0.08367916,0.03); // Fortaleza
  def[20,1] ~ normal(0,0.5); // Botafogo
  
  home[1,1] ~ normal(0.2,0.01);
  home[2,1] ~ normal(0.3,0.01);
  home[3,1] ~ normal(0.1,0.01);
  home[4,1] ~ normal(0.1,0.01);
  home[5,1] ~ normal(0.2,0.01);
  home[6,1] ~ normal(0.3,0.01);
  home[7,1] ~ normal(0.1,0.01);
  home[8,1] ~ normal(0.2,0.01);
  home[9,1] ~ normal(0.3,0.01);
  home[10,1] ~ normal(0.1,0.01);
  home[11,1] ~ normal(0.3,0.01);
  home[12,1] ~ normal(0.1,0.01);
  home[13,1] ~ normal(0.1,0.01);
  home[14,1] ~ normal(0.2,0.01);
  home[15,1] ~ normal(0.1,0.01);
  home[16,1] ~ normal(0.2,0.01);
  home[17,1] ~ normal(0.3,0.01);
  home[18,1] ~ normal(0.1,0.01);
  home[19,1] ~ normal(0.2,0.01);
  home[20,1] ~ normal(0.2,0.01);
  
  
  for (i in 2:t) {
      att[,i] ~ normal(att[,i-1], 0.025);
      def[,i] ~ normal(def[,i-1], 0.025);
      home[,i] ~ normal(home[,i-1], 0.001);
  }
  
  //likelihood
  s1 ~ poisson(theta1);
  s2 ~ poisson(theta2);
  
  
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
      theta1new[n,i] = exp(home[htnew[n],i] + att[htnew[n],i] - def[atnew[n],i] + (elenco[htnew[n]]/1000));
      theta2new[n,i] = exp(att[atnew[n],i] - def[htnew[n],i] + (elenco[atnew[n]]/1000));
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

