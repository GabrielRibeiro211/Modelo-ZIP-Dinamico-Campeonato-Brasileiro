library(dplyr)
library(rstan)

# Leitura da base de dados
bra22 <- `BRA.(1)`
bra_22 <- bra22 %>% filter(Season == 2022) # Especificar o campeonato de 2022

ng = nrow(bra_22) # Número de jogos
nt = length(unique(bra_22$Home))

# Converter o nome das equipes para cada jogo em números
teams = unique(bra_22$Away)
ht = unlist(sapply(1:ng, function(g) which(teams == bra_22$Home[g])))
at = unlist(sapply(1:ng, function(g) which(teams == bra_22$Away[g])))

np=80 # Separar as últimas 80 partidas para previsão
ngob = ng-np # Número de jogos para ajustar

# Valor de elenco das equipes em 2022
elenco = c(65.15, 156.35, 15.50, 15.83, 78.70, 85.75, 18.75, 18.95, 67.20, 
           81.23, 161.10, 15.25, 14.23, 58.60, 19.13, 76.70, 113.25, 19.75,
           30.30, 46.13)

# Aplicação do modelo no Stan

# Data Wrangling 
my_data = list(
  nt = nt,
  ng = ngob ,
  ht = ht[1:ngob],
  at = at[1:ngob],
  s1 = bra_22$HG[1:ngob],
  s2 = bra_22$AG[1:ngob],
  np = np,
  htnew = ht[(ngob+1):ng],
  atnew = at[(ngob+1):ng],
  elenco = elenco,
  t = 8
)

brnhpoolfit22zipdin = stan(file = "ZIPdin.stan", 
                           data = my_data, iter = 2000, chains = 4, seed = 123456)

brnhpoolparamsDINAM = extract(brnhpoolfit22zipdin)

# Traceplot para os parâmetros

traceplot(brnhpoolfit22zipdin, pars = c("att[11,8]"))

# Checagem de convergência

rstan::check_divergences(brnhpoolfit22zipdin)

# Boxplots para os parâmetros de ataque e defesa

boxplot(brnhpoolparamsDINAM$att[,,1], boxfill = NA, border = NA, ylab="VALOR DO PARÂMETRO",xlab="TIME",xaxt = "n")
boxplot(brnhpoolparamsDINAM$att[,,1],xaxt = "n",add = TRUE, boxfill=rgb(0.8,0.1,0.3,0.6), boxwex=0.5, at = 1:20 - 0.27)
boxplot(brnhpoolparamsDINAM$def[,,1],xaxt = "n",add = TRUE, boxfill=rgb(0.1,0.1,0.7,0.5), boxwex=0.5, at = 1:20 + 0.27)
legend("bottomleft",c("DEF","ATT"),col=c(rgb(0.1,0.1,0.7,0.5),rgb(0.8,0.1,0.3,0.6)),pch=15)
axis(1, at=1:20, label= c('SAN','FLA','CEA','GOI','INT','COR','CUI','AME','CAP','BRA','PAL','JUV','AVA','FLU','CTB','SAO','CAM','ACG','FOR','BOT'))


# Relação entre os parâmetros de ataque e defesa

library(matrixStats)

attack = colMedians(brnhpoolparamsDINAM$att[,,1])
defense = colMedians(brnhpoolparamsDINAM$def[,,1])

plot(attack ,defense ,xlim=c( -0.5 ,0.5),ylim=c( -0.5,0.5))
abline(h=0)
abline(v=0)
text(attack ,defense , labels=teams , cex=0.7 , pos=4)


################################################################################################
########################## PREVISÕES PARA AS PARTIDAS ##########################################
################################################################################################

m <- rep(0,80)
e <- rep(0,80)
v <- rep(0,80)

for (i in 1:10) {
  for (j in 1:4000) {
    if(brnhpoolparamsDINAM$s1new[j,i,1]>brnhpoolparamsDINAM$s2new[j,i,1]){m[i]=m[i]+1}
    else if(brnhpoolparamsDINAM$s1new[j,i,1]==brnhpoolparamsDINAM$s2new[j,i,1]){e[i]=e[i]+1}
    else if(brnhpoolparamsDINAM$s1new[j,i,1]<brnhpoolparamsDINAM$s2new[j,i,1]){v[i]=v[i]+1}
  }
  m[i]=m[i]/4000 #prob. mandante
  e[i]=e[i]/4000 #prob. empate
  v[i]=v[i]/4000 #prob. visitante
}

for (i in 11:20) {
  for (j in 1:4000) {
    if(brnhpoolparamsDINAM$s1new[j,i,2]>brnhpoolparamsDINAM$s2new[j,i,2]){m[i]=m[i]+1}
    else if(brnhpoolparamsDINAM$s1new[j,i,2]==brnhpoolparamsDINAM$s2new[j,i,2]){e[i]=e[i]+1}
    else if(brnhpoolparamsDINAM$s1new[j,i,2]<brnhpoolparamsDINAM$s2new[j,i,2]){v[i]=v[i]+1}
  }
  m[i]=m[i]/4000 #prob. mandante
  e[i]=e[i]/4000 #prob. empate
  v[i]=v[i]/4000 #prob. visitante
}

for (i in 21:30) {
  for (j in 1:4000) {
    if(brnhpoolparamsDINAM$s1new[j,i,3]>brnhpoolparamsDINAM$s2new[j,i,3]){m[i]=m[i]+1}
    else if(brnhpoolparamsDINAM$s1new[j,i,3]==brnhpoolparamsDINAM$s2new[j,i,3]){e[i]=e[i]+1}
    else if(brnhpoolparamsDINAM$s1new[j,i,3]<brnhpoolparamsDINAM$s2new[j,i,3]){v[i]=v[i]+1}
  }
  m[i]=m[i]/4000 #prob. mandante
  e[i]=e[i]/4000 #prob. empate
  v[i]=v[i]/4000 #prob. visitante
}

for (i in 31:40) {
  for (j in 1:4000) {
    if(brnhpoolparamsDINAM$s1new[j,i,4]>brnhpoolparamsDINAM$s2new[j,i,4]){m[i]=m[i]+1}
    else if(brnhpoolparamsDINAM$s1new[j,i,4]==brnhpoolparamsDINAM$s2new[j,i,4]){e[i]=e[i]+1}
    else if(brnhpoolparamsDINAM$s1new[j,i,4]<brnhpoolparamsDINAM$s2new[j,i,4]){v[i]=v[i]+1}
  }
  m[i]=m[i]/4000 #prob. mandante
  e[i]=e[i]/4000 #prob. empate
  v[i]=v[i]/4000 #prob. visitante
}

for (i in 41:50) {
  for (j in 1:4000) {
    if(brnhpoolparamsDINAM$s1new[j,i,5]>brnhpoolparamsDINAM$s2new[j,i,5]){m[i]=m[i]+1}
    else if(brnhpoolparamsDINAM$s1new[j,i,5]==brnhpoolparamsDINAM$s2new[j,i,5]){e[i]=e[i]+1}
    else if(brnhpoolparamsDINAM$s1new[j,i,5]<brnhpoolparamsDINAM$s2new[j,i,5]){v[i]=v[i]+1}
  }
  m[i]=m[i]/4000 #prob. mandante
  e[i]=e[i]/4000 #prob. empate
  v[i]=v[i]/4000 #prob. visitante
}

for (i in 51:60) {
  for (j in 1:4000) {
    if(brnhpoolparamsDINAM$s1new[j,i,6]>brnhpoolparamsDINAM$s2new[j,i,6]){m[i]=m[i]+1}
    else if(brnhpoolparamsDINAM$s1new[j,i,6]==brnhpoolparamsDINAM$s2new[j,i,6]){e[i]=e[i]+1}
    else if(brnhpoolparamsDINAM$s1new[j,i,6]<brnhpoolparamsDINAM$s2new[j,i,6]){v[i]=v[i]+1}
  }
  m[i]=m[i]/4000 #prob. mandante
  e[i]=e[i]/4000 #prob. empate
  v[i]=v[i]/4000 #prob. visitante
}

for (i in 61:70) {
  for (j in 1:4000) {
    if(brnhpoolparamsDINAM$s1new[j,i,7]>brnhpoolparamsDINAM$s2new[j,i,7]){m[i]=m[i]+1}
    else if(brnhpoolparamsDINAM$s1new[j,i,7]==brnhpoolparamsDINAM$s2new[j,i,7]){e[i]=e[i]+1}
    else if(brnhpoolparamsDINAM$s1new[j,i,7]<brnhpoolparamsDINAM$s2new[j,i,7]){v[i]=v[i]+1}
  }
  m[i]=m[i]/4000 #prob. mandante
  e[i]=e[i]/4000 #prob. empate
  v[i]=v[i]/4000 #prob. visitante
}

for (i in 71:80) {
  for (j in 1:4000) {
    if(brnhpoolparamsDINAM$s1new[j,i,8]>brnhpoolparamsDINAM$s2new[j,i,8]){m[i]=m[i]+1}
    else if(brnhpoolparamsDINAM$s1new[j,i,8]==brnhpoolparamsDINAM$s2new[j,i,8]){e[i]=e[i]+1}
    else if(brnhpoolparamsDINAM$s1new[j,i,8]<brnhpoolparamsDINAM$s2new[j,i,8]){v[i]=v[i]+1}
  }
  m[i]=m[i]/4000 #prob. mandante
  e[i]=e[i]/4000 #prob. empate
  v[i]=v[i]/4000 #prob. visitante
}


### Medida de DeFinetti ###
r = NULL
for (i in 301:380) {
  if(bra_22$HG[i]>bra_22$AG[i]){r[i-300]='m'}
  else if(bra_22$HG[i]==bra_22$AG[i]){r[i-300]='e'}
  else if(bra_22$HG[i]<bra_22$AG[i]){r[i-300]='v'}
}
r # Resultado real das partidas


acertos = 0
for (i in 1:80) {
  if(r[i]=='m' && m[i]>e[i] && m[i]>v[i]){
    acertos = acertos + 1
  }
  else if(r[i]=='e' && e[i]>m[i] && e[i]>v[i]){acertos = acertos + 1}
  else if(r[i]=='v' && v[i]>max(e[i],m[i])){acertos = acertos + 1}
}
acertos # Número de acertos do modelo


f = rep(0,80)
for (i in 1:80) {
  if(r[i]=='m'){f[i]=((m[i]-1)^2+(e[i]-0)^2+(v[i]-0)^2)}
  else if(r[i]=='e'){f[i]=((m[i]-0)^2+(e[i]-1)^2+(v[i]-0)^2)}
  else if(r[i]=='v'){f[i]=((m[i]-0)^2+(e[i]-0)^2+(v[i]-1)^2)}
}
f
mean(f) # Medida de DeFinetti média


# Para uma melhor análise das propriedades de convergência e comportamento dos parâmetros do modelo

library(shinystan)
launch_shinystan(brnhpoolfit22zipdin)