setwd("!GitHub/PRDL_Group25/results")
dat = read.csv("val_set_results_e.csv", sep=";")

dat[9,'learning_rate']
plot(dat, x=dat$learning_rate, y=dat$acc)
