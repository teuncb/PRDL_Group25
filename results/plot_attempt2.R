setwd("!GitHub/PRDL_Group25/results")
# Install and load the necessary packages
library(ggplot2)

# Your data
dat <- read.csv("val_set_results.csv", sep=";")

dat_cross = dat[dat$data == "cross",]
dat_intra = dat[dat$data == "intra",]

#require(scales)
# Plotting
cbbPalette <- c("#000000", "#E69F00", "#56B4E9")

ggplot(dat, aes(x = learning_rate, y = acc, color = factor(lstm_units), 
                      shape = factor(timeframe), linetype = factor(timeframe))) +
  geom_point() +
  geom_line(aes(group = interaction(timeframe, lstm_units)), size=1) +
  #scale_x_discrete(trans = log2_trans())
  scale_x_discrete(labels=c('1e-4', '1e-3', '1e-2'), limits=rev) +
  labs(x = "Learning Rate",
       y = "Accuracy",
       color = "LSTM units",
       linetype = "Timeframe",
       shape = "Timeframe") +
  scale_colour_manual(values=cbbPalette) +
  theme_minimal() + 
  facet_grid(.~data)
