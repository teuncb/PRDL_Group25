# Install and load the necessary packages
library(ggplot2)

# Your data
dat <- read.csv("val_set_results.csv", sep=";")

dat_cross = dat[dat$data == "cross",]
dat_intra = dat[dat$data == "intra",]

#require(scales)
# Plotting
ggplot(dat_intra, aes(x = learning_rate, y = acc, color = factor(timeframe), 
                      shape = factor(lstm_units), linetype = factor(lstm_units))) +
  geom_point() +
  geom_line(aes(group = interaction(timeframe, lstm_units)), size=1) +
  #scale_x_discrete(trans = log2_trans())
  scale_x_discrete(labels=c('1e-2', '1e-3', '1e-4')) +
  labs(title = "Validation Results",
       x = "Learning Rate",
       y = "Accuracy",
       color = "Timeframe",
       linetype = "LSTM units",
       shape = "LSTM units") +
  theme_minimal()
