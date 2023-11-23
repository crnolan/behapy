# Load libraries
# The issue here is that the user will need R with these packages installed
# I tried with the vignette example but this resulted in some weird hiragana outputs?
library(lme4)
library(parallel)
library(cAIC4)
library(magrittr)
library(dplyr)
library(mgcv) 
library(MASS)
library(lsei) 
library(refund)
library(stringr) 
library(Matrix) 
library(mvtnorm) 
library(arrangements) 
library(progress) 
library(ggplot2)
library(gridExtra)
library(Rfast)
library(fastFMM)

dat <- read.csv("")
dat

mod <- fui(Y ~ treatment + # main effect of cue
              (treatment | id),  # random slope & intercept
              data = dat,
              parallel = TRUE,
              analytic = FALSE) # bootstrap

align_time <- 1 # cue onset is at 2 seconds
sampling_Hz <- 15 # sampling rate
# plot titles: interpretation of beta coefficients
plot_names <- c("Intercept", "Mean Signal Difference: Cue1 - Cue0") 
fui_plot <- plot_fui(mod, # model fit object
                     x_rescale = sampling_Hz, # rescale x-axis to sampling rate
                     align_x = align_time, # align to cue onset
                     title_names = plot_names,
                     xlab = "Time (s)",
                     num_row = 2)


