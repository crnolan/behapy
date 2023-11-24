suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(parallel))
suppressPackageStartupMessages(library(cAIC4))
suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(MASS))
suppressPackageStartupMessages(library(lsei))
suppressPackageStartupMessages(library(refund))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(mvtnorm))
suppressPackageStartupMessages(library(arrangements))
suppressPackageStartupMessages(library(progress))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(gridExtra))
suppressPackageStartupMessages(library(Rfast))
suppressPackageStartupMessages(library(fastFMM))

dat <- read.csv("/Users/uqdkilpa/Documents/Code/Clones/fastFMM/vignettes/time_series.csv")

mod <- fui(Y ~ treatment + # main effect of cue
              (treatment | id),  # random slope & intercept
              data = dat,
              parallel = TRUE,
              analytic = FALSE) # bootstrap

mod_qn <- mod$qn
mod_resid <- mod$residuals
mod_bootsamps <- mod$bootstrap_samps
mod_argvals <- mod$argvals
mod_aic <- mod$aic
mod_betahat <- mod$betaHat
mod_betahatvar <- mod$betaHat.var