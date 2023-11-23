# Load libraries
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

dat <- read.csv("/Users/uqdkilpa/Documents/Code/Clones/fastFMM/vignettes/time_series.csv")
dat


