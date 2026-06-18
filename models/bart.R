#!/usr/bin/env Rscript
# BART (binary/probit) via dbarts. Posterior predictive gives a calibrated probability
# plus a credible interval per site -- the decision-support output.
#
# BART self-regularizes via its priors, so we don't tune it (only LR lambda is tuned).
# Defaults: ntree=200, k=2, with posterior draws for the credible interval.
#
# Usage:
#   Rscript models/bart.R <train.csv> <test.csv> <out.csv> [ndpost] [nskip] [ntree] [k]
# train.csv: feature columns + a `label` column (0/1).
# test.csv : the same feature columns (no label needed).
# out.csv  : test rows with prob_mean, prob_lo (2.5%), prob_hi (97.5%), prob_sd.
# Variable inclusion counts are written next to out.csv as <out>.varcount.csv.

suppressMessages(library(dbarts))

args <- commandArgs(trailingOnly = TRUE)
train_path <- args[1]
test_path  <- args[2]
out_path   <- args[3]
ndpost <- ifelse(length(args) >= 4, as.integer(args[4]), 1000L)
nskip  <- ifelse(length(args) >= 5, as.integer(args[5]), 1000L)
ntree  <- ifelse(length(args) >= 6, as.integer(args[6]), 200L)
kparam <- ifelse(length(args) >= 7, as.numeric(args[7]), 2.0)

train <- read.csv(train_path, check.names = FALSE)
test  <- read.csv(test_path,  check.names = FALSE)

stopifnot("label" %in% colnames(train))
y <- train$label
feat_cols <- setdiff(colnames(train), "label")
Xtr <- as.matrix(train[, feat_cols, drop = FALSE])
Xte <- as.matrix(test[,  feat_cols, drop = FALSE])

set.seed(0)
# binary BART (probit); dbarts uses probit when y is 0/1
fit <- bart(x.train = Xtr, y.train = y, x.test = Xte,
            ntree = ntree, k = kparam,
            ndpost = ndpost, nskip = nskip,
            keeptrees = FALSE, verbose = FALSE)

# fit$yhat.test is the posterior matrix (ndpost x n_test). For binary BART dbarts returns
# the probit-scale latent f(x); convert to probability with pnorm. Guard in case a build
# already returns probabilities (values in [0,1]).
post <- fit$yhat.test
if (max(post) > 1 || min(post) < 0) {
  post <- pnorm(post)
}
prob_mean <- apply(post, 2, mean)
prob_lo   <- apply(post, 2, quantile, probs = 0.025)
prob_hi   <- apply(post, 2, quantile, probs = 0.975)
prob_sd   <- apply(post, 2, sd)

out <- data.frame(prob_mean = prob_mean, prob_lo = prob_lo,
                  prob_hi = prob_hi, prob_sd = prob_sd)
write.csv(out, out_path, row.names = FALSE)

# variable inclusion (proportion of splitting rules using each feature) for axis importance
vc <- fit$varcount                    # ndpost x p
vc_mean <- colMeans(vc)
vc_prop <- vc_mean / sum(vc_mean)
vdf <- data.frame(feature = feat_cols, inclusion = as.numeric(vc_prop))
write.csv(vdf, paste0(out_path, ".varcount.csv"), row.names = FALSE)
