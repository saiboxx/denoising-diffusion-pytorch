#! /bin/sh

rsync -auv \
  --exclude *.pt \
  tobweber@srvcorem1.srv.med.uni-muenchen.de:/projects/core-rad/tobweber/ddpm/results_beta_05 \
  .

rsync -auv \
  tobweber@srvcorem1.srv.med.uni-muenchen.de:/projects/core-rad/tobweber/ddpm/logs \
  .