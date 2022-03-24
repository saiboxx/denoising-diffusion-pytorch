#! /bin/sh

rsync -auv \
  tobweber@srvcorem1.srv.med.uni-muenchen.de:/projects/core-rad/tobweber/ddpm/results/*.png \
  results
