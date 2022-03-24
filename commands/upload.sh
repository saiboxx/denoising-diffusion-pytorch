#! /bin/sh

rsync -auv \
--exclude .git \
--exclude .venv \
--exclude data \
--exclude .idea \
--exclude results \
. \
tobweber@srvcorem1.srv.med.uni-muenchen.de:/projects/core-rad/tobweber/ddpm
