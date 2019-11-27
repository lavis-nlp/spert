#!/usr/bin/env bash
curr_dir=$(pwd)

mkdir -p data
mkdir -p data/models

wget -r -nH --cut-dirs=100 --reject "index.html*" --no-parent http://lavis.cs.hs-rm.de/storage/spert/public/models/conll04/ -P ${curr_dir}/data/models/conll04
wget -r -nH --cut-dirs=100 --reject "index.html*" --no-parent http://lavis.cs.hs-rm.de/storage/spert/public/models/scierc/ -P ${curr_dir}/data/models/scierc
wget -r -nH --cut-dirs=100 --reject "index.html*" --no-parent http://lavis.cs.hs-rm.de/storage/spert/public/models/ade/ -P ${curr_dir}/data/models/ade