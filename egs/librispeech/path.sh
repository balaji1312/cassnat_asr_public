export KALDI_ROOT=/data/balaji/workdir/kaldi/
export E2EASR=`pwd`/../..
export PATH=$E2EASR/src/bin/:$E2EASR/utils/:$PATH

# Original kaldi setup
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
export PATH=$KALDI_ROOT/tools/sctk/bin:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

### Below are the paths used by the optional parts of the recipe

# We only need the Festival stuff below for the optional text normalization(for LM-training) step
FEST_ROOT=tools/festival
NSW_PATH=${FEST_ROOT}/festival/bin:${FEST_ROOT}/nsw/bin
export PATH=$PATH:$NSW_PATH

# Sequitur G2P executable
sequitur=$KALDI_ROOT/tools/sequitur/g2p.py
sequitur_path="$(dirname $sequitur)/lib/$PYTHON/site-packages"

# Directory under which the LM training corpus should be extracted
LM_CORPUS_ROOT=./lm-corpus
