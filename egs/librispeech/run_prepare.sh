#!/usr/bin/env bash

# 2020 (Ruchao Fan)

# The data are already downloaded in the corresponding dir
data=/data/Databases/LibriSpeech/Librispeech

stage=3
end_stage=3
featdir=data/fbank

unit=wp         #word piece
nbpe=1024
bpemode=unigram #bpe or unigram

. ./cmd.sh
. ./path.sh
. parse_options.sh

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  # format the data as Kaldi data directories
  for part in dev-clean test-clean dev-other test-other train-clean-100; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $data/$part data/$(echo $part | sed s/-/_/g)
  done
  echo "[Stage 1] Data Preparation Finished."
fi

train_set="train_clean_100"
test_set="dev_clean test_clean dev_other test_other"
dev_set="dev_clean dev_other"


dict=data/dict/vocab_${unit}.txt ; mkdir -p data/dict
bpemodel=data/dict/bpemodel_${bpemode}_${nbpe}
if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then  
  echo "Create a dictionary..."
  all_text=data/train_text
  
  ( for f in $train_set; do cat data/$f/text; done ) | sort -k1 > $all_text

  if [ $unit == wp ]; then
    cut -f 2- -d " " $all_text > data/dict/input.txt
    spm_train --input=data/dict/input.txt --vocab_size=$nbpe --model_type=$bpemode \
        --model_prefix=$bpemodel --input_sentence_size=100000000

    spm_encode --model=${bpemodel}.model --output_format=piece < data/dict/input.txt | tr ' ' '\n' | \
        sort | uniq | awk '{print $0 }' > $dict
  elif [ $unit == char ]; then
    for part in $test_set; do
        python local/prepare_dict_char.py $dict $all_text data/$part/text > data/$part/token_char.scp
    done
  else
    echo "Not ImplementedError"; exit 1
  fi

  if [ $unit == wp ]; then
    for part in $test_set $train_set; do
      paste -d " " <(awk '{print $1}' data/$part/text) <(cut -f 2- -d" " data/$part/text \
              | spm_encode --model=${bpemodel}.model --output_format=piece) \
              > data/$part/token_wp.scp
    done
  fi
  echo "[Stage 3] Dictionary and Transcription Finished."
fi



