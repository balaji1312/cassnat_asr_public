

import soundfile as sf

dir = ['dev_clean', 'dev_other','test_clean', 'test_other', 'train_clean_100']

for d in dir:

    wav_path = '/data/balaji/workdir/cassnat_asr/egs/librispeech/data/' + d + '/wav.scp'

    new_wav = '/data/balaji/workdir/cassnat_asr/egs/librispeech/data/' + d + '/wav_s.scp'


    with open(wav_path, 'r') as fin:
        for line in fin:
            cont = line.strip().split(' ')

            path = cont[-2]

            audio,_ = sf.read(path)

            cont.append(str(len(audio)))

            line_n  = ' '.join(cont)

            line_n = line_n + "\n"

            with open(new_wav, 'a') as gin:
                gin.write(line_n)
