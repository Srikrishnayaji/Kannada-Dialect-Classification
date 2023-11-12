#!/usr/bin/env python

import os
import librosa
import soundfile as sf

from argparse import ArgumentParser
from logging import error


SAMPLE_RATE = 60000


def get_parser():
    parser = ArgumentParser(description="A parser to parse required "
                                        "script inputs.")
    
    # Expected Structure inside root:
    # root
    #  |---D1 (class 1)
    #  |---D2 (class 2)
    #  |---D3 (class 3)
    #  |---D4 (class 4)
    #  |---D5 (class 5)
    parser.add_argument("--root",
                        required=True,
                        type=str,
                        help="Root path from where audio files are to"
                             "loaded.")
    
    parser.add_argument("--dump",
                        required=True,
                        type=str,
                        help="Dump path to which the data should be saved.")
    
    return parser


def make_dirs_for_5_second_samples():
    try:
        os.mkdir('Cut_Kannada_Dataset')
        os.mkdir('Cut_Kannada_Dataset/D1')
        os.mkdir('Cut_Kannada_Dataset/D2')
        os.mkdir('Cut_Kannada_Dataset/D3')
        os.mkdir('Cut_Kannada_Dataset/D4')
        os.mkdir('Cut_Kannada_Dataset/D5')
    except Exception as err:
        error(f"Error in creating directory: {err}")


def split_audio(path, dump_path, audio_idx, dirname):
    # D2 class has audio files which are quite large. To prevent
    # class imbalance problem create audio samples of 8 seconds
    # in case of D2 class.
    if dirname == 'D2':
        time = 8
    else:
        time = 5
    signal, sr = librosa.load(path, sr=SAMPLE_RATE)
    
    initial = 0
    final   = time * sr
    counter = 1
    
    while final < signal.size:
        cut_signal = signal[initial: final]
        print(initial)
        print(final)
        
        sf.write(f"{dump_path}/{audio_idx}{counter}.wav",
                 data=cut_signal,
                 samplerate=sr)
        
        initial = final
        final  = initial + time * sr
    
        counter+=1
    
    sf.write(f"{dump_path}/{audio_idx}{counter}.wav",
             data=signal[initial:len(signal)],
             samplerate=sr)


def main():
    """Main function of the script."""
    args     = get_parser().parse_args()
    dirnames = os.listdir(args.root)
    
    for dirname in dirnames:
        audio_files = sorted(
            os.listdir(os.path.join(args.root, dirname)),
                       key=_sort_func)
    
    for idx, audio_file in enumerate(audio_files):
        split_audio(os.path.join(args.root, dirname, audio_file),
                    os.path.join(args.dump, dirname),
                    idx,
                    dirname)


def _sort_func(val):
    split_val = val.split('.')
    return int(split_val[0])


if __name__ == "__main__":
    main()
