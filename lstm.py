import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split


dirname_to_label = lambda dirname: int(dirname[-1])

SAMPLE_RATE = 60000
N_FFT       = 2048
HOP_LENGTH  = 512
N_MFCC      = 40


def create_audio_file_name_to_label_map(root, dirnames, to_file=False):
    audio_to_label_data = []
    for dirname in dirnames:
        audio_files = os.listdir(os.path.join(root, dirname))
    
    for audio_file in audio_files:
        audio_to_label_data.append(
            (f"{dirname}/{audio_file}", dirname_to_label(dirname))
        )

    df = pd.DataFrame(audio_to_label_data, columns=["audio_file", "label"])
    
    if to_file:
        df.to_csv("audio_to_label.csv", index=False)
    
    return df


def _get_mfcc(root, filename, display=False):
    path       = os.path.join(root, filename)
    signal, sr = librosa.load(path,
                              sr=SAMPLE_RATE)
    
    if display:
        librosa.display.waveshow(signal, sr=SAMPLE_RATE)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()
        
    mfcc = librosa.feature.mfcc(
        y=signal,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mfcc=N_MFCC
    )
    mfcc_scaled_features = np.mean(mfcc.T, axis=0)
    
    if display:
        librosa.display.specshow(mfcc_scaled_features,
                                 sr=SAMPLE_RATE,
                                 hop_length=HOP_LENGTH)
        plt.xlabel("Time")
        plt.ylabel("MFCC")
        plt.colorbar()
        plt.show()
    
    return mfcc_scaled_features


def get_extracted_features():
    df = create_audio_file_name_to_label_map("./Kannada_Dataset/5class",
                                             ["D1", "D2", "D3", "D4", "D5"])
    
    extracted_features = []
    for _, audio_file, label in df.itertuples():
        mfcc = _get_mfcc(audio_file)
        
        extracted_features.append((mfcc, label-1))
    
    return extracted_features


def get_input_data():
    extracted_features = get_extracted_features()
    input_data         = pd.DataFrame(extracted_features,
                                      columns=('feature', 'label'))
    
    # Features
    X = tf.convert_to_tensor(input_data['feature'].tolist())

    # Labels
    Y = np.array(input_data['label'].tolist())
    Y = tf.one_hot(Y,
                   5,
                   on_value=1,
                   off_value=0,
                   axis=-1)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X.numpy(),
                                                        Y.numpy(),
                                                        test_size=0.2,
                                                        random_state=18)
    return (X_train.reshape((*X_train.shape, 1)),
            X_test.reshape((*X_test.shape, 1)),
            Y_train,
            Y_test)


def lstm_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(20,
                                   input_shape=(13, 1),
                                   return_sequences=False))
    model.add(tf.keras.layers.Dense(5, activation='softmax'))
    
    model.summary()
    
    return model


def main():
    """Main function of the script."""
    X_train, X_test, Y_train, Y_test = get_input_data()

    model = lstm_model()
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer='adam')
    
    model.fit(X_train,
              Y_train,
              batch_size=1,
              epochs=50,
              validation_data=(X_test, Y_test))


if __name__ == "__main__":
    main()


