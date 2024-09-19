import os
from dotenv import load_dotenv

import librosa
import numpy as np
from tqdm import tqdm

load_dotenv()

data_path = os.getenv("DATA_PATH")
clean_train_folder = os.path.join(data_path, os.getenv("CLEAN_TRAIN_FOLDER"))
noisy_train_folder = os.path.join(data_path, os.getenv("NOISY_TRAIN_FOLDER"))
clean_test_folder = os.path.join(data_path, os.getenv("CLEAN_TEST_FOLDER"))
noisy_test_folder = os.path.join(data_path, os.getenv("NOISY_TEST_FOLDER"))
serialized_train_folder = os.path.join(data_path, os.getenv("SERIALIZED_TRAIN_FOLDER"))
serialized_test_folder = os.path.join(data_path, os.getenv("SERIALIZED_TEST_FOLDER"))
window_size = 2**14  # about 1 second of samples
sample_rate = 16000

if os.getenv("FILES_NOT_ALIGNED") == "True":
    print("Files are not aligned. Aligning signals.")


def slice_signal(file, window_size, stride, sample_rate):
    """
    Helper function for slicing the audio file
    by window size and sample rate with [1-stride] percent overlap (default 50%).
    """
    #wav, sr = librosa.load(file, sr=sample_rate)
    hop = int(window_size * stride)
    slices = []
    for end_idx in range(window_size, len(file), hop):
        start_idx = end_idx - window_size
        slice_sig = file[start_idx:end_idx]
        slices.append(slice_sig)
    return slices


def process_and_serialize(data_type):
    """
    Serialize, down-sample the sliced signals and save on separate folder.
    """
    stride = 0.5

    if data_type == "train":
        clean_folder = clean_train_folder
        noisy_folder = noisy_train_folder
        serialized_folder = serialized_train_folder
    else:
        clean_folder = clean_test_folder
        noisy_folder = noisy_test_folder
        serialized_folder = serialized_test_folder
    if not os.path.exists(serialized_folder):
        os.makedirs(serialized_folder)

    clean_filenames = os.listdir(clean_folder)
    noisy_filenames = os.listdir(noisy_folder)
    clean_filenames.sort()
    noisy_filenames.sort()

    # walk through the path, slice the audio file, and save the serialized result
    for clean_file, noisy_file in zip(clean_filenames, noisy_filenames):
            clean_path = os.path.join(clean_folder, clean_file)
            noisy_path = os.path.join(noisy_folder, noisy_file)

            clean_data, _ = librosa.load(clean_path, sr=sample_rate)
            noisy_data, _ = librosa.load(noisy_path, sr=sample_rate)

            # align clean and noisy files if necessary
            if os.getenv("FILES_NOT_ALIGNED") == "True":
                clean_data, noisy_data = align_signals(clean_data, noisy_data)
                print(f"Aligned files {clean_file} and {noisy_file}")

            # slice both clean signal and noisy signal
            clean_sliced = slice_signal(clean_data, window_size, stride, sample_rate)
            noisy_sliced = slice_signal(noisy_data, window_size, stride, sample_rate)
            # serialize - file format goes [original_file]_[slice_number].npy
            # ex) p293_154.wav_5.npy denotes 5th slice of p293_154.wav file
            for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
                pair = np.array([slice_tuple[0], slice_tuple[1]])
                np.save(
                    os.path.join(serialized_folder, "{}_{}".format(noisy_file, idx)),
                    arr=pair,
                )


def data_verify(data_type):
    """
    Verifies the length of each data after pre-process.
    """
    if data_type == "train":
        serialized_folder = serialized_train_folder
    else:
        serialized_folder = serialized_test_folder

    for root, dirs, files in os.walk(serialized_folder):
        for filename in tqdm(
            files, desc="Verify serialized {} audios".format(data_type)
        ):
            data_pair = np.load(os.path.join(root, filename))
            if data_pair.shape[1] != window_size:
                print(
                    "Snippet length not {} : {} instead".format(
                        window_size, data_pair.shape[1]
                    )
                )
                break

def get_lag(clean_signal: np.ndarray, recorded_signal: np.ndarray):
    cross_correlation = np.correlate(clean_signal, recorded_signal, mode='full')

    return np.argmax(cross_correlation) - clean_signal.shape[0]


def align_signals(clean_signal: np.ndarray, recorded_signal: np.ndarray):
    lag = get_lag(clean_signal, recorded_signal)
    assert lag <= 0, "Recorded signal is ahead of clean signal"

    # Pad clean signal in the beginning and cut the end to align them
    clean_signal = np.pad(clean_signal, (-lag, 0))
    clean_signal = clean_signal[:recorded_signal.shape[0]]

    return clean_signal, recorded_signal

if __name__ == "__main__":
    process_and_serialize("train")
    data_verify("train")
    process_and_serialize("test")
    data_verify("test")
