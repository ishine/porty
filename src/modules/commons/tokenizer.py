import re
import torch
import numpy as np
from nnmnkwii.io import hts
from ttslearn.tacotron.frontend.openjtalk import numeric_feature_by_regex
from ttslearn.tacotron.frontend.openjtalk import extra_symbols, phonemes


class Tokenizer:
    def __init__(self):
        self.phoneme_dict = {s: i for i, s in enumerate(['<pad>'] + phonemes)}
        self.accent_dict = {s: i for i, s in enumerate(['<pad>'] + extra_symbols)}

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def tokenize(self, inputs):
        phoneme, accent = inputs
        phoneme = [self.phoneme_dict[s] for s in phoneme]
        accent = [self.accent_dict[s] for s in accent]
        return torch.LongTensor(phoneme), torch.LongTensor(accent)

    def __len__(self):
        return len(self.phoneme_dict) + len(self.accent_dict)

    def extract(self, label_path, sr, y_length):
        label = hts.load(label_path)
        phoneme, prosody = self.pp_symbols(label.contexts)
        assert len(phoneme) == len(prosody)

        duration = self.extract_duration(label, sr, y_length)
        return (phoneme, prosody), duration

    def extract_duration(self, label, sr, y_length):
        duration = list()
        for b, e, _ in label[1:-1]:
            d = (e - b) * 1e-7 * sr / 256
            duration += [d]
        duration = self.refine_duration(duration, y_length)
        return duration

    @staticmethod
    def refine_duration(duration, y_length):
        duration = np.array(duration)
        duration_floor = np.floor(duration)
        diff_rest = y_length - np.sum(duration_floor)
        indices = np.argsort(np.abs(duration - duration_floor))
        for idx in indices:
            duration_floor[idx] += 1
            diff_rest -= 1
            if diff_rest == 0:
                break
        return duration_floor

    @staticmethod
    def pp_symbols(labels, drop_unvoiced_vowels=True):
        phoneme, accent = list(), list()
        N = len(labels)
        for n in range(N):
            lab_curr = labels[n]

            p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)

            if drop_unvoiced_vowels and p3 in "AEIOU":
                p3 = p3.lower()

            if p3 == "sil":
                continue
            else:
                phoneme.append(p3)

            a1 = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
            a2 = numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
            a3 = numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

            f1 = numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

            a2_next = numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])

            if a3 == 1 and a2_next == 1:
                accent.append("#")
            elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
                accent.append("]")
            elif a2 == 1 and a2_next == 2:
                accent.append("[")
            else:
                accent.append('_')

        return phoneme, accent
