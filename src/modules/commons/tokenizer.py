import torch
import numpy as np
from nnmnkwii.io import hts
from ttslearn.tacotron.frontend.openjtalk import pp_symbols
from ttslearn.tacotron.frontend.openjtalk import extra_symbols, num_vocab, _symbol_to_id


class Tokenizer:
    def __init__(self):
        self.extra_symbol_set = set(extra_symbols)

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def tokenize(self, text):
        inp = list()
        is_extra = list()
        for s in text:
            inp.append(_symbol_to_id[s])
            if s in self.extra_symbol_set:
                is_extra.append(1)
            else:
                is_extra.append(0)
        return torch.LongTensor(inp), torch.FloatTensor(is_extra)

    def __len__(self):
        return num_vocab()

    def extract(self, label_path, sr, y_length):
        label = hts.load(label_path)
        phoneme = pp_symbols(label.contexts)

        duration = self.extract_duration(label, sr, y_length)
        final_duration = list()
        i = 0
        for p in phoneme:
            if p != '_' and p in self.extra_symbol_set:
                final_duration.append(0)
            else:
                final_duration.append(duration[i])
                i += 1
        assert len(phoneme) == len(final_duration)
        return phoneme, final_duration

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
