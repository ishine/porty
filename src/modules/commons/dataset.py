from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .tokenizer import Tokenizer


class AudioDataset(Dataset):
    def __init__(self, params):
        super().__init__()
        self.data = list(sorted(Path(params.data_dir).glob('data_*.pt')))
        self.tokenizer = Tokenizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (
            wav,
            spec,
            mel,
            inputs,
            duration,
            *_
        ) = torch.load(self.data[idx])
        phoneme, is_accent = self.tokenizer(inputs)
        return (
            phoneme, is_accent,
            wav.squeeze(),
            spec.transpose(-1, -2),
            mel.transpose(-1, -2),
            duration.float().transpose(-1, -2),
        )


def collate_fn(batch):
    (
        phoneme, is_accent,
        wav,
        spec,
        mel,
        duration
    ) = tuple(zip(*batch))

    x_length = torch.LongTensor([len(x) for x in phoneme])
    phoneme = pad_sequence(phoneme, batch_first=True)
    is_accent = pad_sequence(is_accent, batch_first=True)

    y_length = torch.LongTensor([x.size(0) for x in mel])
    spec = pad_sequence(spec, batch_first=True).transpose(-1, -2)
    mel = pad_sequence(mel, batch_first=True).transpose(-1, -2)
    duration = pad_sequence(duration, batch_first=True).transpose(-1, -2)

    wav = pad_sequence(wav, batch_first=True).unsqueeze(1)

    return (
        phoneme, is_accent,
        x_length,
        wav,
        spec,
        mel,
        y_length,
        duration
    )
