from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .tokenizer import Tokenizer


class AudioDataset(Dataset):
    def __init__(self, params):
        super().__init__()
        self.data = list(sorted(Path(params.data_dir).glob('data_*.pt')))
        self.stats = torch.load(f'{params.data_dir}/stat.pt')
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
            pitch,
            energy
        ) = torch.load(self.data[idx])
        label = self.tokenizer(inputs)
        duration = duration.float()
        pitch = (pitch - self.stats['pitch_mean']) / self.stats['pitch_std']
        energy = (energy - self.stats['energy_mean']) / self.stats['energy_std']
        return (
            label,
            wav.squeeze(),
            spec.transpose(-1, -2),
            mel.transpose(-1, -2),
            duration.transpose(-1, -2),
            pitch.transpose(-1, -2),
            energy.transpose(-1, -2)
        )


def collate_fn(batch):
    (
        labels,
        wav,
        spec,
        mel,
        duration,
        pitch,
        energy
    ) = tuple(zip(*batch))

    x_length = torch.LongTensor([len(x) for x in labels])
    x = pad_sequence(labels, batch_first=True)

    y_length = torch.LongTensor([x.size(0) for x in mel])
    spec = pad_sequence(spec, batch_first=True).transpose(-1, -2)
    mel = pad_sequence(mel, batch_first=True).transpose(-1, -2)

    wav = pad_sequence(wav, batch_first=True).unsqueeze(1)

    pitch = pad_sequence(pitch, batch_first=True).transpose(-1, -2)
    energy = pad_sequence(energy, batch_first=True).transpose(-1, -2)
    duration = pad_sequence(duration, batch_first=True).transpose(-1, -2)

    return (
        x,
        x_length,
        wav,
        spec,
        mel,
        y_length,
        duration,
        pitch,
        energy
    )
