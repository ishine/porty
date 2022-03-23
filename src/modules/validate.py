import torch
import matplotlib.pyplot as plt
import torchaudio
from tqdm import tqdm
from torchaudio.sox_effects import apply_effects_tensor
from pathlib import Path

from .commons.tokenizer import Tokenizer
from .commons.transforms import MelSpectrogram
from .models.model import VITS


SR = 24000


def validate(args, config):
    output_dir = Path(config.output_dir) / 'validate'
    output_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VITS(config.model)
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt['g'])
    model = model.eval().to(device)

    data_dir = Path(config.data.data_dir)
    data_list = list(sorted(data_dir.glob('*.pt')))[:config.data.valid_size]

    tokenizer = Tokenizer()
    to_mel = MelSpectrogram()

    def save_fig(gen, gt, path):
        plt.figure(figsize=(14, 7))
        plt.subplot(211)
        plt.gca().title.set_text('GEN')
        plt.imshow(gen, aspect='auto', origin='lower')
        plt.subplot(212)
        plt.gca().title.set_text('GT')
        plt.imshow(gt, aspect='auto', origin='lower')
        plt.savefig(path)
        plt.close()

    def save_wav(wav, path):
        effects = [
            ['gain', '-n']
        ]
        wav, _ = apply_effects_tensor(wav, SR, effects, channels_first=True)
        torchaudio.save(
            str(path),
            wav,
            SR
        )

    for i, p in tqdm(enumerate(data_list), total=len(data_list)):
        d = output_dir / f'res_{i+1:04d}'
        d.mkdir(exist_ok=True)
        (
            wav,
            spec,
            mel,
            label,
            *_
        ) = torch.load(p)
        x = tokenizer(label)
        length = torch.LongTensor([len(label)])

        x = x.unsqueeze(0).to(device)
        length = length.to(device)

        with torch.no_grad():
            w = model([x, length])
            w = w.squeeze(1).detach().cpu()
        m = to_mel(w).squeeze(0)

        save_wav(wav, d / f'gt.wav')
        save_wav(w, d / f'gen.wav')

        save_fig(m, mel, d / f'mel_gan.png')
