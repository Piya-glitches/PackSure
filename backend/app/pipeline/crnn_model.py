"""
STANDALONE CRNN ARCHITECTURE -- Phase 2 fine-tuning roadmap.

EasyOCR's built-in recognizer (wired up in text_detection_ocr.py) already
IS a CRNN and is what the running pipeline uses today. This file defines
the same architecture family explicitly and separately, as the base to
fine-tune on an OCR-error-augmented, Indian-packaging-specific dataset --
matching the architecture document's Layer 3 "noise tolerance" requirement
("Model trained on OCR-error-augmented text ... production-viable vs. a
demo toy").

Architecture, exactly as specified (CNN -> BiLSTM -> CTC):
  1. A convolutional feature extractor (VGG-style stack) turns a
     variable-width text-line image into a sequence of column feature
     vectors.
  2. A bidirectional LSTM models left-to-right and right-to-left context
     over that sequence (capturing character-shape context the CNN alone
     cannot -- e.g. distinguishing 'rn' from 'm').
  3. A CTC (Connectionist Temporal Classification) output head allows
     training without needing per-character bounding-box alignment --
     only the transcribed string is needed as a label, which is exactly
     what makes it practical to build a training set from real package
     photos (photograph + type out what it says, nothing more).

Training this on a labelled Indian-packaging dataset (recommended:
synthetic rendering of Indian regulatory phrases in varied fonts/glare/
blur, augmented with real photographed labels) is the concrete Phase 2
path to replacing/ensembling with EasyOCR's generic recognizer for higher
accuracy on this specific domain.
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, img_height: int = 32, num_channels: int = 1, num_classes: int = 96, rnn_hidden: int = 256):
        super().__init__()
        assert img_height % 16 == 0, "img_height must be a multiple of 16"

        # VGG-style convolutional feature extractor.
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),      # /2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),               # /4
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),    # /8 height, /4 width
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),    # /16 height, /4 width
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),             # collapse height to 1
        )

        # Bidirectional LSTM sequence modeling over the width dimension.
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        # CTC output head: one score per character class (+ blank) per timestep.
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, height, width)
        conv = self.cnn(x)                       # (batch, 512, 1, width')
        conv = conv.squeeze(2)                    # (batch, 512, width')
        conv = conv.permute(0, 2, 1)              # (batch, width', 512) -- sequence over width

        recurrent, _ = self.rnn(conv)             # (batch, width', hidden*2)
        output = self.fc(recurrent)               # (batch, width', num_classes)
        return output.permute(1, 0, 2)            # (width', batch, num_classes) -- CTC expects (T, N, C)


def ctc_greedy_decode(logits: torch.Tensor, idx_to_char: dict, blank_idx: int = 0) -> list[str]:
    """Simple greedy CTC decode (argmax + collapse repeats + drop blanks).
    A beam-search decoder (with an optional language model) is the natural
    upgrade for production accuracy."""
    probs = logits.softmax(dim=2)                 # (T, N, C)
    preds = probs.argmax(dim=2).permute(1, 0)      # (N, T)

    results = []
    for seq in preds:
        chars = []
        prev = None
        for idx in seq.tolist():
            if idx != prev and idx != blank_idx:
                chars.append(idx_to_char.get(idx, ""))
            prev = idx
        results.append("".join(chars))
    return results
