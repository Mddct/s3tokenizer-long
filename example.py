import torch
import s3tokenizer
from typing import List


def sliding_window_audio(waveform,
                         sample_rate=24000,
                         window_size=30,
                         overlap=4,
                         pad_from_head=False):
    """
    Splits an audio waveform into overlapping 30-second segments.

    Args:
    - waveform (torch.Tensor): 1D tensor of raw audio samples.
    - sample_rate (int): Sample rate in Hz (default: 24000).
    - window_size (int): Window duration in seconds (default: 30s).
    - overlap (int): Overlapping duration in seconds (default: 4s).

    Returns:
    - List[torch.Tensor]: A list of 30s waveform segments.
    """
    frame_length = window_size * sample_rate  # Samples in a 30s window
    stride_length = (window_size -
                     overlap) * sample_rate  # Move forward by (30 - 4)s
    audio_segments = []

    start = 0
    end = 0
    for i in range(0, len(waveform), stride_length):
        start = i
        end = min(start + frame_length, len(waveform))

        # Pad last segment if needed
        if end - start < frame_length and pad_from_head:
            pad_length = frame_length - (end - start)
            pad_audio = waveform[:pad_length]
            segment = torch.cat([waveform[start:end], pad_audio])
        else:
            segment = waveform[start:end]

        audio_segments.append(segment)

    return audio_segments


def merge_tokenized_segments(tokenized_segments, overlap, token_rate):
    """
    Merges tokenized outputs by keeping the middle and dropping half of the overlapped tokens.

    Args:
    - tokenized_segments (List[List[int]]): List of tokenized sequences.
    - overlap (int): Overlapping duration in seconds (default: 4s).
    - token_rate (int): Number of tokens per second.

    Returns:
    - List[int]: A single merged token sequence.
    """
    merged_tokens = []
    overlap_tokens = (
        overlap //
        2) * token_rate  # Tokens corresponding to half of the overlap duration

    for i, tokens in enumerate(tokenized_segments):
        if i == 0:
            merged_tokens.extend(tokens)  # First window: keep all tokens
        elif i == len(tokenized_segments) - 1:
            # Keep only the middle part (drop overlap / 2 from both sides)
            merged_tokens.extend(tokens[overlap_tokens:-overlap_tokens])
        else:
            merged_tokens.extend(tokens[overlap_tokens:])

    return merged_tokens


class _S3tokenizer:

    def __init__(self, batch_size=20, device='cpu'):
        self.speech_tokenizer = s3tokenizer.load_model(
            "speech_tokenizer_v2_25hz")
        self.speech_tokenizer.freeze()
        self.speech_tokenizer.to(device)
        self.device = device

    def __call__(self, audios: List[torch.Tensor]) -> List[int]:
        mels = []
        for audio in audios:
            mels.append(s3tokenizer.log_mel_spectrogram(audio))
        mels, mels_lens = s3tokenizer.padding(mels)
        codes, codes_lens = self.speech_tokenizer.quantize(
            mels.to(self.device), mels_lens.to(self.device))
        codes = codes.cpu()
        codes_lens = codes_lens.cpu()
        codes_int = []
        for i in range(len(audios)):
            codes_int.append(codes[i, :codes_lens[i].item()].numpy().tolist())
        return codes_int


if __name__ == '__main__':

    sr = 16000
    window_size_in_seconds = 30
    overlap_in_seconds = 4
    token_rate = 25

    tokenizer = _S3tokenizer()
    wav = s3tokenizer.load_audio(
        "test.wav",
        sr,
    )

    segments = sliding_window_audio(wav, sr, window_size_in_seconds,
                                    overlap_in_seconds)
    segments_tokens = tokenizer(segments)
    final_tokens = merge_tokenized_segments(segments_tokens,
                                            overlap=4,
                                            token_rate=token_rate)
    assert len(final_tokens) == 6000
