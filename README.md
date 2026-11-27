# fullfat-audio

**Clean-room `#![no_std]`, zero-dependency parser, software synthesizers, sequence decoder, WAV16 exporter, and SoundFont 2.01 (`.sf2`) builder for the Full Fat GBA Mono DirectSound audio engine.**

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![no_std](https://img.shields.io/badge/no__std-compatible-success.svg)](#features)

> **Looking for a ready-to-use CLI?** You can scan GBA ROMs, extract WAV samples, and compile SoundFont (`.sf2`) banks directly from the command line using our **[gbakit](https://github.com/JidouGhost-TL/gbakit)** toolkit (`gbakit audio scan`, `gbakit audio export`, `gbakit audio soundfont`).

---

## Features

- **Zero External Dependencies**: Pure Rust with empty `[dependencies]`, `#![no_std]` compatible (`alloc` optional), and `#![forbid(unsafe_code)]`.
- **Two Engine Revisions Detected**: Automatically distinguishes `Revision::RevA` (`16,780,000 Hz` Timer 0 clock, `bpl` clamp loop) and `Revision::RevB` (`16,777,216 Hz` Timer 0 clock, `bgt` clamp loop) via `FullFatVersion::detect_from_bytes` / `detect_details`.
- **Dual Synthesis Engines (`#![no_std]`)**:
  - **`AccuracySynth`**: Hardware-accurate mono DirectSound mixer matching the engine's ARM `MixChannel` (4.28 fixed-point pitch accumulator, `(master_vol * chan_vol) >> 12` gain, signed 16-bit mix buffer, modulo loop wrap) and `FinalizeBuffer` (`asr #4` + `[-128, +127]` clamp), supporting both signed 8-bit PCM and 4-bit OKI/Dialogic ADPCM.
  - **`StudioSynth`**: High-fidelity 48,000 Hz `f32` stereo renderer with a compile-time L1-resident Lanczos-3 polyphase LUT (256 phases × 6 taps, 64-byte aligned), linear fallback, and equal-power stereo panning.
- **Hardened WAV16 & SoundFont 2.01 Export (`alloc`)**: Single-allocation streaming SF2 builder with `rom_offset` sample deduplication, a 32 MiB PCM allocation cap (`MAX_TOTAL_PCM_BYTES`), collision-free monotonic GM preset numbering, and native 4-bit ADPCM decoding.
- **Zero-Allocation Sequence Decoder**: `SeqIter` implements `FusedIterator` with explicit truncation detection for the engine's variable-length sequence bytecode.

---

## Quickstart

```rust
use fullfat_audio::{
    AccuracySynth, FullFatVoice, Sample, SampleFormat, SampleHeader, StudioSynth,
    BASE_SAMPLE_RATE_HZ,
};

let pcm: [u8; 8] = [0, 32, 64, 96, 127, 96, 64, 32];
let sample = Sample {
    header: SampleHeader {
        rom_offset: 0,
        format: SampleFormat::Pcm8Signed,
        length: pcm.len() as u32,
        loop_start: 2,
        reserved: 0,
    },
    pcm: &pcm,
};

// Hardware-accurate mono DirectSound synthesis (i16 frames)
let synth = AccuracySynth::new(BASE_SAMPLE_RATE_HZ);
let mut voice = FullFatVoice::trigger(&sample, 60, 0x0100, BASE_SAMPLE_RATE_HZ);
let mut mono = [0i16; 32];
synth.render_voice_i16(&sample, &mut voice, &mut mono);
assert!(mono.iter().any(|&s| s != 0));

// 48 kHz studio-quality stereo rendering (Lanczos-3 + equal-power pan)
let studio = StudioSynth::new(48_000);
let mut pos = 0.0f64;
let mut left = [0.0f32; 32];
let mut right = [0.0f32; 32];
studio.render_voice_f32(&sample, &mut pos, 11_025.0, 0x0100, 0, &mut left, &mut right);
assert!(left.iter().all(|s| s.is_finite()));
assert!(left.iter().any(|&s| s != 0.0));
```

---

## Benchmarks

```sh
cargo bench --bench bench_audio
```

| Benchmark | Measured Throughput |
|---|---:|
| `detect` (16 MiB ROM) | ~8,407 MiB/s (1.90 ms) |
| `parse` (16 MiB ROM, 16 samples) | ~4,300 MiB/s (3.72 ms) |
| `export_wav_default` (4 KiB PCM8) | ~21,469 MiB/s (0.18 µs) |
| `build_sf2` (16 samples, 64 KiB PCM) | ~20,010 MiB/s (3.12 µs) |
| `SeqIter` decode (4,001 bytes) | ~1,354 MiB/s (2.82 µs) |
| `AccuracySynth::mix_voice` (1:1, 1,024f) | ~27.39 GSamples/s |
| `AccuracySynth::mix_voice` (resampled, 1,024f) | ~1.75 GSamples/s |
| `AccuracySynth::finalize_i16` (1,024f) | ~19.04 GSamples/s |
| `StudioSynth::render_voice_f32` (Lanczos-3 48 kHz, 1,024f) | ~432.8 MSamples/s |

---

## Community & Support

Come discuss GBA reverse engineering, audio synthesis, and tooling — or report issues and ask for help on your own project: https://discord.gg/khwxFSVNrS

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))
