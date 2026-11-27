//! Clean-room `#![no_std]` parser, software synthesizers, sequence decoder, WAV16 exporter,
//! and SoundFont 2.01 (`.sf2`) builder for the Full Fat GBA Mono DirectSound audio engine.
//!
//! # Architecture Overview
//! - Mono DirectSound A (`FIFO_A` @ `0x040000A0`, `SOUNDCNT_H = 0x0B04`, `DMA1CNT = 0xB6000000`)
//! - Dynamic sample rate via Timer 0 (`0x04000100`) + Timer 1 cascade IRQ (`0x04000104`)
//! - ARM IWRAM software mixer with 4.28 fixed-point pitch accumulator and signed 16-bit mix buffer,
//!   finalized via `>> 4` and clamped to signed 8-bit PCM `[-128, +127]`
//! - Base sample rate: `11,025 Hz` (`0x2B11`) at MIDI root note 60 (Middle C, `261 Hz` reference)
//! - Sound banks store a 16-byte header (`song_count`, `sample_count`, `song_table_ptr`,
//!   `sample_table_ptr`), followed immediately by `song_count * 8`-byte song entries and
//!   `sample_count * 4`-byte pointers to 16-byte sample headers + PCM8 / ADPCM4 payloads.
//!
//! # Quickstart
//!
//! ```rust
//! use fullfat_audio::{
//!     AccuracySynth, FullFatVoice, Sample, SampleFormat, SampleHeader, StudioSynth,
//!     BASE_SAMPLE_RATE_HZ,
//! };
//!
//! let pcm: [u8; 8] = [0, 32, 64, 96, 127, 96, 64, 32];
//! let sample = Sample {
//!     header: SampleHeader {
//!         rom_offset: 0,
//!         format: SampleFormat::Pcm8Signed,
//!         length: pcm.len() as u32,
//!         loop_start: 2,
//!         reserved: 0,
//!     },
//!     pcm: &pcm,
//! };
//!
//! // Hardware-accurate mono DirectSound synthesis (i16 frames)
//! let synth = AccuracySynth::new(BASE_SAMPLE_RATE_HZ);
//! let mut voice = FullFatVoice::trigger(&sample, 60, 0x0100, BASE_SAMPLE_RATE_HZ);
//! let mut mono = [0i16; 32];
//! synth.render_voice_i16(&sample, &mut voice, &mut mono);
//! assert!(mono.iter().any(|&s| s != 0));
//!
//! // 48 kHz studio-quality stereo rendering (Lanczos-3 + equal-power pan)
//! let studio = StudioSynth::new(48_000);
//! let mut pos = 0.0f64;
//! let mut left = [0.0f32; 32];
//! let mut right = [0.0f32; 32];
//! studio.render_voice_f32(&sample, &mut pos, 11_025.0, 0x0100, 0, &mut left, &mut right);
//! assert!(left.iter().all(|s| s.is_finite()));
//! assert!(left.iter().any(|&s| s != 0.0));
//! ```

#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod adpcm;
pub mod detect;
pub mod parse;
pub mod seq;
pub mod synth;

#[cfg(feature = "alloc")]
pub mod export;

pub use adpcm::Adpcm4Decoder;
pub use detect::{detect, detect_details, Detection, FullFatVersion, Revision};
pub use parse::{
    parse, BankSampleIter, BankSummary, Error, FullFatAudio, Result, Sample, SampleFormat,
    SampleHeader, SongEntry, MAX_BANKS, MAX_GBA_ROM_BYTES,
};
pub use seq::{SeqEvent, SeqIter};
pub use synth::{
    compute_channel_gain, compute_pitch_step, finalize_sample_dma8, note_to_playback_hz,
    AccuracySynth, FullFatVoice, StudioSynth, FREQ_TABLE_Q16, MAX_VOICE_VOLUME,
};

#[cfg(feature = "alloc")]
pub use export::{build_sf2, export_wav, export_wav_default, MAX_TOTAL_PCM_BYTES};

/// GBA ROM base address in the ARM7TDMI memory map.
pub const GBA_ROM_BASE: u32 = 0x0800_0000;

/// Base sample rate in Hz (`0x2B11`) used by the engine's pitch-table generator.
pub const BASE_SAMPLE_RATE_HZ: u32 = 11_025;

/// MIDI root key corresponding to the engine's `261 Hz` (`0x0105`) pitch reference (Middle C).
pub const ROOT_KEY: u8 = 60;
