//! Clean-room `#![no_std]` software synthesizers for the Full Fat GBA Mono DirectSound engine.
//!
//! Provides two synthesis pipelines:
//! - [`AccuracySynth`]: Hardware-accurate mono DirectSound mixer matching Full Fat's ARM
//!   `MixChannel` (4.28 fixed-point pitch accumulator, `(master_vol * chan_vol) >> 12` gain,
//!   signed 16-bit mix buffer, modulo loop wrap) and `FinalizeBuffer` (`asr #4`, `[-128, +127]`
//!   clamp), supporting both signed 8-bit PCM and 4-bit ADPCM samples.
//! - [`StudioSynth`]: High-fidelity 48,000 Hz `f32` stereo renderer with a compile-time
//!   L1-resident Lanczos-3 polyphase LUT (256 phases, 64-byte aligned), general Lanczos-`a` /
//!   linear fallback, and equal-power stereo panning.

use crate::adpcm::Adpcm4Decoder;
use crate::detect::Revision;
use crate::parse::{Sample, SampleFormat};
use crate::BASE_SAMPLE_RATE_HZ;

/// Root pitch reference frequency in Hz (`261 Hz`, MIDI note 60 / Middle C).
pub const ROOT_PITCH_HZ: u32 = 261;

/// Default maximum voice volume (`0x0100`).
pub const MAX_VOICE_VOLUME: u16 = 0x0100;

/// Default master volume (`0x0100`).
pub const DEFAULT_MASTER_VOLUME: u16 = 0x0100;

/// 128-entry 16.16 fixed-point MIDI frequency table (`A4 = note 69 = 440 << 16 = 28,835,840`).
///
/// Computed at compile time from equal temperament `440 * 2^((n - 69) / 12) * 65536`.
pub const FREQ_TABLE_Q16: [u32; 128] = build_freq_table_q16();

const fn build_freq_table_q16() -> [u32; 128] {
    // 2^(1/12) = 1.0594630943592953
    const SEMITONE: f64 = 1.059_463_094_359_295_3;
    let mut table = [0u32; 128];
    let mut i = 0usize;
    while i < 128 {
        let mut f = 440.0 * 65536.0;
        if i > 69 {
            let mut k = 69;
            while k < i {
                f *= SEMITONE;
                k += 1;
            }
        } else if i < 69 {
            let mut k = i;
            while k < 69 {
                f /= SEMITONE;
                k += 1;
            }
        }
        table[i] = (f + 0.5) as u32;
        i += 1;
    }
    table[69] = 28_835_840;
    table
}

/// Computes the engine's 4.28 fixed-point pitch step for `note` at `output_rate` Hz.
///
/// Matches the engine's pitch-table formula:
/// `((FREQ_TABLE_Q16[note.min(127)] * 11025) << 12) / (261 * output_rate)`.
///
/// Returns `0` if `output_rate == 0`.
#[inline]
#[must_use]
pub const fn compute_pitch_step(note: u8, output_rate: u32) -> u32 {
    if output_rate == 0 {
        return 0;
    }
    let idx = if note > 127 { 127 } else { note as usize };
    let freq_q16 = FREQ_TABLE_Q16[idx] as u64;
    let num = (freq_q16 * (BASE_SAMPLE_RATE_HZ as u64)) << 12;
    let den = (ROOT_PITCH_HZ as u64) * (output_rate as u64);
    (num / den) as u32
}

/// Computes the per-channel mixer gain `(master_vol * chan_vol) >> 12`.
#[inline]
#[must_use]
pub const fn compute_channel_gain(master_vol: u16, chan_vol: u16) -> i32 {
    (((master_vol as u32) * (chan_vol as u32)) >> 12) as i32
}

/// Clamps a signed 16-bit mix accumulator to a signed 8-bit DirectSound DMA byte,
/// matching `FinalizeBuffer`: `asr #4` followed by `[-128, +127]` saturation.
#[inline]
#[must_use]
pub const fn finalize_sample_dma8(acc: i16) -> i8 {
    let shifted = (acc as i32) >> 4;
    if shifted < -128 {
        -128
    } else if shifted > 127 {
        127
    } else {
        shifted as i8
    }
}

/// Active voice state for [`AccuracySynth`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FullFatVoice {
    /// Current integer sample index (byte index for PCM8; nibble/sample index for ADPCM4).
    pub pos: usize,
    /// 32-bit fractional phase accumulator (advanced by `pitch_step << 4` with carry into `pos`).
    pub frac_phase: u32,
    /// 4.28 fixed-point pitch step per output frame.
    pub pitch_step: u32,
    /// Channel volume (`0x0000..=0x0100`).
    pub volume: u16,
    /// Loop start offset in bytes (`0` = one-shot).
    pub loop_start: usize,
    /// Whether the voice is actively playing.
    pub active: bool,
    /// Stateful 4-bit ADPCM decoder (used when mixing [`SampleFormat::Adpcm4`]).
    pub adpcm: Adpcm4Decoder,
    /// Cached decoded pair for sequential ADPCM4 playback.
    adpcm_cached_byte: usize,
    adpcm_pair: [i16; 2],
    adpcm_valid: bool,
}

impl Default for FullFatVoice {
    fn default() -> Self {
        Self::new()
    }
}

impl FullFatVoice {
    /// Creates an inactive default voice.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pos: 0,
            frac_phase: 0,
            pitch_step: 0x1000_0000, // 1.0 in 4.28
            volume: MAX_VOICE_VOLUME,
            loop_start: 0,
            active: false,
            adpcm: Adpcm4Decoder::new(),
            adpcm_cached_byte: 0,
            adpcm_pair: [0, 0],
            adpcm_valid: false,
        }
    }

    /// Triggers a voice for `sample` at MIDI `note`, `volume` (`0..=0x0100`), and `output_rate` Hz.
    #[inline]
    #[must_use]
    pub const fn trigger(sample: &Sample<'_>, note: u8, volume: u16, output_rate: u32) -> Self {
        let loop_start = if sample.header.is_looped() {
            sample.header.loop_start as usize
        } else {
            0
        };
        let clamped_vol = if volume > MAX_VOICE_VOLUME {
            MAX_VOICE_VOLUME
        } else {
            volume
        };
        Self {
            pos: 0,
            frac_phase: 0,
            pitch_step: compute_pitch_step(note, output_rate),
            volume: clamped_vol,
            loop_start,
            active: !sample.pcm.is_empty(),
            adpcm: Adpcm4Decoder::new(),
            adpcm_cached_byte: 0,
            adpcm_pair: [0, 0],
            adpcm_valid: false,
        }
    }

    /// Creates an active voice with explicit raw 4.28 `pitch_step`, `volume`, and `loop_start`.
    #[inline]
    #[must_use]
    pub const fn with_raw_params(pitch_step: u32, volume: u16, loop_start: usize) -> Self {
        Self {
            pos: 0,
            frac_phase: 0,
            pitch_step,
            volume,
            loop_start,
            active: true,
            adpcm: Adpcm4Decoder::new(),
            adpcm_cached_byte: 0,
            adpcm_pair: [0, 0],
            adpcm_valid: false,
        }
    }

    /// Stops this voice immediately.
    #[inline]
    pub const fn stop(&mut self) {
        self.active = false;
    }
}

/// Hardware-accurate mono DirectSound software mixer for the Full Fat audio engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AccuracySynth {
    /// Output sample rate in Hz (default `11,025 Hz`).
    pub sample_rate: u32,
    /// Master volume (`0..=0x0100`, default `0x0100`).
    pub master_volume: u16,
    /// Engine build revision (determines Timer 0 clock constant).
    pub revision: Revision,
}

impl Default for AccuracySynth {
    fn default() -> Self {
        Self::new(BASE_SAMPLE_RATE_HZ)
    }
}

impl AccuracySynth {
    /// Creates a new `AccuracySynth` at `sample_rate` Hz using [`Revision::RevB`].
    #[inline]
    #[must_use]
    pub const fn new(sample_rate: u32) -> Self {
        Self::with_revision(sample_rate, Revision::RevB)
    }

    /// Creates a new `AccuracySynth` for a specific [`Revision`].
    #[inline]
    #[must_use]
    pub const fn with_revision(sample_rate: u32, revision: Revision) -> Self {
        Self {
            sample_rate,
            master_volume: DEFAULT_MASTER_VOLUME,
            revision,
        }
    }

    /// Mixes a single [`Sample`] voice into a signed 16-bit mono mix buffer (`acc`),
    /// matching Full Fat's ARM `MixChannel` inner loop.
    ///
    /// When `first_voice` is `true`, the accumulator is overwritten (`mul`); otherwise
    /// the voice is accumulated (`mla`) with 16-bit wrapping semantics.
    pub fn mix_voice(
        &self,
        sample: &Sample<'_>,
        voice: &mut FullFatVoice,
        acc: &mut [i16],
        first_voice: bool,
    ) {
        if acc.is_empty() || !voice.active || sample.pcm.is_empty() {
            if first_voice {
                acc.fill(0);
            }
            return;
        }
        let gain = compute_channel_gain(self.master_volume, voice.volume);
        match sample.header.format {
            SampleFormat::Pcm8Signed => {
                Self::mix_pcm8(sample.pcm, voice, gain, acc, first_voice);
            }
            SampleFormat::Adpcm4 => {
                Self::mix_adpcm4(sample.pcm, voice, gain, acc, first_voice);
            }
            SampleFormat::Unknown(_) => {
                if first_voice {
                    acc.fill(0);
                }
            }
        }
    }

    /// Mixes raw signed 8-bit PCM bytes into `acc` using Full Fat's 4.28 pitch accumulator
    /// and modulo loop-wrap semantics.
    pub fn mix_pcm8(
        pcm: &[u8],
        voice: &mut FullFatVoice,
        gain: i32,
        acc: &mut [i16],
        first_voice: bool,
    ) {
        let len = acc.len();
        let pcm_len = pcm.len();
        if len == 0 || pcm_len == 0 || !voice.active {
            if first_voice {
                acc.fill(0);
            }
            return;
        }

        let step = voice.pitch_step;
        let loop_start = if voice.loop_start > 0 && voice.loop_start < pcm_len {
            voice.loop_start
        } else {
            0
        };

        // Fast path: 1:1 playback (`step == 0x1000_0000`, `frac_phase == 0`).
        if step == 0x1000_0000 && voice.frac_phase == 0 && voice.pos < pcm_len {
            let mut off = 0usize;
            let mut pos = voice.pos;
            while off < len {
                let chunk = (pcm_len - pos).min(len - off);
                let src = &pcm[pos..pos + chunk];
                let dst = &mut acc[off..off + chunk];
                if first_voice {
                    for (d, &b) in dst.iter_mut().zip(src.iter()) {
                        *d = ((b as i8 as i32).wrapping_mul(gain)) as i16;
                    }
                } else {
                    for (d, &b) in dst.iter_mut().zip(src.iter()) {
                        let prod = (b as i8 as i32).wrapping_mul(gain);
                        *d = ((*d as i32).wrapping_add(prod)) as i16;
                    }
                }
                pos += chunk;
                off += chunk;
                if pos >= pcm_len {
                    if loop_start == 0 {
                        voice.active = false;
                        voice.pos = pcm_len;
                        if first_voice && off < len {
                            acc[off..].fill(0);
                        }
                        return;
                    }
                    let loop_len = pcm_len - loop_start;
                    pos = loop_start + ((pos - pcm_len) % loop_len);
                }
            }
            voice.pos = pos;
            return;
        }

        // General 4.28 fixed-point resampled path matching ARM `adds r3, r3, r4, lsl #4; adc r8, r8, r4, lsr #28`.
        let frac_step = step << 4;
        let int_step = (step >> 28) as usize;
        let mut pos = voice.pos;
        let mut frac = voice.frac_phase;

        for i in 0..len {
            if pos >= pcm_len {
                if loop_start == 0 {
                    voice.active = false;
                    voice.pos = pos;
                    voice.frac_phase = frac;
                    if first_voice {
                        acc[i..].fill(0);
                    }
                    return;
                }
                let loop_len = pcm_len - loop_start;
                pos = loop_start + ((pos - pcm_len) % loop_len);
            }

            let sample_i8 = pcm[pos] as i8 as i32;
            let prod = sample_i8.wrapping_mul(gain);
            if first_voice {
                acc[i] = prod as i16;
            } else {
                acc[i] = ((acc[i] as i32).wrapping_add(prod)) as i16;
            }

            let (new_frac, carry) = frac.overflowing_add(frac_step);
            frac = new_frac;
            pos = pos.saturating_add(int_step + usize::from(carry));
        }

        // Handle post-loop boundary check matching the engine's post-mix wrap.
        if pos >= pcm_len {
            if loop_start == 0 {
                voice.active = false;
            } else {
                let loop_len = pcm_len - loop_start;
                pos = loop_start + ((pos - pcm_len) % loop_len);
            }
        }
        voice.pos = pos;
        voice.frac_phase = frac;
    }

    /// Mixes a 4-bit ADPCM voice into `acc`, decoding packed nibbles sequentially
    /// through [`Adpcm4Decoder`].
    fn mix_adpcm4(
        packed: &[u8],
        voice: &mut FullFatVoice,
        gain: i32,
        acc: &mut [i16],
        first_voice: bool,
    ) {
        let total_samples = packed.len().saturating_mul(2);
        if total_samples == 0 || !voice.active {
            if first_voice {
                acc.fill(0);
            }
            return;
        }
        let loop_start_samples = if voice.loop_start > 0 && voice.loop_start < packed.len() {
            voice.loop_start * 2
        } else {
            0
        };

        let mut pos = voice.pos;
        let mut frac = voice.frac_phase;
        let frac_step = voice.pitch_step << 4;
        let int_step = (voice.pitch_step >> 28) as usize;

        for d in acc.iter_mut() {
            if pos >= total_samples {
                if loop_start_samples == 0 {
                    voice.active = false;
                    if first_voice {
                        *d = 0;
                    }
                    continue;
                }
                let loop_len = total_samples - loop_start_samples;
                pos = loop_start_samples + ((pos - total_samples) % loop_len);
                voice.adpcm.reset();
                voice.adpcm_valid = false;
            }

            let byte_idx = pos >> 1;
            let nib_sel = pos & 1;

            if !voice.adpcm_valid || byte_idx < voice.adpcm_cached_byte {
                voice.adpcm.reset();
                for &b in &packed[..=byte_idx] {
                    voice.adpcm_pair = voice.adpcm.decode_byte_mixer(b);
                }
                voice.adpcm_cached_byte = byte_idx;
                voice.adpcm_valid = true;
            } else if byte_idx > voice.adpcm_cached_byte {
                for &b in &packed[voice.adpcm_cached_byte + 1..=byte_idx] {
                    voice.adpcm_pair = voice.adpcm.decode_byte_mixer(b);
                }
                voice.adpcm_cached_byte = byte_idx;
            }

            let s = voice.adpcm_pair[nib_sel] as i32;
            let prod = s.wrapping_mul(gain);
            if first_voice {
                *d = prod as i16;
            } else {
                *d = ((*d as i32).wrapping_add(prod)) as i16;
            }

            let (new_frac, carry) = frac.overflowing_add(frac_step);
            frac = new_frac;
            pos = pos.saturating_add(int_step + usize::from(carry));
        }

        if pos >= total_samples {
            if loop_start_samples == 0 {
                voice.active = false;
            } else {
                let loop_len = total_samples - loop_start_samples;
                pos = loop_start_samples + ((pos - total_samples) % loop_len);
                voice.adpcm.reset();
                voice.adpcm_valid = false;
            }
        }
        voice.pos = pos;
        voice.frac_phase = frac;
    }

    /// Finalizes a 16-bit mix buffer into signed 8-bit DirectSound DMA bytes
    /// (`asr #4` + clamp `[-128, +127]`), matching `FinalizeBuffer`.
    pub fn finalize_dma8(acc: &[i16], out: &mut [i8]) {
        for (dst, &src) in out.iter_mut().zip(acc.iter()) {
            *dst = finalize_sample_dma8(src);
        }
    }

    /// Finalizes a 16-bit mix buffer into signed 16-bit PCM frames by applying
    /// `FinalizeBuffer`'s 8-bit DirectSound quantization and scaling to `i16` (`<< 8`).
    pub fn finalize_i16(acc: &[i16], out: &mut [i16]) {
        for (dst, &src) in out.iter_mut().zip(acc.iter()) {
            *dst = (finalize_sample_dma8(src) as i16) << 8;
        }
    }

    /// Convenience helper: mixes a single voice and finalizes directly to `i16` frames.
    pub fn render_voice_i16(&self, sample: &Sample<'_>, voice: &mut FullFatVoice, out: &mut [i16]) {
        self.mix_voice(sample, voice, out, true);
        for s in out.iter_mut() {
            *s = (finalize_sample_dma8(*s) as i16) << 8;
        }
    }
}

// ============================================================================
// StudioSynth (48 kHz f32 Lanczos-3 / Linear Stereo Renderer)
// ============================================================================

/// Number of polyphase sub-sample phases in the compile-time Lanczos-3 LUT.
const LANCZOS3_PHASES: usize = 256;

/// 64-byte aligned Lanczos-3 polyphase LUT (256 phases × 6 taps, 6 KiB — fits in L1 cache).
#[repr(align(64))]
struct Lanczos3Lut([[f32; 6]; LANCZOS3_PHASES]);

static LANCZOS3_LUT: Lanczos3Lut = build_lanczos3_lut();

const fn build_lanczos3_lut() -> Lanczos3Lut {
    let mut table = [[0.0f32; 6]; LANCZOS3_PHASES];
    let mut p = 0usize;
    while p < LANCZOS3_PHASES {
        let frac = (p as f64) / (LANCZOS3_PHASES as f64);
        let mut weights = [0.0f64; 6];
        let mut sum = 0.0f64;
        let mut idx = 0usize;
        while idx < 6 {
            let tap = (idx as i32) - 2; // -2, -1, 0, +1, +2, +3
            let w = lanczos_kernel(frac - (tap as f64), 3.0);
            weights[idx] = w;
            sum += w;
            idx += 1;
        }
        let inv = if sum > 1e-12 { 1.0 / sum } else { 0.0 };
        let mut j = 0usize;
        while j < 6 {
            table[p][j] = (weights[j] * inv) as f32;
            j += 1;
        }
        p += 1;
    }
    Lanczos3Lut(table)
}

const fn lanczos_kernel(x: f64, a: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let ax = if x < 0.0 { -x } else { x };
    if ax >= a {
        return 0.0;
    }
    let pi_x = core::f64::consts::PI * x;
    let pi_x_a = pi_x / a;
    (sin_f64(pi_x) / pi_x) * (sin_f64(pi_x_a) / pi_x_a)
}

const fn sin_f64(mut x: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let half_pi = core::f64::consts::FRAC_PI_2;
    let two_pi = 2.0 * pi;
    while x > pi {
        x -= two_pi;
    }
    while x < -pi {
        x += two_pi;
    }
    if x > half_pi {
        x = pi - x;
    } else if x < -half_pi {
        x = -pi - x;
    }
    let x2 = x * x;
    let mut term = x;
    let mut sum = x;
    let mut n = 1.0f64;
    let mut sign = -1.0f64;
    let mut i = 0;
    while i < 8 {
        term *= x2 / ((n + 1.0) * (n + 2.0));
        sum += sign * term;
        sign = -sign;
        n += 2.0;
        i += 1;
    }
    sum
}

const fn cos_f64(x: f64) -> f64 {
    sin_f64(x + core::f64::consts::FRAC_PI_2)
}

#[inline]
const fn floor_f64(x: f64) -> f64 {
    let t = x as i64 as f64;
    if x < 0.0 && x != t {
        t - 1.0
    } else {
        t
    }
}

/// Computes the playback frequency in Hz for a MIDI `note` relative to [`ROOT_KEY`] (60)
/// at the engine's base sample rate (`11,025 Hz`).
#[inline]
#[must_use]
pub fn note_to_playback_hz(note: u8) -> f64 {
    let idx = (note as usize).min(127);
    let freq_q16 = FREQ_TABLE_Q16[idx] as f64;
    (freq_q16 / 65536.0) * (BASE_SAMPLE_RATE_HZ as f64) / (ROOT_PITCH_HZ as f64)
}

/// High-fidelity 48,000 Hz `f32` studio renderer with Lanczos-3 interpolation
/// (or linear interpolation when `sinc_lobes == 0`) and equal-power stereo panning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StudioSynth {
    /// Output sample rate in Hz (default `48,000`).
    pub sample_rate: u32,
    /// Interpolation lobes (`3` = compile-time Lanczos-3 LUT, `0` = linear interpolation).
    pub sinc_lobes: usize,
}

impl Default for StudioSynth {
    fn default() -> Self {
        Self::new(48_000)
    }
}

impl StudioSynth {
    /// Creates a new `StudioSynth` at `sample_rate` Hz with Lanczos-3 interpolation.
    #[inline]
    #[must_use]
    pub const fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            sinc_lobes: 3,
        }
    }

    /// Creates a new `StudioSynth` using fast linear interpolation.
    #[inline]
    #[must_use]
    pub const fn with_linear(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            sinc_lobes: 0,
        }
    }

    /// Renders a [`Sample`] voice into stereo `f32` buffers (accumulating).
    ///
    /// - `pos`: current fractional sample position (advanced in place; non-finite values are
    ///   treated as no-ops so output is guaranteed finite).
    /// - `pitch_hz`: playback sample rate in Hz (e.g. from [`note_to_playback_hz`]).
    /// - `volume`: `0..=256` (`0x0100` = unity gain).
    /// - `pan`: `-64..=64` (`-64` = hard left, `0` = center, `+64` = hard right).
    #[allow(clippy::too_many_arguments)]
    pub fn render_voice_f32(
        &self,
        sample: &Sample<'_>,
        pos: &mut f64,
        pitch_hz: f64,
        volume: u16,
        pan: i8,
        out_left: &mut [f32],
        out_right: &mut [f32],
    ) {
        let len = out_left.len().min(out_right.len());
        if len == 0
            || sample.pcm.is_empty()
            || self.sample_rate == 0
            || !pos.is_finite()
            || !pitch_hz.is_finite()
            || pitch_hz <= 0.0
            || *pos < 0.0
        {
            return;
        }

        let step = pitch_hz / (self.sample_rate as f64);
        if !step.is_finite() || step <= 0.0 {
            return;
        }

        let vol_norm = (volume.min(MAX_VOICE_VOLUME) as f64) / (MAX_VOICE_VOLUME as f64);
        let pan_norm = ((pan.clamp(-64, 64) as f64) + 64.0) / 128.0;
        let angle = pan_norm * core::f64::consts::FRAC_PI_2;
        let gain_l = (vol_norm * cos_f64(angle)) as f32;
        let gain_r = (vol_norm * sin_f64(angle)) as f32;

        match sample.header.format {
            SampleFormat::Pcm8Signed => {
                let looped = sample.header.is_looped();
                let loop_start = if looped {
                    sample.header.loop_start as usize
                } else {
                    0
                };
                // Fold 1/128 i8 normalization into channel gains.
                let gl = gain_l * (1.0 / 128.0);
                let gr = gain_r * (1.0 / 128.0);
                if self.sinc_lobes == 3 {
                    Self::render_pcm8_lanczos3(
                        sample.pcm,
                        pos,
                        step,
                        gl,
                        gr,
                        looped,
                        loop_start,
                        &mut out_left[..len],
                        &mut out_right[..len],
                    );
                } else {
                    Self::render_pcm8_linear(
                        sample.pcm,
                        pos,
                        step,
                        gl,
                        gr,
                        looped,
                        loop_start,
                        &mut out_left[..len],
                        &mut out_right[..len],
                    );
                }
            }
            SampleFormat::Adpcm4 => {
                let looped = sample.header.is_looped();
                let loop_start_samples = if looped {
                    (sample.header.loop_start as usize).saturating_mul(2)
                } else {
                    0
                };
                // Fold 1/2048 12-bit predictor normalization into channel gains.
                let gl = gain_l * (1.0 / 2048.0);
                let gr = gain_r * (1.0 / 2048.0);
                Self::render_adpcm4_linear(
                    sample.pcm,
                    pos,
                    step,
                    gl,
                    gr,
                    looped,
                    loop_start_samples,
                    &mut out_left[..len],
                    &mut out_right[..len],
                );
            }
            SampleFormat::Unknown(_) => {}
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_pcm8_lanczos3(
        pcm: &[u8],
        pos: &mut f64,
        step: f64,
        gain_l: f32,
        gain_r: f32,
        looped: bool,
        loop_start: usize,
        out_left: &mut [f32],
        out_right: &mut [f32],
    ) {
        const Q32_SCALE: f64 = 4_294_967_296.0;
        const Q32_INV: f64 = 1.0 / 4_294_967_296.0;

        let len = out_left.len();
        let pcm_len = pcm.len();
        let pcm_end = pcm_len as f64;
        let loop_len = pcm_len.saturating_sub(loop_start);
        let loop_len_f = loop_len as f64;
        let mut cur_pos = *pos;
        let mut off = 0usize;

        let can_interior = step > 0.0 && pcm_len >= 6 && step < 65_536.0;
        let step_fp = if can_interior {
            (step * Q32_SCALE) as u64
        } else {
            0
        };
        let min_interior_f = 2.0f64;
        let max_interior_f = pcm_len.saturating_sub(3) as f64;
        let min_fp = 2u64 << 32;
        let limit_fp = (pcm_len.saturating_sub(3) as u64) << 32;

        while off < len {
            if step_fp > 0 && cur_pos >= min_interior_f && cur_pos < max_interior_f {
                let mut fp = (cur_pos * Q32_SCALE) as u64;
                if fp >= min_fp && fp < limit_fp {
                    let safe = (((limit_fp - 1 - fp) / step_fp) as usize + 1).min(len - off);
                    for (l, r) in out_left[off..off + safe]
                        .iter_mut()
                        .zip(out_right[off..off + safe].iter_mut())
                    {
                        let base = (fp >> 32) as usize;
                        let phase = ((fp >> 24) & 0xFF) as usize;
                        let w = &LANCZOS3_LUT.0[phase];
                        let s = &pcm[base - 2..base + 4];
                        let p0 = (s[0] as i8 as f32) * w[0] + (s[1] as i8 as f32) * w[1];
                        let p1 = (s[2] as i8 as f32) * w[2] + (s[3] as i8 as f32) * w[3];
                        let p2 = (s[4] as i8 as f32) * w[4] + (s[5] as i8 as f32) * w[5];
                        let value = (p0 + p1) + p2;
                        *l += value * gain_l;
                        *r += value * gain_r;
                        fp = fp.wrapping_add(step_fp);
                    }
                    cur_pos = (fp as f64) * Q32_INV;
                    off += safe;
                    continue;
                }
            }

            if !looped && cur_pos >= pcm_end {
                cur_pos = pcm_end;
                break;
            }

            let base_f = floor_f64(cur_pos);
            let frac = cur_pos - base_f;
            let phase = ((frac * (LANCZOS3_PHASES as f64)) as usize).min(LANCZOS3_PHASES - 1);
            let w = &LANCZOS3_LUT.0[phase];
            let base = base_f as i64;
            let s0 = Self::fetch_pcm8(pcm, base - 2, looped, loop_start) as f32;
            let s1 = Self::fetch_pcm8(pcm, base - 1, looped, loop_start) as f32;
            let s2 = Self::fetch_pcm8(pcm, base, looped, loop_start) as f32;
            let s3 = Self::fetch_pcm8(pcm, base + 1, looped, loop_start) as f32;
            let s4 = Self::fetch_pcm8(pcm, base + 2, looped, loop_start) as f32;
            let s5 = Self::fetch_pcm8(pcm, base + 3, looped, loop_start) as f32;
            let value = (s0 * w[0] + s1 * w[1]) + (s2 * w[2] + s3 * w[3]) + (s4 * w[4] + s5 * w[5]);
            out_left[off] += value * gain_l;
            out_right[off] += value * gain_r;

            cur_pos += step;
            off += 1;
            if looped && loop_len > 0 && cur_pos >= pcm_end {
                let over = cur_pos - pcm_end;
                let wraps = floor_f64(over / loop_len_f);
                cur_pos = (loop_start as f64) + (over - wraps * loop_len_f);
            }
        }

        *pos = cur_pos;
    }

    #[allow(clippy::too_many_arguments)]
    fn render_pcm8_linear(
        pcm: &[u8],
        pos: &mut f64,
        step: f64,
        gain_l: f32,
        gain_r: f32,
        looped: bool,
        loop_start: usize,
        out_left: &mut [f32],
        out_right: &mut [f32],
    ) {
        let pcm_len = pcm.len();
        let pcm_end = pcm_len as f64;
        let loop_len = pcm_len.saturating_sub(loop_start);
        let loop_len_f = loop_len as f64;
        let mut cur_pos = *pos;

        for (l, r) in out_left.iter_mut().zip(out_right.iter_mut()) {
            if !looped && cur_pos >= pcm_end {
                cur_pos = pcm_end;
                break;
            }
            let base_f = floor_f64(cur_pos);
            let frac = (cur_pos - base_f) as f32;
            let base = base_f as i64;
            let s0 = Self::fetch_pcm8(pcm, base, looped, loop_start) as f32;
            let s1 = Self::fetch_pcm8(pcm, base + 1, looped, loop_start) as f32;
            let value = s0 + (s1 - s0) * frac;
            *l += value * gain_l;
            *r += value * gain_r;

            cur_pos += step;
            if looped && loop_len > 0 && cur_pos >= pcm_end {
                let over = cur_pos - pcm_end;
                let wraps = floor_f64(over / loop_len_f);
                cur_pos = (loop_start as f64) + (over - wraps * loop_len_f);
            }
        }
        *pos = cur_pos;
    }

    #[allow(clippy::too_many_arguments)]
    fn render_adpcm4_linear(
        packed: &[u8],
        pos: &mut f64,
        step: f64,
        gain_l: f32,
        gain_r: f32,
        looped: bool,
        loop_start: usize,
        out_left: &mut [f32],
        out_right: &mut [f32],
    ) {
        let total_samples = packed.len().saturating_mul(2);
        if total_samples == 0 {
            return;
        }
        let end_f = total_samples as f64;
        let loop_len = total_samples.saturating_sub(loop_start);
        let loop_len_f = loop_len as f64;
        let mut cur_pos = *pos;

        let mut dec = Adpcm4Decoder::new();
        let mut decoded_bytes = 0usize;
        let mut prev_pair = [0i16; 2];
        let mut cur_pair = [0i16; 2];

        for (l, r) in out_left.iter_mut().zip(out_right.iter_mut()) {
            if !looped && cur_pos >= end_f {
                cur_pos = end_f;
                break;
            }
            let base_f = floor_f64(cur_pos);
            let frac = (cur_pos - base_f) as f32;
            let idx0 = (base_f as usize).min(total_samples - 1);
            let idx1 = if idx0 + 1 < total_samples {
                idx0 + 1
            } else if looped && loop_len > 0 {
                loop_start
            } else {
                idx0
            };

            let target_byte = (idx0.max(idx1)) >> 1;
            if target_byte + 1 < decoded_bytes {
                dec.reset();
                decoded_bytes = 0;
                prev_pair = [0, 0];
                cur_pair = [0, 0];
            }
            while decoded_bytes <= target_byte && decoded_bytes < packed.len() {
                prev_pair = cur_pair;
                cur_pair = dec.decode_byte_12bit(packed[decoded_bytes]);
                decoded_bytes += 1;
            }

            let fetch_pair = |idx: usize, cur_b: usize, prev_p: [i16; 2], cur_p: [i16; 2]| -> f32 {
                let b = idx >> 1;
                let nib = idx & 1;
                if b + 1 == cur_b {
                    cur_p[nib] as f32
                } else if b + 2 == cur_b {
                    prev_p[nib] as f32
                } else {
                    cur_p[nib] as f32
                }
            };

            let s0 = fetch_pair(idx0, decoded_bytes, prev_pair, cur_pair);
            let s1 = fetch_pair(idx1, decoded_bytes, prev_pair, cur_pair);
            let value = s0 + (s1 - s0) * frac;
            *l += value * gain_l;
            *r += value * gain_r;

            cur_pos += step;
            if looped && loop_len > 0 && cur_pos >= end_f {
                let over = cur_pos - end_f;
                let wraps = floor_f64(over / loop_len_f);
                cur_pos = (loop_start as f64) + (over - wraps * loop_len_f);
            }
        }
        *pos = cur_pos;
    }

    #[inline]
    fn fetch_pcm8(pcm: &[u8], idx: i64, looped: bool, loop_start: usize) -> i8 {
        let len = pcm.len() as i64;
        if idx < 0 {
            return 0;
        }
        if idx < len {
            return pcm[idx as usize] as i8;
        }
        if looped && loop_start < pcm.len() {
            let loop_len = (pcm.len() - loop_start) as i64;
            let wrapped = loop_start as i64 + ((idx - len) % loop_len);
            pcm[wrapped as usize] as i8
        } else {
            0
        }
    }
}
