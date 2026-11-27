//! Clean-room `#![no_std]` 4-bit ADPCM decoder for Full Fat `SampleFormat::Adpcm4`.
//!
//! The engine's ADPCM format packs two 4-bit OKI/Dialogic ADPCM nibbles per byte
//! (high nibble first), XOR-masked with `0xEC`. The 12-bit signed predictor is
//! clamped to `[-2048, +2047]` using the standard 49-entry step table.

/// XOR mask applied to each packed ADPCM byte before nibble extraction.
pub const ADPCM_BYTE_XOR: u8 = 0xEC;

/// Minimum clamped 12-bit predictor value (`-2048`).
pub const ADPCM_PREDICTOR_MIN: i16 = -2048;

/// Maximum clamped 12-bit predictor value (`+2047`).
pub const ADPCM_PREDICTOR_MAX: i16 = 2047;

/// 49-entry OKI/Dialogic ADPCM step-size table.
const STEP_TABLE: [i16; 49] = [
    16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130,
    143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
    876, 963, 1060, 1166, 1282, 1411, 1552,
];

/// Step-index delta table indexed by magnitude bits `0..=7`.
const INDEX_ADJUST: [i8; 8] = [-1, -1, -1, -1, 2, 4, 6, 8];

/// Stateful 4-bit OKI/Dialogic ADPCM decoder matching the engine's ARM mixer path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Adpcm4Decoder {
    predictor: i16,
    step_index: u8,
}

impl Adpcm4Decoder {
    /// Creates a new decoder initialized to predictor `0` and step index `0`.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            predictor: 0,
            step_index: 0,
        }
    }

    /// Resets the decoder state to predictor `0` and step index `0`.
    #[inline]
    pub const fn reset(&mut self) {
        self.predictor = 0;
        self.step_index = 0;
    }

    /// Returns the current 12-bit signed predictor in `[-2048, +2047]`.
    #[inline]
    #[must_use]
    pub const fn predictor(&self) -> i16 {
        self.predictor
    }

    /// Returns the current step table index in `0..=48`.
    #[inline]
    #[must_use]
    pub const fn step_index(&self) -> u8 {
        self.step_index
    }

    /// Decodes a single unmasked 4-bit nibble (`0..=15`) and returns the updated
    /// 12-bit signed predictor (`[-2048, +2047]`).
    #[inline]
    pub fn decode_nibble(&mut self, nibble: u8) -> i16 {
        let idx = (self.step_index as usize).min(STEP_TABLE.len() - 1);
        let step = STEP_TABLE[idx] as i32;
        let mag = (nibble & 0x07) as usize;

        let mut diff = step >> 3;
        if (mag & 1) != 0 {
            diff += step >> 2;
        }
        if (mag & 2) != 0 {
            diff += step >> 1;
        }
        if (mag & 4) != 0 {
            diff += step;
        }
        if (nibble & 0x08) != 0 {
            diff = -diff;
        }

        let pred = (self.predictor as i32 + diff)
            .clamp(ADPCM_PREDICTOR_MIN as i32, ADPCM_PREDICTOR_MAX as i32)
            as i16;
        self.predictor = pred;

        let next_idx = (idx as i32 + INDEX_ADJUST[mag] as i32).clamp(0, 48) as u8;
        self.step_index = next_idx;
        pred
    }

    /// Decodes one packed byte (applying the `0xEC` XOR mask, high nibble first)
    /// into two 12-bit signed predictors (`[-2048, +2047]`).
    #[inline]
    pub fn decode_byte_12bit(&mut self, packed: u8) -> [i16; 2] {
        let unmasked = packed ^ ADPCM_BYTE_XOR;
        let s0 = self.decode_nibble(unmasked >> 4);
        let s1 = self.decode_nibble(unmasked & 0x0F);
        [s0, s1]
    }

    /// Decodes one packed byte into two 16-bit signed linear PCM samples (`predictor << 4`).
    #[inline]
    pub fn decode_byte_pcm16(&mut self, packed: u8) -> [i16; 2] {
        let [s0, s1] = self.decode_byte_12bit(packed);
        [s0 << 4, s1 << 4]
    }

    /// Decodes one packed byte into two mixer-scaled samples (`predictor >> 3`),
    /// matching the ARM mixer's `asr #3` shift before volume multiplication.
    #[inline]
    pub fn decode_byte_mixer(&mut self, packed: u8) -> [i16; 2] {
        let [s0, s1] = self.decode_byte_12bit(packed);
        [s0 >> 3, s1 >> 3]
    }
}
