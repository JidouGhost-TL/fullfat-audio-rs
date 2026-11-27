//! Binary signature detection for the Full Fat GBA Mono DirectSound audio engine.

/// Size in bytes of the 128-entry 16.16 fixed-point MIDI frequency table.
pub const FREQ_TABLE_BYTES: usize = 512;

/// MIDI note number for A4 (440.0 Hz).
const MIDI_NOTE_A4: usize = 69;

/// Exact 16.16 fixed-point representation of 440.0 Hz (`440 << 16`).
const FREQ_A4_Q16: u32 = 28_835_840;

/// Search window in bytes after `MixChannel` for the `FinalizeBuffer` clamp loop.
const FINALIZE_SEARCH_WINDOW: usize = 0x400;

/// GBA master clock frequency used by Rev A Timer 0 reload calculation (`0x01000AE0`).
pub const REVA_CLOCK_HZ: u32 = 16_780_000;

/// GBA master clock frequency used by Rev B Timer 0 reload calculation (`0x01000000` = 2^24).
pub const REVB_CLOCK_HZ: u32 = 16_777_216;

/// Detected engine build revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Revision {
    /// Early build revision: `FinalizeBuffer` uses `bpl` loop branch (`0x5AFFFFF6`) and
    /// Timer 0 reload uses `16,780,000 Hz` (`0x01000AE0`).
    RevA,
    /// Later build revision: `FinalizeBuffer` uses `bgt` loop branch (`0xCAFFFFF6`) and
    /// Timer 0 reload uses `16,777,216 Hz` (`0x01000000`).
    RevB,
    /// Core mixer and frequency table matched, but finalize loop revision was not classified.
    Unknown,
}

/// Alias matching the standard `*Version` naming convention across GBA synth crates.
pub type FullFatVersion = Revision;

impl Revision {
    /// Detects the Full Fat engine revision directly from a GBA ROM image.
    #[must_use]
    pub fn detect_from_bytes(rom: &[u8]) -> Option<Self> {
        detect_details(rom).map(|d| d.revision)
    }

    /// Returns the CPU clock frequency constant used by this revision's Timer 0 reload formula.
    #[inline]
    #[must_use]
    pub const fn clock_hz(self) -> u32 {
        match self {
            Self::RevA => REVA_CLOCK_HZ,
            Self::RevB | Self::Unknown => REVB_CLOCK_HZ,
        }
    }

    /// Computes the 16-bit Timer 0 reload value (`0x10000 - clock_hz / rate`) for `sample_rate`.
    ///
    /// Returns `0` if `sample_rate == 0`.
    #[inline]
    #[must_use]
    pub const fn timer0_reload(self, sample_rate: u32) -> u16 {
        if sample_rate == 0 {
            return 0;
        }
        let ticks = self.clock_hz() / sample_rate;
        0x10000u32.wrapping_sub(ticks) as u16
    }
}

/// Detailed detection metadata returned by [`detect_details`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Detection {
    /// Detected engine build revision.
    pub revision: Revision,
    /// ROM file offset of the ARM IWRAM `MixChannel` inner loop.
    pub mixer_offset: u32,
    /// ROM file offset of the 128-entry 16.16 fixed-point MIDI frequency table.
    pub freq_table_offset: u32,
}

/// 32-byte ARM IWRAM `MixChannel` inner mixing loop:
/// `ldrsb r0, [r8]; ldrsh r1, [r2]; adds r3, r3, r4, lsl #4; adc r8, r8, r4, lsr #28;`
/// `mla r1, r0, sb, r1; strh r1, [r2], #2; subs r6, r6, #1; bpl -9`
pub const MIX_LOOP_SIG: [u8; 32] = [
    0xD0, 0x00, 0xD8, 0xE1, // ldrsb r0, [r8]
    0xF0, 0x10, 0xD2, 0xE1, // ldrsh r1, [r2]
    0x04, 0x32, 0x93, 0xE0, // adds  r3, r3, r4, lsl #4
    0x24, 0x8E, 0xA8, 0xE0, // adc   r8, r8, r4, lsr #28
    0x90, 0x19, 0x21, 0xE0, // mla   r1, r0, sb, r1
    0xB2, 0x10, 0xC2, 0xE0, // strh  r1, [r2], #2
    0x01, 0x60, 0x56, 0xE2, // subs  r6, r6, #1
    0xF7, 0xFF, 0xFF, 0x5A, // bpl   #-36
];

/// 32-byte prefix (notes 0..7) of the 128-entry 16.16 fixed-point MIDI frequency table
/// (`A4 = note 69 = 440.0 * 65536 = 28,835,840`).
pub const FREQ_TABLE_PREFIX: [u8; 32] = [
    0x01, 0x2D, 0x08, 0x00, // note 0: 535,809 (~8.1758 Hz)
    0x76, 0xA9, 0x08, 0x00, // note 1: 567,670
    0x51, 0x2D, 0x09, 0x00, // note 2: 601,425
    0x04, 0xB9, 0x09, 0x00, // note 3: 637,188
    0x05, 0x4D, 0x0A, 0x00, // note 4: 675,077
    0xD3, 0xE9, 0x0A, 0x00, // note 5: 715,219
    0xF4, 0x8F, 0x0B, 0x00, // note 6: 757,748
    0xF6, 0x3F, 0x0C, 0x00, // note 7: 802,806
];

/// 36-byte ARM `FinalizeBuffer` clamp loop (Rev A: `bpl`).
const FINALIZE_LOOP_REVA: [u8; 36] = [
    0xF2, 0x00, 0xD2, 0xE0, // ldrsh r0, [r2], #2
    0x40, 0x02, 0xA0, 0xE1, // asr   r0, r0, #4
    0x80, 0x00, 0x70, 0xE3, // cmn   r0, #0x80
    0x7F, 0x00, 0xE0, 0xB3, // mvnlt r0, #0x7f
    0x7F, 0x00, 0x50, 0xE3, // cmp   r0, #0x7f
    0x7F, 0x00, 0xA0, 0xC3, // movgt r0, #0x7f
    0x01, 0x00, 0xCE, 0xE4, // strb  r0, [lr], #1
    0x01, 0xC0, 0x5C, 0xE2, // subs  ip, ip, #1
    0xF6, 0xFF, 0xFF, 0x5A, // bpl   #-40
];

/// 36-byte ARM `FinalizeBuffer` clamp loop (Rev B: `bgt`).
const FINALIZE_LOOP_REVB: [u8; 36] = [
    0xF2, 0x00, 0xD2, 0xE0, // ldrsh r0, [r2], #2
    0x40, 0x02, 0xA0, 0xE1, // asr   r0, r0, #4
    0x80, 0x00, 0x70, 0xE3, // cmn   r0, #0x80
    0x7F, 0x00, 0xE0, 0xB3, // mvnlt r0, #0x7f
    0x7F, 0x00, 0x50, 0xE3, // cmp   r0, #0x7f
    0x7F, 0x00, 0xA0, 0xC3, // movgt r0, #0x7f
    0x01, 0x00, 0xCE, 0xE4, // strb  r0, [lr], #1
    0x01, 0xC0, 0x5C, 0xE2, // subs  ip, ip, #1
    0xF6, 0xFF, 0xFF, 0xCA, // bgt   #-40
];

/// Returns `true` if `rom` contains the Full Fat Mono DirectSound audio engine.
#[must_use]
pub fn detect(rom: &[u8]) -> bool {
    detect_details(rom).is_some()
}

/// Detects the Full Fat Mono DirectSound engine and returns its revision and binary offsets.
#[must_use]
pub fn detect_details(rom: &[u8]) -> Option<Detection> {
    if rom.len() < FREQ_TABLE_BYTES {
        return None;
    }

    let mixer_offset = find_aligned4(rom, &MIX_LOOP_SIG)?;
    let freq_table_offset = find_aligned4(rom, &FREQ_TABLE_PREFIX)?;

    // Verify the full 128-entry (512-byte) frequency table is in-bounds and note 69 == 440 << 16.
    let ft_end = freq_table_offset.checked_add(FREQ_TABLE_BYTES)?;
    if ft_end > rom.len() {
        return None;
    }
    let a4_off = freq_table_offset.checked_add(MIDI_NOTE_A4 * 4)?;
    let a4_val = read_u32_le(rom, a4_off)?;
    if a4_val != FREQ_A4_Q16 {
        return None;
    }

    // Classify revision from FinalizeBuffer loop within FINALIZE_SEARCH_WINDOW bytes after MixChannel.
    let search_start = mixer_offset;
    let search_end = mixer_offset
        .saturating_add(FINALIZE_SEARCH_WINDOW)
        .min(rom.len());
    let window = rom.get(search_start..search_end).unwrap_or(&[]);
    let revision = if find_aligned4(window, &FINALIZE_LOOP_REVB).is_some() {
        Revision::RevB
    } else if find_aligned4(window, &FINALIZE_LOOP_REVA).is_some() {
        Revision::RevA
    } else {
        Revision::Unknown
    };

    Some(Detection {
        revision,
        mixer_offset: u32::try_from(mixer_offset).ok()?,
        freq_table_offset: u32::try_from(freq_table_offset).ok()?,
    })
}

/// Reads a little-endian `u32` at byte offset `off`, returning `None` if out of bounds.
#[inline]
pub(crate) fn read_u32_le(data: &[u8], off: usize) -> Option<u32> {
    let b = data.get(off..off.checked_add(4)?)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Searches `haystack` on 4-byte boundaries for `needle` (must be at least 4 bytes).
#[inline]
pub(crate) fn find_aligned4(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.len() < 4 || haystack.len() < needle.len() {
        return None;
    }
    let first = u32::from_ne_bytes([needle[0], needle[1], needle[2], needle[3]]);
    let max_words = (haystack.len() - needle.len()) / 4 + 1;
    let (words, _) = haystack.as_chunks::<4>();
    for (idx, &word) in words.iter().take(max_words).enumerate() {
        if u32::from_ne_bytes(word) == first {
            let off = idx * 4;
            if &haystack[off..off + needle.len()] == needle {
                return Some(off);
            }
        }
    }
    None
}
