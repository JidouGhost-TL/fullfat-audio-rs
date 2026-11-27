//! Clean-room `#![no_std]` parser for Full Fat sound banks, song tables, and PCM sample headers.

use core::fmt;

use crate::detect::{detect_details, read_u32_le, Revision, FREQ_TABLE_BYTES};
use crate::GBA_ROM_BASE;

/// Maximum number of distinct sound banks tracked per ROM in `#![no_std]` mode.
pub const MAX_BANKS: usize = 16;

/// Maximum GBA ROM image size in bytes (32 MiB).
pub const MAX_GBA_ROM_BYTES: usize = 0x0200_0000;

/// Maximum plausible song count per bank.
const MAX_SONGS_PER_BANK: u32 = 256;

/// Maximum plausible sample count per bank.
const MAX_SAMPLES_PER_BANK: u32 = 2048;

/// Maximum plausible single sample length in bytes (4 MiB).
const MAX_SAMPLE_LEN: u32 = 0x0040_0000;

/// Convenience result alias for `fullfat-audio`.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors returned by parsing and export operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Error {
    /// ROM does not match the Full Fat Mono DirectSound engine signature.
    NotDetected,
    /// ROM is too small or exceeds the 32 MiB GBA ROM address space.
    InvalidRom,
    /// Engine was detected, but no valid sound bank structures were found.
    NoBanksFound,
    /// Sample encoding format is not supported for this operation.
    UnsupportedSampleFormat(SampleFormat),
    /// Sound bank contains no exportable samples.
    EmptyBank,
    /// Total decoded PCM bytes would exceed the configured safety limit.
    AllocationLimitExceeded {
        /// Requested PCM byte count.
        requested: usize,
        /// Hard upper limit in bytes.
        limit: usize,
    },
    /// SoundFont table index or generator count exceeds the 16-bit SF2 limit.
    TableLimitExceeded,
    /// RIFF/SF2 chunk size exceeds 32-bit container bounds.
    SizeOverflow,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotDetected => write!(f, "Full Fat audio engine signature not detected"),
            Self::InvalidRom => {
                write!(f, "ROM size outside valid GBA range (512 B ..= 32 MiB)")
            }
            Self::NoBanksFound => write!(f, "no valid Full Fat sound banks found in ROM"),
            Self::UnsupportedSampleFormat(fmt_tag) => {
                write!(f, "unsupported Full Fat sample format: {fmt_tag:?}")
            }
            Self::EmptyBank => write!(f, "sound bank contains no exportable samples"),
            Self::AllocationLimitExceeded { requested, limit } => write!(
                f,
                "PCM export size ({requested} bytes) exceeds safety limit ({limit} bytes)"
            ),
            Self::TableLimitExceeded => {
                write!(f, "SoundFont table entry or generator count exceeds u16 limit")
            }
            Self::SizeOverflow => write!(f, "RIFF container size overflowed u32 bounds"),
        }
    }
}

impl core::error::Error for Error {}

/// Sample encoding tag stored at offset `+0x00` of a 16-byte sample header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SampleFormat {
    /// 4-bit OKI/Dialogic ADPCM format (`0`, two samples per byte).
    Adpcm4,
    /// Signed 8-bit linear PCM (`1`, one sample per byte).
    Pcm8Signed,
    /// Unrecognized format tag.
    Unknown(u32),
}

impl SampleFormat {
    /// Converts a raw 32-bit format tag into a [`SampleFormat`].
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        match raw {
            0 => Self::Adpcm4,
            1 => Self::Pcm8Signed,
            other => Self::Unknown(other),
        }
    }
}

/// 16-byte Full Fat sample header preceding the raw sample payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SampleHeader {
    /// File offset of the 16-byte header inside the ROM.
    pub rom_offset: u32,
    /// Sample encoding format (`+0x00`).
    pub format: SampleFormat,
    /// Sample payload length in bytes (`+0x04`).
    pub length: u32,
    /// Loop start offset in bytes from the start of the payload (`+0x08`).
    /// `0` indicates a one-shot (non-looping) sample.
    pub loop_start: u32,
    /// Reserved field (`+0x0C`, always `0` in valid banks).
    pub reserved: u32,
}

impl SampleHeader {
    /// Returns `true` if this sample loops (`loop_start != 0 && loop_start < length`).
    #[must_use]
    pub const fn is_looped(&self) -> bool {
        self.loop_start != 0 && self.loop_start < self.length
    }
}

/// Borrowed sample header and raw payload slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sample<'a> {
    /// Parsed 16-byte header.
    pub header: SampleHeader,
    /// Raw payload bytes (signed 8-bit PCM for [`SampleFormat::Pcm8Signed`], or packed 4-bit
    /// nibbles for [`SampleFormat::Adpcm4`]).
    pub pcm: &'a [u8],
}

/// 8-byte song table entry (`start_ptr`, `loop_ptr`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SongEntry {
    /// ROM file offset of the sequence start.
    pub start_offset: u32,
    /// ROM file offset of the sequence loop target (`0` if unspecified).
    pub loop_offset: u32,
}

/// Parsed 16-byte `SoundBank` header summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BankSummary {
    /// ROM file offset of the 16-byte bank header.
    pub rom_offset: u32,
    /// Number of song entries (`+0x00`).
    pub song_count: u32,
    /// Number of sample entries (`+0x04`).
    pub sample_count: u32,
    /// ROM file offset of the song table (`+0x08`, equals `rom_offset + 0x10`).
    pub song_table_offset: u32,
    /// ROM file offset of the sample pointer table (`+0x0C`, equals `song_table_offset + song_count * 8`).
    pub sample_table_offset: u32,
}

impl BankSummary {
    const EMPTY: Self = Self {
        rom_offset: 0,
        song_count: 0,
        sample_count: 0,
        song_table_offset: 0,
        sample_table_offset: 0,
    };
}

/// Parsed Full Fat audio engine view over a borrowed GBA ROM image.
#[derive(Debug, Clone)]
pub struct FullFatAudio<'a> {
    rom: &'a [u8],
    revision: Revision,
    mixer_offset: u32,
    freq_table_offset: u32,
    banks: [BankSummary; MAX_BANKS],
    bank_count: usize,
}

impl<'a> FullFatAudio<'a> {
    /// Returns the detected engine build revision.
    #[must_use]
    pub const fn revision(&self) -> Revision {
        self.revision
    }

    /// Returns the ROM file offset of the ARM `MixChannel` inner loop.
    #[must_use]
    pub const fn mixer_offset(&self) -> u32 {
        self.mixer_offset
    }

    /// Returns the ROM file offset of the 128-entry 16.16 MIDI frequency table.
    #[must_use]
    pub const fn freq_table_offset(&self) -> u32 {
        self.freq_table_offset
    }

    /// Returns the slice of discovered sound banks.
    #[must_use]
    pub fn banks(&self) -> &[BankSummary] {
        &self.banks[..self.bank_count]
    }

    /// Returns the total number of songs across all discovered sound banks.
    #[must_use]
    pub fn total_songs(&self) -> usize {
        let mut total = 0usize;
        for bank in self.banks() {
            total = total.saturating_add(bank.song_count as usize);
        }
        total
    }

    /// Returns the total number of samples across all discovered sound banks.
    #[must_use]
    pub fn total_samples(&self) -> usize {
        let mut total = 0usize;
        for bank in self.banks() {
            total = total.saturating_add(bank.sample_count as usize);
        }
        total
    }

    /// Returns song `song_idx` from sound bank `bank_idx`.
    #[must_use]
    pub fn bank_song(&self, bank_idx: usize, song_idx: usize) -> Option<SongEntry> {
        let bank = self.banks().get(bank_idx)?;
        if song_idx >= bank.song_count as usize {
            return None;
        }
        let off = (bank.song_table_offset as usize).checked_add(song_idx.checked_mul(8)?)?;
        let start_ptr = read_u32_le(self.rom, off)?;
        let loop_ptr = read_u32_le(self.rom, off.checked_add(4)?)?;
        let start_offset = rom_ptr_to_offset(start_ptr, self.rom.len())?;
        let loop_offset = if loop_ptr == 0 {
            0
        } else {
            rom_ptr_to_offset(loop_ptr, self.rom.len())?
        };
        Some(SongEntry {
            start_offset,
            loop_offset,
        })
    }

    /// Returns the raw bytecode slice starting at `song.start_offset` through the end of the ROM.
    #[must_use]
    pub fn song_bytes(&self, song: SongEntry) -> Option<&'a [u8]> {
        self.rom.get(song.start_offset as usize..)
    }

    /// Returns sample `sample_idx` from sound bank `bank_idx`.
    #[must_use]
    pub fn bank_sample(&self, bank_idx: usize, sample_idx: usize) -> Option<Sample<'a>> {
        let bank = self.banks().get(bank_idx)?;
        if sample_idx >= bank.sample_count as usize {
            return None;
        }
        sample_at_table_index(self.rom, bank.sample_table_offset as usize, sample_idx)
    }

    /// Returns a lightweight zero-allocation iterator over all samples in sound bank `bank_idx`.
    #[must_use]
    pub fn bank_samples(&self, bank_idx: usize) -> BankSampleIter<'a> {
        match self.banks().get(bank_idx) {
            Some(bank) => BankSampleIter {
                rom: self.rom,
                sample_table_offset: bank.sample_table_offset as usize,
                next_idx: 0,
                count: bank.sample_count as usize,
            },
            None => BankSampleIter {
                rom: self.rom,
                sample_table_offset: 0,
                next_idx: 0,
                count: 0,
            },
        }
    }

    /// Returns the borrowed ROM slice.
    #[must_use]
    pub const fn rom(&self) -> &'a [u8] {
        self.rom
    }
}

/// Zero-allocation iterator over samples in a single sound bank.
#[derive(Debug, Clone, Copy)]
pub struct BankSampleIter<'a> {
    rom: &'a [u8],
    sample_table_offset: usize,
    next_idx: usize,
    count: usize,
}

impl<'a> Iterator for BankSampleIter<'a> {
    type Item = Sample<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.next_idx < self.count {
            let idx = self.next_idx;
            self.next_idx += 1;
            if let Some(sample) = sample_at_table_index(self.rom, self.sample_table_offset, idx) {
                return Some(sample);
            }
        }
        None
    }
}

impl core::iter::FusedIterator for BankSampleIter<'_> {}

#[inline]
fn sample_at_table_index(rom: &[u8], sample_table_offset: usize, idx: usize) -> Option<Sample<'_>> {
    let ptr_off = sample_table_offset.checked_add(idx.checked_mul(4)?)?;
    let samp_ptr = read_u32_le(rom, ptr_off)?;
    let samp_off = rom_ptr_to_offset(samp_ptr, rom.len())? as usize;
    parse_sample_at(rom, samp_off)
}

/// Detects and parses all Full Fat sound banks in `rom`.
pub fn parse(rom: &[u8]) -> Result<FullFatAudio<'_>> {
    if rom.len() < FREQ_TABLE_BYTES || rom.len() > MAX_GBA_ROM_BYTES {
        return Err(Error::InvalidRom);
    }

    let det = detect_details(rom).ok_or(Error::NotDetected)?;

    let mut banks = [BankSummary::EMPTY; MAX_BANKS];
    let mut bank_count = 0usize;

    let mut off = 0usize;
    let limit = rom.len().saturating_sub(16);
    while off <= limit && bank_count < MAX_BANKS {
        if let Some(summary) = try_parse_bank(rom, off) {
            banks[bank_count] = summary;
            bank_count += 1;
            // Advance past the bank header + tables to avoid re-scanning inside them.
            let skip = 16usize
                .saturating_add((summary.song_count as usize).saturating_mul(8))
                .saturating_add((summary.sample_count as usize).saturating_mul(4));
            off = off.saturating_add(skip.max(4));
            continue;
        }
        off += 4;
    }

    if bank_count == 0 {
        return Err(Error::NoBanksFound);
    }

    Ok(FullFatAudio {
        rom,
        revision: det.revision,
        mixer_offset: det.mixer_offset,
        freq_table_offset: det.freq_table_offset,
        banks,
        bank_count,
    })
}

fn try_parse_bank(rom: &[u8], off: usize) -> Option<BankSummary> {
    // Fast-path reject: `song_table_ptr` at `+0x08` must equal `GBA_ROM_BASE + off + 16`.
    // Checking this single 32-bit equality first rejects >99.9999999% of offsets immediately.
    let bank_addr = GBA_ROM_BASE.checked_add(u32::try_from(off).ok()?)?;
    let expected_st = bank_addr.checked_add(16)?;
    let song_table_ptr = read_u32_le(rom, off + 8)?;
    if song_table_ptr != expected_st {
        return None;
    }

    let song_count = read_u32_le(rom, off)?;
    let sample_count = read_u32_le(rom, off + 4)?;
    if song_count > MAX_SONGS_PER_BANK || sample_count > MAX_SAMPLES_PER_BANK {
        return None;
    }
    if song_count == 0 && sample_count == 0 {
        return None;
    }

    let sample_table_ptr = read_u32_le(rom, off + 12)?;
    let expected_nt = expected_st.checked_add(song_count.checked_mul(8)?)?;
    if sample_table_ptr != expected_nt {
        return None;
    }

    let st_off = (song_table_ptr - GBA_ROM_BASE) as usize;
    let nt_off = (sample_table_ptr - GBA_ROM_BASE) as usize;

    // Validate all song table entries.
    let st_bytes = (song_count as usize).checked_mul(8)?;
    if st_off.checked_add(st_bytes)? > rom.len() {
        return None;
    }
    for i in 0..(song_count as usize) {
        let e_off = st_off + i * 8;
        let start_ptr = read_u32_le(rom, e_off)?;
        let loop_ptr = read_u32_le(rom, e_off + 4)?;
        rom_ptr_to_offset(start_ptr, rom.len())?;
        if loop_ptr != 0 {
            rom_ptr_to_offset(loop_ptr, rom.len())?;
        }
    }

    // Validate all sample pointers and sample headers.
    let nt_bytes = (sample_count as usize).checked_mul(4)?;
    if nt_off.checked_add(nt_bytes)? > rom.len() {
        return None;
    }
    for i in 0..(sample_count as usize) {
        let samp_ptr = read_u32_le(rom, nt_off + i * 4)?;
        if (samp_ptr & 3) != 0 {
            return None;
        }
        let samp_off = rom_ptr_to_offset(samp_ptr, rom.len())? as usize;
        let sample = parse_sample_at(rom, samp_off)?;
        match sample.header.format {
            SampleFormat::Adpcm4 | SampleFormat::Pcm8Signed => {}
            SampleFormat::Unknown(_) => return None,
        }
        if sample.header.reserved != 0 {
            return None;
        }
    }

    Some(BankSummary {
        rom_offset: u32::try_from(off).ok()?,
        song_count,
        sample_count,
        song_table_offset: u32::try_from(st_off).ok()?,
        sample_table_offset: u32::try_from(nt_off).ok()?,
    })
}

fn parse_sample_at(rom: &[u8], off: usize) -> Option<Sample<'_>> {
    if off.checked_add(16)? > rom.len() {
        return None;
    }
    let fmt_raw = read_u32_le(rom, off)?;
    let length = read_u32_le(rom, off + 4)?;
    let loop_start = read_u32_le(rom, off + 8)?;
    let reserved = read_u32_le(rom, off + 12)?;

    if length == 0 || length > MAX_SAMPLE_LEN {
        return None;
    }
    if loop_start >= length {
        return None;
    }
    let pcm_start = off.checked_add(16)?;
    let pcm_end = pcm_start.checked_add(length as usize)?;
    let pcm = rom.get(pcm_start..pcm_end)?;

    Some(Sample {
        header: SampleHeader {
            rom_offset: u32::try_from(off).ok()?,
            format: SampleFormat::from_raw(fmt_raw),
            length,
            loop_start,
            reserved,
        },
        pcm,
    })
}

#[inline]
fn rom_ptr_to_offset(ptr: u32, rom_len: usize) -> Option<u32> {
    if ptr < GBA_ROM_BASE {
        return None;
    }
    let off = ptr - GBA_ROM_BASE;
    if (off as usize) < rom_len {
        Some(off)
    } else {
        None
    }
}
