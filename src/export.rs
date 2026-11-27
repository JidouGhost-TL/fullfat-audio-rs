//! 16-bit PCM WAV encoder and SoundFont 2.01 (`.sf2`) builder (requires `alloc`).

use alloc::vec::Vec;

use crate::adpcm::Adpcm4Decoder;
use crate::parse::{Error, FullFatAudio, Result, Sample, SampleFormat};
use crate::{BASE_SAMPLE_RATE_HZ, ROOT_KEY};

/// Maximum total decoded 16-bit PCM bytes allowed in a single export (32 MiB).
///
/// Prevents memory-amplification attacks via aliased or oversized sample pointers.
pub const MAX_TOTAL_PCM_BYTES: usize = 32 * 1024 * 1024;

/// RIFF WAVE header size in bytes (12-byte RIFF header + 24-byte `fmt ` + 8-byte `data` header).
const WAV_HEADER_BYTES: usize = 44;

/// Byte size of the WAVE format header after the 8-byte RIFF prefix (`4 + 24 + 8`).
const WAV_RIFF_OVERHEAD: u32 = 36;

/// Number of zero-padding samples appended after each sample in the SF2 `smpl` pool.
const SF2_SAMPLE_PAD: u32 = 46;

/// Zero-padding bytes appended after each sample (`46 * 2`).
const SF2_SAMPLE_PAD_BYTES: [u8; 92] = [0u8; 92];

/// SF2 generator opcode for `instrument` (`41`).
const SF2_GEN_INSTRUMENT: u16 = 41;

/// SF2 generator opcode for `sampleID` (`53`).
const SF2_GEN_SAMPLE_ID: u16 = 53;

/// SF2 generator opcode for `sampleModes` (`54`).
const SF2_GEN_SAMPLE_MODES: u16 = 54;

/// Encodes a Full Fat [`Sample`] as a 16-bit signed mono PCM RIFF WAVE file.
///
/// Supports both [`SampleFormat::Pcm8Signed`] and [`SampleFormat::Adpcm4`].
/// Returns [`Error::UnsupportedSampleFormat`] for unrecognized formats and
/// [`Error::AllocationLimitExceeded`] if the decoded payload exceeds [`MAX_TOTAL_PCM_BYTES`].
pub fn export_wav(sample: &Sample<'_>, sample_rate: u32) -> Result<Vec<u8>> {
    let rate = if sample_rate == 0 {
        BASE_SAMPLE_RATE_HZ
    } else {
        sample_rate
    };
    let num_samples = decoded_sample_count(sample)?;
    let data_bytes = num_samples.checked_mul(2).ok_or(Error::SizeOverflow)?;
    if data_bytes > MAX_TOTAL_PCM_BYTES {
        return Err(Error::AllocationLimitExceeded {
            requested: data_bytes,
            limit: MAX_TOTAL_PCM_BYTES,
        });
    }
    let data_bytes_u32 = u32::try_from(data_bytes).map_err(|_| Error::SizeOverflow)?;
    let riff_size = WAV_RIFF_OVERHEAD
        .checked_add(data_bytes_u32)
        .ok_or(Error::SizeOverflow)?;
    let byte_rate = rate.checked_mul(2).ok_or(Error::SizeOverflow)?;
    let total_len = WAV_HEADER_BYTES
        .checked_add(data_bytes)
        .ok_or(Error::SizeOverflow)?;

    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // fmt chunk (PCM, mono, 16-bit)
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // channels = 1 (mono)
    out.extend_from_slice(&rate.to_le_bytes()); // sample rate
    out.extend_from_slice(&byte_rate.to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data chunk
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes_u32.to_le_bytes());
    append_decoded_pcm16(&mut out, sample)?;
    Ok(out)
}

/// Encodes a [`Sample`] as a 16-bit mono WAV at the engine's base sample rate (`11,025 Hz`).
pub fn export_wav_default(sample: &Sample<'_>) -> Result<Vec<u8>> {
    export_wav(sample, BASE_SAMPLE_RATE_HZ)
}

/// Builds a SoundFont 2.01 (`.sf2`) bank containing all samples across all banks in `audio`.
///
/// Each sample is exposed as its own GM preset with monotonic collision-free numbering
/// (`bank = idx / 128`, `program = idx % 128`) at the engine's native base rate (`11,025 Hz`),
/// root key 60 (Middle C), and loop points preserved when `SampleHeader::is_looped()` is true.
///
/// Duplicate sample pointers sharing the same `rom_offset` are deduplicated in the `smpl` pool,
/// and the total `smpl` size is bounded by [`MAX_TOTAL_PCM_BYTES`].
pub fn build_sf2(audio: &FullFatAudio<'_>, bank_name: &str) -> Result<Vec<u8>> {
    struct UniqueSample<'a> {
        rom_offset: u32,
        sample: Sample<'a>,
        start: u32,
        end: u32,
    }

    struct ShdrEntry {
        bank_idx: u16,
        prog_idx: u16,
        start: u32,
        end: u32,
        loop_start: u32,
        loop_end: u32,
        looped: bool,
    }

    let mut unique_samples: Vec<UniqueSample<'_>> = Vec::new();
    let mut entries: Vec<ShdrEntry> = Vec::new();
    let mut cursor_samples: u32 = 0;
    let mut total_smpl_bytes: usize = 0;
    let mut total_igens: usize = 0;

    // Pass 1: Deduplicate sample payloads by rom_offset, validate caps, and collect metadata.
    for (b_idx, bank) in audio.banks().iter().enumerate() {
        for s_idx in 0..(bank.sample_count as usize) {
            let Some(sample) = audio.bank_sample(b_idx, s_idx) else {
                continue;
            };
            let decoded_len = decoded_sample_count(&sample)?;
            if decoded_len == 0 {
                continue;
            }
            let decoded_len_u32 = u32::try_from(decoded_len).map_err(|_| Error::SizeOverflow)?;

            let (start, end) = if let Some(existing) = unique_samples
                .iter()
                .find(|u| u.rom_offset == sample.header.rom_offset)
            {
                (existing.start, existing.end)
            } else {
                let added_bytes = decoded_len
                    .checked_add(SF2_SAMPLE_PAD as usize)
                    .and_then(|n| n.checked_mul(2))
                    .ok_or(Error::SizeOverflow)?;
                let next_smpl_bytes = total_smpl_bytes
                    .checked_add(added_bytes)
                    .ok_or(Error::SizeOverflow)?;
                if next_smpl_bytes > MAX_TOTAL_PCM_BYTES {
                    return Err(Error::AllocationLimitExceeded {
                        requested: next_smpl_bytes,
                        limit: MAX_TOTAL_PCM_BYTES,
                    });
                }
                total_smpl_bytes = next_smpl_bytes;

                let s_start = cursor_samples;
                let s_end = s_start
                    .checked_add(decoded_len_u32)
                    .ok_or(Error::SizeOverflow)?;
                cursor_samples = s_end
                    .checked_add(SF2_SAMPLE_PAD)
                    .ok_or(Error::SizeOverflow)?;

                unique_samples.push(UniqueSample {
                    rom_offset: sample.header.rom_offset,
                    sample,
                    start: s_start,
                    end: s_end,
                });
                (s_start, s_end)
            };

            let (loop_start, loop_end, looped) = if sample.header.is_looped() {
                let loop_off_samples = match sample.header.format {
                    SampleFormat::Adpcm4 => sample
                        .header
                        .loop_start
                        .checked_mul(2)
                        .ok_or(Error::SizeOverflow)?,
                    SampleFormat::Pcm8Signed => sample.header.loop_start,
                    other => return Err(Error::UnsupportedSampleFormat(other)),
                };
                let ls = start
                    .checked_add(loop_off_samples)
                    .ok_or(Error::SizeOverflow)?;
                if ls < end {
                    (ls, end, true)
                } else {
                    (start, end, false)
                }
            } else {
                (start, end, false)
            };

            let global_idx = entries.len();
            let bank_idx = u16::try_from(global_idx / 128).map_err(|_| Error::TableLimitExceeded)?;
            let prog_idx = (global_idx % 128) as u16;
            total_igens = total_igens
                .checked_add(if looped { 2 } else { 1 })
                .ok_or(Error::TableLimitExceeded)?;

            entries.push(ShdrEntry {
                bank_idx,
                prog_idx,
                start,
                end,
                loop_start,
                loop_end,
                looped,
            });
        }
    }

    if entries.is_empty() {
        return Err(Error::EmptyBank);
    }
    if entries.len() >= u16::MAX as usize || total_igens >= u16::MAX as usize {
        return Err(Error::TableLimitExceeded);
    }

    let n = entries.len();
    let count_u16 = n as u16;

    // Pass 2: Precompute exact byte sizes for single-allocation streaming assembly.
    let name = if bank_name.is_empty() {
        "FullFatAudio"
    } else {
        bank_name
    };
    let isng_payload_len = zstr_even_len("EMU8000");
    let inam_payload_len = zstr_even_len(name);
    // INFO body: ifil (8 + 4) + isng (8 + isng_payload_len) + INAM (8 + inam_payload_len)
    let info_body_len = (8usize + 4)
        .checked_add(8 + isng_payload_len)
        .and_then(|v| v.checked_add(8 + inam_payload_len))
        .ok_or(Error::SizeOverflow)?;

    // sdta body: smpl header (8) + total_smpl_bytes (always even)
    let smpl_bytes_u32 = u32::try_from(total_smpl_bytes).map_err(|_| Error::SizeOverflow)?;
    let sdta_body_len = 8usize
        .checked_add(total_smpl_bytes)
        .ok_or(Error::SizeOverflow)?;

    // pdta subchunk payload sizes (all multiples of 2)
    let phdr_len = (n + 1).checked_mul(38).ok_or(Error::SizeOverflow)?;
    let pbag_len = (n + 1).checked_mul(4).ok_or(Error::SizeOverflow)?;
    let pmod_len = 10usize;
    let pgen_len = (n + 1).checked_mul(4).ok_or(Error::SizeOverflow)?;
    let inst_len = (n + 1).checked_mul(22).ok_or(Error::SizeOverflow)?;
    let ibag_len = (n + 1).checked_mul(4).ok_or(Error::SizeOverflow)?;
    let imod_len = 10usize;
    let igen_len = (total_igens + 1)
        .checked_mul(4)
        .ok_or(Error::SizeOverflow)?;
    let shdr_len = (n + 1).checked_mul(46).ok_or(Error::SizeOverflow)?;

    let pdta_body_len = (9usize * 8)
        .checked_add(phdr_len)
        .and_then(|v| v.checked_add(pbag_len))
        .and_then(|v| v.checked_add(pmod_len))
        .and_then(|v| v.checked_add(pgen_len))
        .and_then(|v| v.checked_add(inst_len))
        .and_then(|v| v.checked_add(ibag_len))
        .and_then(|v| v.checked_add(imod_len))
        .and_then(|v| v.checked_add(igen_len))
        .and_then(|v| v.checked_add(shdr_len))
        .ok_or(Error::SizeOverflow)?;

    // Each LIST adds 12 bytes (b"LIST" + u32 size + 4-byte list_type).
    let info_list_size = u32::try_from(4usize.checked_add(info_body_len).ok_or(Error::SizeOverflow)?)
        .map_err(|_| Error::SizeOverflow)?;
    let sdta_list_size = u32::try_from(4usize.checked_add(sdta_body_len).ok_or(Error::SizeOverflow)?)
        .map_err(|_| Error::SizeOverflow)?;
    let pdta_list_size = u32::try_from(4usize.checked_add(pdta_body_len).ok_or(Error::SizeOverflow)?)
        .map_err(|_| Error::SizeOverflow)?;

    let sfbk_len = 4usize
        .checked_add(12 + info_body_len)
        .and_then(|v| v.checked_add(12 + sdta_body_len))
        .and_then(|v| v.checked_add(12 + pdta_body_len))
        .ok_or(Error::SizeOverflow)?;
    let sfbk_len_u32 = u32::try_from(sfbk_len).map_err(|_| Error::SizeOverflow)?;
    let total_file_len = 8usize.checked_add(sfbk_len).ok_or(Error::SizeOverflow)?;

    // Pass 3: Single allocation — write RIFF sfbk directly.
    let mut out = Vec::with_capacity(total_file_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&sfbk_len_u32.to_le_bytes());
    out.extend_from_slice(b"sfbk");

    // INFO LIST
    out.extend_from_slice(b"LIST");
    out.extend_from_slice(&info_list_size.to_le_bytes());
    out.extend_from_slice(b"INFO");
    push_subchunk(&mut out, b"ifil", &[2, 0, 1, 0]);
    push_zstr_subchunk(&mut out, b"isng", "EMU8000");
    push_zstr_subchunk(&mut out, b"INAM", name);

    // sdta LIST -> smpl subchunk (written directly, zero extra copies)
    out.extend_from_slice(b"LIST");
    out.extend_from_slice(&sdta_list_size.to_le_bytes());
    out.extend_from_slice(b"sdta");
    out.extend_from_slice(b"smpl");
    out.extend_from_slice(&smpl_bytes_u32.to_le_bytes());
    for u in &unique_samples {
        append_decoded_pcm16(&mut out, &u.sample)?;
        out.extend_from_slice(&SF2_SAMPLE_PAD_BYTES);
    }

    // pdta LIST
    out.extend_from_slice(b"LIST");
    out.extend_from_slice(&pdta_list_size.to_le_bytes());
    out.extend_from_slice(b"pdta");

    // phdr
    write_chunk_header(&mut out, b"phdr", phdr_len)?;
    for (idx, e) in entries.iter().enumerate() {
        let idx_u16 = idx as u16;
        push_name20(&mut out, b"P", idx);
        out.extend_from_slice(&e.prog_idx.to_le_bytes());
        out.extend_from_slice(&e.bank_idx.to_le_bytes());
        out.extend_from_slice(&idx_u16.to_le_bytes());
        out.extend_from_slice(&[0u8; 12]);
    }
    push_fixed20(&mut out, b"EOP");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&count_u16.to_le_bytes());
    out.extend_from_slice(&[0u8; 12]);

    // pbag
    write_chunk_header(&mut out, b"pbag", pbag_len)?;
    for idx in 0..n {
        out.extend_from_slice(&(idx as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out.extend_from_slice(&count_u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());

    // pmod
    write_chunk_header(&mut out, b"pmod", pmod_len)?;
    out.extend_from_slice(&[0u8; 10]);

    // pgen
    write_chunk_header(&mut out, b"pgen", pgen_len)?;
    for idx in 0..n {
        out.extend_from_slice(&SF2_GEN_INSTRUMENT.to_le_bytes());
        out.extend_from_slice(&(idx as u16).to_le_bytes());
    }
    out.extend_from_slice(&[0u8; 4]);

    // inst
    write_chunk_header(&mut out, b"inst", inst_len)?;
    for idx in 0..n {
        push_name20(&mut out, b"I", idx);
        out.extend_from_slice(&(idx as u16).to_le_bytes());
    }
    push_fixed20(&mut out, b"EOI");
    out.extend_from_slice(&count_u16.to_le_bytes());

    // ibag
    write_chunk_header(&mut out, b"ibag", ibag_len)?;
    let mut igen_idx: u16 = 0;
    for e in &entries {
        out.extend_from_slice(&igen_idx.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        igen_idx += if e.looped { 2 } else { 1 };
    }
    out.extend_from_slice(&igen_idx.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());

    // imod
    write_chunk_header(&mut out, b"imod", imod_len)?;
    out.extend_from_slice(&[0u8; 10]);

    // igen
    write_chunk_header(&mut out, b"igen", igen_len)?;
    for (idx, e) in entries.iter().enumerate() {
        if e.looped {
            out.extend_from_slice(&SF2_GEN_SAMPLE_MODES.to_le_bytes());
            out.extend_from_slice(&1u16.to_le_bytes());
        }
        out.extend_from_slice(&SF2_GEN_SAMPLE_ID.to_le_bytes());
        out.extend_from_slice(&(idx as u16).to_le_bytes());
    }
    out.extend_from_slice(&[0u8; 4]);

    // shdr
    write_chunk_header(&mut out, b"shdr", shdr_len)?;
    for (idx, e) in entries.iter().enumerate() {
        push_name20(&mut out, b"S", idx);
        out.extend_from_slice(&e.start.to_le_bytes());
        out.extend_from_slice(&e.end.to_le_bytes());
        out.extend_from_slice(&e.loop_start.to_le_bytes());
        out.extend_from_slice(&e.loop_end.to_le_bytes());
        out.extend_from_slice(&BASE_SAMPLE_RATE_HZ.to_le_bytes());
        out.push(ROOT_KEY);
        out.push(0); // chPitchCorrection
        out.extend_from_slice(&0u16.to_le_bytes()); // wSampleLink
        out.extend_from_slice(&1u16.to_le_bytes()); // sfSampleType = monoSample
    }
    push_fixed20(&mut out, b"EOS");
    out.extend_from_slice(&[0u8; 26]);

    Ok(out)
}

#[inline]
fn decoded_sample_count(sample: &Sample<'_>) -> Result<usize> {
    match sample.header.format {
        SampleFormat::Pcm8Signed => Ok(sample.pcm.len()),
        SampleFormat::Adpcm4 => sample.pcm.len().checked_mul(2).ok_or(Error::SizeOverflow),
        other => Err(Error::UnsupportedSampleFormat(other)),
    }
}

fn append_decoded_pcm16(out: &mut Vec<u8>, sample: &Sample<'_>) -> Result<()> {
    match sample.header.format {
        SampleFormat::Pcm8Signed => {
            let base = out.len();
            let add_bytes = sample.pcm.len().checked_mul(2).ok_or(Error::SizeOverflow)?;
            out.resize(base + add_bytes, 0);
            let (pairs, _) = out[base..].as_chunks_mut::<2>();
            for (dst, &src) in pairs.iter_mut().zip(sample.pcm) {
                dst[0] = 0;
                dst[1] = src;
            }
            Ok(())
        }
        SampleFormat::Adpcm4 => {
            let base = out.len();
            let add_bytes = sample.pcm.len().checked_mul(4).ok_or(Error::SizeOverflow)?;
            out.resize(base + add_bytes, 0);
            let mut dec = Adpcm4Decoder::new();
            let (quads, _) = out[base..].as_chunks_mut::<4>();
            for (dst, &packed) in quads.iter_mut().zip(sample.pcm) {
                let [s0, s1] = dec.decode_byte_pcm16(packed);
                let b0 = s0.to_le_bytes();
                let b1 = s1.to_le_bytes();
                dst[0] = b0[0];
                dst[1] = b0[1];
                dst[2] = b1[0];
                dst[3] = b1[1];
            }
            Ok(())
        }
        other => Err(Error::UnsupportedSampleFormat(other)),
    }
}

#[inline]
fn write_chunk_header(out: &mut Vec<u8>, tag: &[u8; 4], len: usize) -> Result<()> {
    let len_u32 = u32::try_from(len).map_err(|_| Error::SizeOverflow)?;
    out.extend_from_slice(tag);
    out.extend_from_slice(&len_u32.to_le_bytes());
    Ok(())
}

#[inline]
fn zstr_even_len(s: &str) -> usize {
    let with_nul = s.len() + 1;
    with_nul + (with_nul & 1)
}

fn push_subchunk(out: &mut Vec<u8>, tag: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(tag);
    let len_u32 = u32::try_from(payload.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len_u32.to_le_bytes());
    out.extend_from_slice(payload);
    if (payload.len() & 1) != 0 {
        out.push(0);
    }
}

fn push_zstr_subchunk(out: &mut Vec<u8>, tag: &[u8; 4], s: &str) {
    let even_len = zstr_even_len(s);
    out.extend_from_slice(tag);
    let len_u32 = u32::try_from(even_len).unwrap_or(u32::MAX);
    out.extend_from_slice(&len_u32.to_le_bytes());
    out.extend_from_slice(s.as_bytes());
    out.push(0);
    if ((s.len() + 1) & 1) != 0 {
        out.push(0);
    }
}

fn push_fixed20(out: &mut Vec<u8>, s: &[u8]) {
    let mut buf = [0u8; 20];
    let n = s.len().min(19);
    buf[..n].copy_from_slice(&s[..n]);
    out.extend_from_slice(&buf);
}

fn push_name20(out: &mut Vec<u8>, prefix: &[u8], idx: usize) {
    let mut buf = [0u8; 20];
    let p_len = prefix.len().min(4);
    buf[..p_len].copy_from_slice(&prefix[..p_len]);
    let mut tmp = [0u8; 20];
    let mut v = idx;
    let mut d = 0usize;
    if v == 0 {
        tmp[0] = b'0';
        d = 1;
    } else {
        while v > 0 && d < tmp.len() {
            tmp[d] = b'0' + (v % 10) as u8;
            v /= 10;
            d += 1;
        }
    }
    let mut pos = p_len;
    while d > 0 && pos < 19 {
        d -= 1;
        buf[pos] = tmp[d];
        pos += 1;
    }
    out.extend_from_slice(&buf);
}
