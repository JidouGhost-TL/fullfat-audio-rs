use fullfat_audio::detect::{FREQ_TABLE_PREFIX, MIX_LOOP_SIG};
use fullfat_audio::{
    compute_channel_gain, compute_pitch_step, detect, detect_details, finalize_sample_dma8,
    note_to_playback_hz, parse, AccuracySynth, Adpcm4Decoder, Error, FullFatVersion, FullFatVoice,
    Revision, Sample, SampleFormat, SampleHeader, SeqEvent, SeqIter, StudioSynth,
    BASE_SAMPLE_RATE_HZ, FREQ_TABLE_Q16, GBA_ROM_BASE, MAX_BANKS,
};

fn build_synthetic_rom(rev_b: bool, with_song: bool) -> [u8; 4096] {
    let mut rom = [0u8; 4096];

    // 1. Place MixChannel inner loop at 0x100
    rom[0x100..0x120].copy_from_slice(&MIX_LOOP_SIG);

    // 2. Place FinalizeBuffer clamp loop at 0x200
    let mut fin = [
        0xF2, 0x00, 0xD2, 0xE0, 0x40, 0x02, 0xA0, 0xE1, 0x80, 0x00, 0x70, 0xE3, 0x7F, 0x00, 0xE0,
        0xB3, 0x7F, 0x00, 0x50, 0xE3, 0x7F, 0x00, 0xA0, 0xC3, 0x01, 0x00, 0xCE, 0xE4, 0x01, 0xC0,
        0x5C, 0xE2, 0xF6, 0xFF, 0xFF, 0x5A,
    ];
    if rev_b {
        fin[35] = 0xCA;
    }
    rom[0x200..0x224].copy_from_slice(&fin);

    // 3. Place 128-entry frequency table at 0x300 (prefix + A4 note 69)
    rom[0x300..0x320].copy_from_slice(&FREQ_TABLE_PREFIX);
    rom[0x300 + 69 * 4..0x300 + 69 * 4 + 4].copy_from_slice(&28_835_840u32.to_le_bytes());

    // 4. Place sequence bytecode at 0x600
    let seq_bytes: [u8; 19] = [
        0xB2, 0x00, 0x04, 0x00, 0x00, // SetTempoStep(0x00040000)
        0xC4, 0x00, // ProgramChange(ch=4, sample=0)
        0x94, 0x3C, 0xFF, // NoteOn(ch=4, note=60, vel=255)
        0x00, 0x40, // DeltaTime(64)
        0x84, 0x3C, 0x00, // NoteOff(ch=4, note=60, vel=0)
        0xB1, 0xAA, // SkipByte(0xAA)
        0xB0, // Loop
        0xB3, // EndTrack (unreachable after Loop)
    ];
    rom[0x600..0x600 + seq_bytes.len()].copy_from_slice(&seq_bytes);

    // 5. Place Sample 0 (one-shot, 8 bytes PCM8) at 0x700
    rom[0x700..0x704].copy_from_slice(&1u32.to_le_bytes()); // Pcm8Signed
    rom[0x704..0x708].copy_from_slice(&8u32.to_le_bytes()); // length
    rom[0x708..0x70C].copy_from_slice(&0u32.to_le_bytes()); // loop_start = 0
    rom[0x70C..0x710].copy_from_slice(&0u32.to_le_bytes()); // reserved
    rom[0x710..0x718].copy_from_slice(&[0x00, 0x10, 0x40, 0x7F, 0x80, 0xC0, 0xF0, 0x00]);

    // 6. Place Sample 1 (looped at offset 4, 16 bytes PCM8) at 0x720
    rom[0x720..0x724].copy_from_slice(&1u32.to_le_bytes());
    rom[0x724..0x728].copy_from_slice(&16u32.to_le_bytes());
    rom[0x728..0x72C].copy_from_slice(&4u32.to_le_bytes());
    rom[0x72C..0x730].copy_from_slice(&0u32.to_le_bytes());
    for i in 0..16u8 {
        rom[0x730 + i as usize] = i.wrapping_mul(16);
    }

    // 7. Place SoundBank at 0x800
    let song_count: u32 = if with_song { 1 } else { 0 };
    let sample_count: u32 = 2;
    let st_addr = GBA_ROM_BASE + 0x810;
    let nt_addr = st_addr + song_count * 8;
    rom[0x800..0x804].copy_from_slice(&song_count.to_le_bytes());
    rom[0x804..0x808].copy_from_slice(&sample_count.to_le_bytes());
    rom[0x808..0x80C].copy_from_slice(&st_addr.to_le_bytes());
    rom[0x80C..0x810].copy_from_slice(&nt_addr.to_le_bytes());

    let mut cur = 0x810;
    if with_song {
        rom[cur..cur + 4].copy_from_slice(&(GBA_ROM_BASE + 0x600).to_le_bytes());
        rom[cur + 4..cur + 8].copy_from_slice(&(GBA_ROM_BASE + 0x600).to_le_bytes());
        cur += 8;
    }
    rom[cur..cur + 4].copy_from_slice(&(GBA_ROM_BASE + 0x700).to_le_bytes());
    rom[cur + 4..cur + 8].copy_from_slice(&(GBA_ROM_BASE + 0x720).to_le_bytes());

    rom
}

#[test]
fn detect_and_parse_synthetic_rev_b_with_song() {
    let rom = build_synthetic_rom(true, true);
    assert!(detect(&rom));
    assert_eq!(FullFatVersion::detect_from_bytes(&rom), Some(Revision::RevB));
    let det = detect_details(&rom).expect("detection details");
    assert_eq!(det.revision, Revision::RevB);
    assert_eq!(det.mixer_offset, 0x100);
    assert_eq!(det.freq_table_offset, 0x300);

    let audio = parse(&rom).expect("synthetic ROM should parse");
    assert_eq!(audio.revision(), Revision::RevB);
    assert_eq!(audio.banks().len(), 1);
    assert_eq!(audio.total_songs(), 1);
    assert_eq!(audio.total_samples(), 2);

    let s0 = audio.bank_sample(0, 0).expect("sample 0");
    assert_eq!(s0.header.format, SampleFormat::Pcm8Signed);
    assert_eq!(s0.header.length, 8);
    assert!(!s0.header.is_looped());
    assert_eq!(s0.pcm.len(), 8);

    let s1 = audio.bank_sample(0, 1).expect("sample 1");
    assert_eq!(s1.header.length, 16);
    assert!(s1.header.is_looped());
    assert_eq!(s1.header.loop_start, 4);

    assert_eq!(audio.bank_samples(0).count(), 2);
    assert_eq!(audio.bank_samples(99).count(), 0);

    let song = audio.bank_song(0, 0).expect("song 0");
    assert_eq!(song.start_offset, 0x600);
    assert_eq!(song.loop_offset, 0x600);

    let seq_slice = audio.song_bytes(song).expect("song bytes");
    let mut iter = SeqIter::new(seq_slice);
    assert_eq!(iter.next(), Some(SeqEvent::SetTempoStep(0x0004_0000)));
    assert_eq!(
        iter.next(),
        Some(SeqEvent::ProgramChange {
            channel: 4,
            sample_id: 0
        })
    );
    assert_eq!(
        iter.next(),
        Some(SeqEvent::NoteOn {
            channel: 4,
            note: 60,
            velocity: 255
        })
    );
    assert_eq!(iter.next(), Some(SeqEvent::DeltaTime(64)));
    assert_eq!(
        iter.next(),
        Some(SeqEvent::NoteOff {
            channel: 4,
            note: 60,
            velocity: 0
        })
    );
    assert_eq!(iter.next(), Some(SeqEvent::SkipByte(0xAA)));
    assert_eq!(iter.next(), Some(SeqEvent::Loop));
    assert_eq!(iter.next(), None);
    assert!(!iter.is_truncated());
    assert_eq!(iter.position(), 18);
}

#[test]
fn detect_synthetic_rev_a_and_unknown_revision() {
    let rom_a = build_synthetic_rom(false, false);
    assert_eq!(
        FullFatVersion::detect_from_bytes(&rom_a),
        Some(Revision::RevA)
    );
    let audio_a = parse(&rom_a).expect("rev A should parse");
    assert_eq!(audio_a.revision(), Revision::RevA);
    assert_eq!(audio_a.total_songs(), 0);
    assert_eq!(audio_a.total_samples(), 2);

    // Erase FinalizeBuffer signature -> Revision::Unknown
    let mut rom_u = build_synthetic_rom(true, false);
    rom_u[0x200..0x224].fill(0);
    assert_eq!(
        FullFatVersion::detect_from_bytes(&rom_u),
        Some(Revision::Unknown)
    );
}

#[test]
fn multi_bank_cap_at_max_banks() {
    let mut rom = [0u8; 8192];
    let base = build_synthetic_rom(true, false);
    rom[..4096].copy_from_slice(&base);
    rom[0x800..0x840].fill(0);

    // Write 18 valid banks starting at 0x900 (stride 0x40), all pointing to Sample 0 at 0x700.
    for b in 0..18usize {
        let off = 0x900 + b * 0x40;
        let st_addr = GBA_ROM_BASE + (off + 16) as u32;
        rom[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
        rom[off + 4..off + 8].copy_from_slice(&1u32.to_le_bytes());
        rom[off + 8..off + 12].copy_from_slice(&st_addr.to_le_bytes());
        rom[off + 12..off + 16].copy_from_slice(&st_addr.to_le_bytes());
        rom[off + 16..off + 20].copy_from_slice(&(GBA_ROM_BASE + 0x700).to_le_bytes());
    }

    let audio = parse(&rom).expect("multi-bank ROM should parse");
    assert_eq!(audio.banks().len(), MAX_BANKS);
}

#[test]
fn loop_boundary_validation_and_adpcm4_header() {
    // loop_start == length must be rejected
    let mut bad_loop = build_synthetic_rom(true, false);
    bad_loop[0x708..0x70C].copy_from_slice(&8u32.to_le_bytes());
    assert_eq!(parse(&bad_loop).unwrap_err(), Error::NoBanksFound);

    // loop_start == length - 1 must be accepted
    let mut ok_loop = build_synthetic_rom(true, false);
    ok_loop[0x708..0x70C].copy_from_slice(&7u32.to_le_bytes());
    let audio = parse(&ok_loop).expect("loop_start == length - 1 should parse");
    let s0 = audio.bank_sample(0, 0).expect("s0");
    assert!(s0.header.is_looped());
    assert_eq!(s0.header.loop_start, 7);

    // SampleFormat::Adpcm4 (format == 0) must be accepted
    let mut adpcm_rom = build_synthetic_rom(true, false);
    adpcm_rom[0x700..0x704].copy_from_slice(&0u32.to_le_bytes());
    let audio = parse(&adpcm_rom).expect("Adpcm4 sample should parse");
    assert_eq!(
        audio.bank_sample(0, 0).expect("s0").header.format,
        SampleFormat::Adpcm4
    );
}

#[test]
fn seq_iter_events_truncation_and_fused() {
    let data = [0xA3, 0xDF, 0xB3, 0x90, 0x3C, 0x7F];
    let mut it = SeqIter::new(&data);
    assert_eq!(it.next(), Some(SeqEvent::Nop { channel: 3 }));
    assert_eq!(it.next(), Some(SeqEvent::UnknownStatus(0xDF)));
    assert_eq!(it.next(), Some(SeqEvent::EndTrack));
    assert_eq!(it.next(), None);
    assert_eq!(it.next(), None);
    assert!(!it.is_truncated());

    let trunc = [0x90, 0x3C];
    let mut it2 = SeqIter::new(&trunc);
    assert_eq!(it2.next(), None);
    assert!(it2.is_truncated());
    assert_eq!(it2.next(), None);
}

#[test]
fn adversarial_and_truncated_inputs_never_panic() {
    assert!(!detect(&[]));
    assert_eq!(parse(&[]).unwrap_err(), Error::InvalidRom);

    let zeros = [0u8; 1024];
    assert!(!detect(&zeros));
    assert_eq!(parse(&zeros).unwrap_err(), Error::NotDetected);

    let ones = [0xFFu8; 2048];
    assert!(!detect(&ones));
    assert_eq!(parse(&ones).unwrap_err(), Error::NotDetected);

    // Truncated after detection signature (no banks)
    let mut rom = build_synthetic_rom(true, false);
    rom[0x800..0x820].fill(0);
    assert!(detect(&rom));
    assert_eq!(parse(&rom).unwrap_err(), Error::NoBanksFound);

    for trunc in &[
        &[][..],
        &[0x00][..],
        &[0x80][..],
        &[0x80, 0x3C][..],
        &[0x90, 0x3C][..],
        &[0xB1][..],
        &[0xB2, 0x00, 0x04][..],
        &[0xC0][..],
        &[0xFF][..],
    ] {
        let count = SeqIter::new(trunc).count();
        assert!(count <= 1);
    }
}

#[test]
fn adpcm4_decoder_deterministic_and_bounded() {
    let mut dec = Adpcm4Decoder::new();
    let packed = [0x00u8, 0xEC, 0x55, 0xAA, 0xFF, 0x12, 0x34, 0x78];
    let mut any_nonzero = false;
    for &b in &packed {
        let [s0, s1] = dec.decode_byte_12bit(b);
        assert!((-2048..=2047).contains(&s0));
        assert!((-2048..=2047).contains(&s1));
        if s0 != 0 || s1 != 0 {
            any_nonzero = true;
        }
    }
    assert!(any_nonzero);
    assert!(dec.step_index() <= 48);
}

#[test]
fn accuracy_synth_silence_pcm8_adpcm4_and_looping() {
    assert_eq!(FREQ_TABLE_Q16[69], 28_835_840);
    assert_eq!(compute_channel_gain(0x0100, 0x0100), 16);
    assert_eq!(finalize_sample_dma8(0), 0);
    assert_eq!(finalize_sample_dma8(127 * 16), 127);
    assert_eq!(finalize_sample_dma8(i16::MAX), 127);
    assert_eq!(finalize_sample_dma8(i16::MIN), -128);

    let synth = AccuracySynth::new(BASE_SAMPLE_RATE_HZ);

    // Silence when voice is inactive
    let pcm = [32u8, 64, 96, 127, 96, 64, 32, 0];
    let sample_oneshot = Sample {
        header: SampleHeader {
            rom_offset: 0,
            format: SampleFormat::Pcm8Signed,
            length: pcm.len() as u32,
            loop_start: 0,
            reserved: 0,
        },
        pcm: &pcm,
    };
    let mut inactive = FullFatVoice::new();
    let mut out = [123i16; 16];
    synth.render_voice_i16(&sample_oneshot, &mut inactive, &mut out);
    assert_eq!(out, [0i16; 16]);

    // One-shot PCM8 triggers and stops cleanly
    let mut v1 = FullFatVoice::with_raw_params(0x1000_0000, 0x0100, 0);
    synth.render_voice_i16(&sample_oneshot, &mut v1, &mut out);
    assert!(!v1.active);
    assert_eq!(out[0], 32 << 8);
    assert_eq!(out[3], 127 << 8);
    assert_eq!(out[8], 0);

    // Looped PCM8 stays active past sample end
    let sample_loop = Sample {
        header: SampleHeader {
            rom_offset: 0,
            format: SampleFormat::Pcm8Signed,
            length: pcm.len() as u32,
            loop_start: 2,
            reserved: 0,
        },
        pcm: &pcm,
    };
    let mut v_loop = FullFatVoice::trigger(&sample_loop, 60, 0x0100, BASE_SAMPLE_RATE_HZ);
    let mut out_loop = [0i16; 64];
    synth.render_voice_i16(&sample_loop, &mut v_loop, &mut out_loop);
    assert!(v_loop.active);
    assert!(out_loop[32..].iter().any(|&s| s != 0));

    // ADPCM4 trigger produces non-zero audio and stops when one-shot finishes
    let adpcm_bytes = [0x12u8, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
    let sample_adpcm = Sample {
        header: SampleHeader {
            rom_offset: 0,
            format: SampleFormat::Adpcm4,
            length: adpcm_bytes.len() as u32,
            loop_start: 0,
            reserved: 0,
        },
        pcm: &adpcm_bytes,
    };
    let mut v_adpcm = FullFatVoice::with_raw_params(0x1000_0000, 0x0100, 0);
    let mut out_adpcm = [0i16; 32];
    synth.render_voice_i16(&sample_adpcm, &mut v_adpcm, &mut out_adpcm);
    assert!(!v_adpcm.active);
    assert!(out_adpcm[..16].iter().any(|&s| s != 0));
    assert_eq!(out_adpcm[24..], [0i16; 8]);
}

#[test]
fn studio_synth_pcm8_adpcm4_pan_and_finite_guarantee() {
    let studio = StudioSynth::new(48_000);
    let pcm = [0u8, 32, 64, 96, 127, 96, 64, 32, 0, 224, 192, 160, 129, 160, 192, 224];
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

    // Center pan: L == R and all finite
    let mut pos = 0.0f64;
    let mut l = [0.0f32; 64];
    let mut r = [0.0f32; 64];
    studio.render_voice_f32(&sample, &mut pos, note_to_playback_hz(60), 0x0100, 0, &mut l, &mut r);
    assert!(l.iter().all(|s| s.is_finite()));
    assert!(r.iter().all(|s| s.is_finite()));
    assert!(l.iter().any(|&s| s != 0.0));
    for (a, b) in l.iter().zip(r.iter()) {
        assert!((a - b).abs() < 1e-5);
    }

    // Hard right pan: L ~ 0, R non-zero
    let mut pos_r = 2.0f64;
    let mut l2 = [0.0f32; 32];
    let mut r2 = [0.0f32; 32];
    studio.render_voice_f32(&sample, &mut pos_r, 11_025.0, 0x0100, 64, &mut l2, &mut r2);
    assert!(l2.iter().all(|&s| s.abs() < 1e-6));
    assert!(r2.iter().any(|&s| s != 0.0));

    // Malformed / NaN / infinite inputs produce zero output without NaN
    let mut nan_pos = f64::NAN;
    let mut l3 = [0.0f32; 16];
    let mut r3 = [0.0f32; 16];
    studio.render_voice_f32(&sample, &mut nan_pos, 11_025.0, 0x0100, 0, &mut l3, &mut r3);
    assert!(l3.iter().all(|&s| s == 0.0 && s.is_finite()));
    let mut inf_pos = 0.0f64;
    studio.render_voice_f32(&sample, &mut inf_pos, f64::INFINITY, 0x0100, 0, &mut l3, &mut r3);
    assert!(l3.iter().all(|&s| s == 0.0 && s.is_finite()));

    // ADPCM4 rendering in StudioSynth
    let adpcm_bytes = [0x12u8, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
    let sample_adpcm = Sample {
        header: SampleHeader {
            rom_offset: 0,
            format: SampleFormat::Adpcm4,
            length: adpcm_bytes.len() as u32,
            loop_start: 1,
            reserved: 0,
        },
        pcm: &adpcm_bytes,
    };
    let mut pos_a = 0.0f64;
    let mut la = [0.0f32; 32];
    let mut ra = [0.0f32; 32];
    studio.render_voice_f32(&sample_adpcm, &mut pos_a, 11_025.0, 0x0100, 0, &mut la, &mut ra);
    assert!(la.iter().all(|s| s.is_finite()));
    assert!(la.iter().any(|&s| s != 0.0));

    // Pitch step sanity
    assert!(compute_pitch_step(60, BASE_SAMPLE_RATE_HZ) > 0);
    assert_eq!(compute_pitch_step(60, 0), 0);
}

#[cfg(feature = "alloc")]
#[test]
fn export_wav_sf2_adpcm4_and_allocation_bomb_dedup() {
    use fullfat_audio::{build_sf2, export_wav, export_wav_default};

    let rom = build_synthetic_rom(true, true);
    let audio = parse(&rom).expect("parse");
    let s0 = audio.bank_sample(0, 0).expect("s0");
    let wav = export_wav_default(&s0).expect("wav");
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(&wav[36..40], b"data");
    assert_eq!(wav.len(), 44 + 8 * 2);

    // sample_rate == 0 falls back to BASE_SAMPLE_RATE_HZ
    let wav0 = export_wav(&s0, 0).expect("wav0");
    assert_eq!(wav, wav0);

    // ADPCM4 WAV decodes 2 samples per byte (8 bytes -> 16 samples -> 32 PCM16 bytes)
    let mut adpcm_sample = s0;
    adpcm_sample.header.format = SampleFormat::Adpcm4;
    let wav_adpcm = export_wav_default(&adpcm_sample).expect("adpcm wav");
    assert_eq!(wav_adpcm.len(), 44 + 16 * 2);

    // Unknown sample format returns explicit Error::UnsupportedSampleFormat
    let mut bad_sample = s0;
    bad_sample.header.format = SampleFormat::from_raw(99);
    assert_eq!(
        export_wav_default(&bad_sample).unwrap_err(),
        Error::UnsupportedSampleFormat(SampleFormat::from_raw(99))
    );

    // SF2 RIFF size verification (including odd-length INAM padding)
    let sf2 = build_sf2(&audio, "OddName").expect("sf2");
    assert_eq!(&sf2[0..4], b"RIFF");
    let riff_len = u32::from_le_bytes([sf2[4], sf2[5], sf2[6], sf2[7]]) as usize;
    assert_eq!(riff_len + 8, sf2.len());
    assert_eq!(&sf2[8..12], b"sfbk");

    // Allocation-bomb regression: 512 aliased pointers to the SAME 4 KiB sample must deduplicate
    // to a single 4 KiB pool in `smpl` rather than 512x duplication!
    let mut bomb = vec![0u8; 16384];
    bomb[..4096].copy_from_slice(&rom);
    bomb[0x800..0x840].fill(0);
    // Single 4096-byte sample at 0x2000
    bomb[0x2000..0x2004].copy_from_slice(&1u32.to_le_bytes());
    bomb[0x2004..0x2008].copy_from_slice(&4096u32.to_le_bytes());
    bomb[0x2008..0x200C].copy_from_slice(&0u32.to_le_bytes());
    bomb[0x200C..0x2010].copy_from_slice(&0u32.to_le_bytes());
    // Bank at 0x1000 with 512 pointers all referencing 0x2000
    let st_addr = GBA_ROM_BASE + 0x1010;
    bomb[0x1000..0x1004].copy_from_slice(&0u32.to_le_bytes());
    bomb[0x1004..0x1008].copy_from_slice(&512u32.to_le_bytes());
    bomb[0x1008..0x100C].copy_from_slice(&st_addr.to_le_bytes());
    bomb[0x100C..0x1010].copy_from_slice(&st_addr.to_le_bytes());
    for i in 0..512usize {
        let p = 0x1010 + i * 4;
        bomb[p..p + 4].copy_from_slice(&(GBA_ROM_BASE + 0x2000).to_le_bytes());
    }
    let bomb_audio = parse(&bomb).expect("aliased bank should parse");
    let bomb_sf2 = build_sf2(&bomb_audio, "DedupTest").expect("dedup sf2");
    // Without deduplication this would be >4 MiB; with dedup it's ~8 KiB smpl + ~70 KiB pdta.
    assert!(
        bomb_sf2.len() < 128 * 1024,
        "expected deduplicated SF2 < 128 KiB, got {}",
        bomb_sf2.len()
    );
}
