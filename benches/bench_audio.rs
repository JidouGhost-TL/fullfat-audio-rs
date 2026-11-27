use std::hint::black_box;
use std::time::Instant;

use fullfat_audio::detect::{FREQ_TABLE_PREFIX, MIX_LOOP_SIG};
use fullfat_audio::{
    build_sf2, detect, export_wav_default, parse, AccuracySynth, FullFatVoice, Sample,
    SampleFormat, SampleHeader, SeqIter, StudioSynth, BASE_SAMPLE_RATE_HZ, GBA_ROM_BASE,
};

fn bench_samples<F: FnMut()>(name: &str, frames: usize, iters: usize, mut f: F) {
    for _ in 0..50 {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let ns_per_iter = elapsed.as_nanos() as f64 / iters as f64;
    let msamples_per_sec = ((iters * frames) as f64 / elapsed.as_secs_f64()) / 1_000_000.0;
    println!("{name:<52} {ns_per_iter:10.2} ns/iter ({msamples_per_sec:8.2} MSamples/sec)");
}

fn bench_throughput<F: FnMut()>(name: &str, bytes_per_iter: usize, iters: usize, mut f: F) {
    for _ in 0..5 {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let us_per_iter = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
    let mib_per_sec =
        ((iters * bytes_per_iter) as f64 / (1024.0 * 1024.0)) / elapsed.as_secs_f64();
    println!("{name:<52} {us_per_iter:10.2} us/iter ({mib_per_sec:8.2} MiB/s)");
}

fn build_synthetic_16mib_rom() -> Vec<u8> {
    let mut rom = vec![0u8; 16 * 1024 * 1024];

    // Place MixChannel + FinalizeBuffer + FreqTable near the end to measure full-ROM scan speed.
    let base = 15 * 1024 * 1024;
    rom[base..base + 32].copy_from_slice(&MIX_LOOP_SIG);
    let fin_revb: [u8; 36] = [
        0xF2, 0x00, 0xD2, 0xE0, 0x40, 0x02, 0xA0, 0xE1, 0x80, 0x00, 0x70, 0xE3, 0x7F, 0x00, 0xE0,
        0xB3, 0x7F, 0x00, 0x50, 0xE3, 0x7F, 0x00, 0xA0, 0xC3, 0x01, 0x00, 0xCE, 0xE4, 0x01, 0xC0,
        0x5C, 0xE2, 0xF6, 0xFF, 0xFF, 0xCA,
    ];
    rom[base + 0x100..base + 0x124].copy_from_slice(&fin_revb);
    rom[base + 0x200..base + 0x220].copy_from_slice(&FREQ_TABLE_PREFIX);
    rom[base + 0x200 + 69 * 4..base + 0x200 + 69 * 4 + 4]
        .copy_from_slice(&28_835_840u32.to_le_bytes());

    // Place 16 samples of 4 KiB each + 1 bank at base + 0x1000.
    let sample_count = 16usize;
    let sample_len = 4096usize;
    let bank_off = base + 0x1000;
    let st_addr = GBA_ROM_BASE + (bank_off + 16) as u32;
    let nt_addr = st_addr;
    rom[bank_off..bank_off + 4].copy_from_slice(&0u32.to_le_bytes());
    rom[bank_off + 4..bank_off + 8].copy_from_slice(&(sample_count as u32).to_le_bytes());
    rom[bank_off + 8..bank_off + 12].copy_from_slice(&st_addr.to_le_bytes());
    rom[bank_off + 12..bank_off + 16].copy_from_slice(&nt_addr.to_le_bytes());

    let samples_base = bank_off + 0x200;
    for i in 0..sample_count {
        let s_off = samples_base + i * (16 + sample_len);
        let ptr_off = bank_off + 16 + i * 4;
        rom[ptr_off..ptr_off + 4].copy_from_slice(&(GBA_ROM_BASE + s_off as u32).to_le_bytes());
        rom[s_off..s_off + 4].copy_from_slice(&1u32.to_le_bytes());
        rom[s_off + 4..s_off + 8].copy_from_slice(&(sample_len as u32).to_le_bytes());
        rom[s_off + 8..s_off + 12].copy_from_slice(&0u32.to_le_bytes());
        rom[s_off + 12..s_off + 16].copy_from_slice(&0u32.to_le_bytes());
        for j in 0..sample_len {
            rom[s_off + 16 + j] = (((j % 32) as i8 - 16) * 8) as u8;
        }
    }
    rom
}

fn main() {
    println!("=== fullfat-audio Performance Benchmarks ===");
    let rom = build_synthetic_16mib_rom();

    bench_throughput("detect (16 MiB ROM)", rom.len(), 200, || {
        black_box(detect(black_box(&rom)));
    });

    bench_throughput("parse (16 MiB ROM, 16 samples)", rom.len(), 100, || {
        let parsed = parse(black_box(&rom));
        let _ = black_box(parsed);
    });

    if let Ok(audio) = parse(&rom) {
        if let Some(s0) = audio.bank_sample(0, 0) {
            bench_throughput("export_wav_default (4 KiB PCM8)", s0.pcm.len(), 20_000, || {
                let _ = black_box(export_wav_default(black_box(&s0)));
            });
        }
        bench_throughput("build_sf2 (16 samples, 64 KiB PCM)", 65_536, 2_000, || {
            let _ = black_box(build_sf2(black_box(&audio), "BenchBank"));
        });
    }

    // Sequence decoder throughput
    let mut seq = Vec::with_capacity(4096);
    for _ in 0..500 {
        seq.extend_from_slice(&[0x90, 0x3C, 0xFF, 0x00, 0x10, 0x80, 0x3C, 0x00]);
    }
    seq.push(0xB3);
    bench_throughput("SeqIter decode (4,001 bytes)", seq.len(), 50_000, || {
        let count = SeqIter::new(black_box(&seq)).count();
        black_box(count);
    });

    // Synth benchmarks
    let pcm: Vec<u8> = (0i32..4096).map(|i| (((i % 32) - 16) * 8) as u8).collect();
    let sample = Sample {
        header: SampleHeader {
            rom_offset: 0,
            format: SampleFormat::Pcm8Signed,
            length: pcm.len() as u32,
            loop_start: 1,
            reserved: 0,
        },
        pcm: &pcm,
    };

    let synth = AccuracySynth::new(BASE_SAMPLE_RATE_HZ);
    let mut acc = vec![0i16; 1024];
    let mut out_i16 = vec![0i16; 1024];
    let iters = 100_000usize;

    bench_samples("AccuracySynth::mix_voice (1:1, 1024f)", 1024, iters, || {
        let mut voice = FullFatVoice::with_raw_params(0x1000_0000, 0x0100, 1);
        synth.mix_voice(black_box(&sample), &mut voice, &mut acc, true);
        black_box(&acc);
    });

    bench_samples(
        "AccuracySynth::mix_voice (resampled, 1024f)",
        1024,
        iters,
        || {
            let mut voice = FullFatVoice::with_raw_params(0x1565_4321, 0x0100, 1);
            synth.mix_voice(black_box(&sample), &mut voice, &mut acc, true);
            black_box(&acc);
        },
    );

    bench_samples("AccuracySynth::finalize_i16 (1024f)", 1024, iters, || {
        AccuracySynth::finalize_i16(black_box(&acc), &mut out_i16);
        black_box(&out_i16);
    });

    let studio = StudioSynth::new(48_000);
    let mut f_l = vec![0.0f32; 1024];
    let mut f_r = vec![0.0f32; 1024];
    bench_samples(
        "StudioSynth::render_voice_f32 (Lanczos3 48k, 1024f)",
        1024,
        10_000,
        || {
            let mut pos = 0.0f64;
            f_l.fill(0.0);
            f_r.fill(0.0);
            studio.render_voice_f32(
                black_box(&sample),
                &mut pos,
                black_box(11_025.0),
                0x0100,
                0,
                &mut f_l,
                &mut f_r,
            );
            black_box((&f_l, &f_r));
        },
    );
}
