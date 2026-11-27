//! Clean-room `#![no_std]` bytecode decoder for Full Fat music sequences.
//!
//! Sequence events are variable-length byte commands processed by the engine's
//! per-frame sequencer update routine:
//! - `0x00..=0x7F`: 15-bit big-endian delta time `(b0 << 8) | b1` (shifted `<< 16` into 16.16 ticks)
//! - `0x80..=0x8F`: Note Off on channel `b & 0x0F`, 2 data bytes (`note`, `unused`)
//! - `0x90..=0x9F`: Note On on channel `b & 0x0F`, 2 data bytes (`note`, `velocity`)
//! - `0xA0..=0xAF`: No-op (0 data bytes)
//! - `0xB0`: Loop back to `SongEntry.loop_offset`
//! - `0xB1`: Skip 1 byte
//! - `0xB2`: Set 16.16 tempo step (4-byte big-endian `u32`)
//! - `0xB3`: End of track
//! - `0xC0..=0xCF`: Program Change on channel `b & 0x0F`, 1 data byte (`sample_id`)

/// Decoded Full Fat sequence bytecode event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SeqEvent {
    /// 15-bit big-endian delta time in ticks (`b0 < 0x80`).
    DeltaTime(u16),
    /// Note Off (`0x8c`, `note`, `release_vel`).
    NoteOff {
        /// MIDI-style channel `0..=15`.
        channel: u8,
        /// Note number `0..=127`.
        note: u8,
        /// Trailing byte (ignored by engine).
        velocity: u8,
    },
    /// Note On (`0x9c`, `note`, `velocity`).
    NoteOn {
        /// MIDI-style channel `0..=15`.
        channel: u8,
        /// Note number `0..=127`.
        note: u8,
        /// Volume/velocity `0..=255` (clamped to `0x100` by engine).
        velocity: u8,
    },
    /// No-op (`0xAc`).
    Nop {
        /// Low nibble `0..=15`.
        channel: u8,
    },
    /// Loop to song's `loop_offset` (`0xB0`).
    Loop,
    /// Ignored 1-byte meta command (`0xB1`).
    SkipByte(u8),
    /// Set 16.16 fixed-point tempo step per mix frame (`0xB2`, big-endian `u32`).
    SetTempoStep(u32),
    /// End of track marker (`0xB3`).
    EndTrack,
    /// Program Change (`0xCc`, `sample_id`).
    ProgramChange {
        /// MIDI-style channel `0..=15`.
        channel: u8,
        /// Sample index into `SoundBank.sample_table`.
        sample_id: u8,
    },
    /// Unrecognized status byte (`>= 0x80` not handled by the jump table).
    UnknownStatus(u8),
}

/// Zero-allocation streaming iterator over a Full Fat sequence byte slice.
///
/// Stops cleanly on [`SeqEvent::EndTrack`], [`SeqEvent::Loop`], or end-of-slice / truncation.
/// Implements [`core::iter::FusedIterator`].
#[derive(Debug, Clone, Copy)]
pub struct SeqIter<'a> {
    data: &'a [u8],
    pos: usize,
    done: bool,
    truncated: bool,
}

impl<'a> SeqIter<'a> {
    /// Creates a new sequence iterator starting at byte 0 of `data`.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            done: false,
            truncated: false,
        }
    }

    /// Returns the current byte offset consumed from the input slice.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.pos
    }

    /// Returns `true` if iteration stopped because a multi-byte command was truncated.
    #[must_use]
    pub const fn is_truncated(&self) -> bool {
        self.truncated
    }

    #[inline]
    fn next_byte(&mut self) -> Option<u8> {
        let &b = self.data.get(self.pos)?;
        self.pos += 1;
        Some(b)
    }

    #[inline]
    fn next_payload_byte(&mut self) -> Option<u8> {
        match self.next_byte() {
            Some(b) => Some(b),
            None => {
                self.done = true;
                self.truncated = true;
                None
            }
        }
    }
}

impl Iterator for SeqIter<'_> {
    type Item = SeqEvent;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let Some(b0) = self.next_byte() else {
            self.done = true;
            return None;
        };
        if (b0 & 0x80) == 0 {
            let b1 = self.next_payload_byte()?;
            let delta = u16::from_be_bytes([b0, b1]);
            return Some(SeqEvent::DeltaTime(delta));
        }

        let hi = b0 >> 4;
        let lo = b0 & 0x0F;
        match hi {
            0x8 => {
                let note = self.next_payload_byte()?;
                let velocity = self.next_payload_byte()?;
                Some(SeqEvent::NoteOff {
                    channel: lo,
                    note,
                    velocity,
                })
            }
            0x9 => {
                let note = self.next_payload_byte()?;
                let velocity = self.next_payload_byte()?;
                Some(SeqEvent::NoteOn {
                    channel: lo,
                    note,
                    velocity,
                })
            }
            0xA => Some(SeqEvent::Nop { channel: lo }),
            0xB => match lo {
                0 => {
                    self.done = true;
                    Some(SeqEvent::Loop)
                }
                1 => {
                    let skipped = self.next_payload_byte()?;
                    Some(SeqEvent::SkipByte(skipped))
                }
                2 => {
                    let t0 = self.next_payload_byte()?;
                    let t1 = self.next_payload_byte()?;
                    let t2 = self.next_payload_byte()?;
                    let t3 = self.next_payload_byte()?;
                    Some(SeqEvent::SetTempoStep(u32::from_be_bytes([t0, t1, t2, t3])))
                }
                3 => {
                    self.done = true;
                    Some(SeqEvent::EndTrack)
                }
                _ => Some(SeqEvent::UnknownStatus(b0)),
            },
            0xC => {
                let sample_id = self.next_payload_byte()?;
                Some(SeqEvent::ProgramChange {
                    channel: lo,
                    sample_id,
                })
            }
            _ => Some(SeqEvent::UnknownStatus(b0)),
        }
    }
}

impl core::iter::FusedIterator for SeqIter<'_> {}
