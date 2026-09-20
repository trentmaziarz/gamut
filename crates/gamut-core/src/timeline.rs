//! One video track: clips end to end, each a span of one media file. A
//! clip's place on the timeline is the sum of the clips before it, so
//! trimming, splitting and deleting are list operations on pure data.

use serde::{Deserialize, Serialize};

/// The smallest span a clip can be trimmed to, in seconds: one frame at
/// the highest rate slate plays.
pub const MIN_CLIP_SECONDS: f64 = 1.0 / 60.0;

/// A span of one media file on the track.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Clip {
    /// Index into the project's media list.
    pub media: usize,
    /// Where the span starts in the media, in seconds.
    pub source_in: f64,
    /// Where the span ends in the media, in seconds, exclusive.
    pub source_out: f64,
}

impl Default for Clip {
    fn default() -> Self {
        Clip {
            media: 0,
            source_in: 0.0,
            source_out: 0.0,
        }
    }
}

impl Clip {
    /// The whole of a media file of `duration` seconds.
    pub fn whole(media: usize, duration: f64) -> Clip {
        Clip {
            media,
            source_in: 0.0,
            source_out: duration.max(0.0),
        }
    }

    /// The length of the span in seconds.
    pub fn duration(&self) -> f64 {
        (self.source_out - self.source_in).max(0.0)
    }
}

/// The one video track of M2.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Track {
    pub clips: Vec<Clip>,
}

impl Track {
    /// A track holding one clip.
    pub fn one(clip: Clip) -> Track {
        Track { clips: vec![clip] }
    }

    /// The length of the track in seconds.
    pub fn duration(&self) -> f64 {
        self.clips.iter().map(Clip::duration).sum()
    }

    /// Where the clip at `index` starts on the timeline, in seconds.
    pub fn start_of(&self, index: usize) -> f64 {
        self.clips.iter().take(index).map(Clip::duration).sum()
    }

    /// The clip under `seconds` and the offset into it, or `None` past the
    /// end. At a boundary the later clip wins; the very end belongs to the
    /// last clip so the playhead can rest there.
    pub fn clip_at(&self, seconds: f64) -> Option<(usize, f64)> {
        if self.clips.is_empty() || seconds < 0.0 {
            return None;
        }
        let mut start = 0.0;
        for (index, clip) in self.clips.iter().enumerate() {
            let end = start + clip.duration();
            if seconds < end {
                return Some((index, seconds - start));
            }
            start = end;
        }
        let last = self.clips.len() - 1;
        (seconds <= start + 1e-9).then(|| (last, self.clips[last].duration()))
    }

    /// Splits the clip under `seconds` into two at that point. Nothing
    /// happens at a boundary or within one frame of one, or past the end.
    /// Returns the index of the second half.
    pub fn split_at(&mut self, seconds: f64) -> Option<usize> {
        let (index, offset) = self.clip_at(seconds)?;
        let clip = self.clips[index];
        if offset < MIN_CLIP_SECONDS || clip.duration() - offset < MIN_CLIP_SECONDS {
            return None;
        }
        let at = clip.source_in + offset;
        self.clips[index].source_out = at;
        self.clips.insert(
            index + 1,
            Clip {
                media: clip.media,
                source_in: at,
                source_out: clip.source_out,
            },
        );
        Some(index + 1)
    }

    /// Removes the clip at `index`; the clips after it move up to close
    /// the gap. Returns the removed clip.
    pub fn ripple_delete(&mut self, index: usize) -> Option<Clip> {
        (index < self.clips.len()).then(|| self.clips.remove(index))
    }

    /// Moves the in point of the clip at `index` to `seconds` in the media,
    /// kept at least one frame before the out point and never below zero.
    pub fn trim_in(&mut self, index: usize, seconds: f64) {
        if let Some(clip) = self.clips.get_mut(index) {
            clip.source_in = seconds.clamp(0.0, (clip.source_out - MIN_CLIP_SECONDS).max(0.0));
        }
    }

    /// Moves the out point of the clip at `index` to `seconds` in the
    /// media, kept at least one frame after the in point and never past
    /// `media_duration`.
    pub fn trim_out(&mut self, index: usize, seconds: f64, media_duration: f64) {
        if let Some(clip) = self.clips.get_mut(index) {
            let floor = clip.source_in + MIN_CLIP_SECONDS;
            clip.source_out = seconds.clamp(floor, media_duration.max(floor));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Clip, MIN_CLIP_SECONDS, Track};

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn a_split_in_the_middle_makes_two_clips_that_meet() {
        let mut track = Track::one(Clip::whole(0, 10.0));
        assert_eq!(track.split_at(4.0), Some(1));
        assert_eq!(track.clips.len(), 2);
        assert!(near(track.clips[0].source_out, 4.0));
        assert!(near(track.clips[1].source_in, 4.0));
        assert!(near(track.clips[1].source_out, 10.0));
        assert!(near(track.duration(), 10.0));
        assert_eq!(track.clip_at(4.0), Some((1, 0.0)));
        assert!(near(track.start_of(1), 4.0));
    }

    #[test]
    fn a_split_at_a_boundary_or_past_the_end_does_nothing() {
        let mut track = Track::one(Clip::whole(0, 10.0));
        track.split_at(4.0);
        assert_eq!(track.split_at(4.0), None);
        assert_eq!(track.split_at(0.0), None);
        assert_eq!(track.split_at(4.0 + MIN_CLIP_SECONDS / 2.0), None);
        assert_eq!(track.split_at(10.0), None);
        assert_eq!(track.split_at(11.0), None);
        assert_eq!(track.clips.len(), 2);
    }

    #[test]
    fn a_ripple_delete_closes_the_gap() {
        let mut track = Track::one(Clip::whole(0, 10.0));
        track.split_at(3.0);
        track.split_at(7.0);
        let removed = track.ripple_delete(1).expect("the middle clip");
        assert!(near(removed.source_in, 3.0) && near(removed.source_out, 7.0));
        assert_eq!(track.clips.len(), 2);
        assert!(near(track.duration(), 6.0));
        assert_eq!(
            track.clip_at(3.0).map(|(i, o)| (i, (o * 1e6).round())),
            Some((1, 0.0))
        );
        assert!(near(track.clips[1].source_in, 7.0));
        assert_eq!(track.ripple_delete(5), None);
        track.ripple_delete(0);
        track.ripple_delete(0);
        assert!(track.clips.is_empty());
        assert_eq!(track.clip_at(0.0), None);
    }

    #[test]
    fn trims_stop_at_one_frame_and_the_media_ends() {
        let mut track = Track::one(Clip::whole(0, 10.0));
        track.trim_in(0, 2.5);
        assert!(near(track.clips[0].source_in, 2.5));
        track.trim_in(0, -1.0);
        assert!(near(track.clips[0].source_in, 0.0));
        track.trim_in(0, 20.0);
        assert!(near(track.clips[0].source_in, 10.0 - MIN_CLIP_SECONDS));
        track.trim_in(0, 0.0);
        track.trim_out(0, 4.0, 10.0);
        assert!(near(track.clips[0].source_out, 4.0));
        track.trim_out(0, 30.0, 10.0);
        assert!(near(track.clips[0].source_out, 10.0));
        track.trim_out(0, -5.0, 10.0);
        assert!(near(track.clips[0].source_out, MIN_CLIP_SECONDS));
        track.trim_in(9, 1.0);
        track.trim_out(9, 1.0, 10.0);
    }

    #[test]
    fn the_end_of_the_track_belongs_to_the_last_clip() {
        let mut track = Track::one(Clip::whole(0, 5.0));
        track.split_at(2.0);
        assert_eq!(track.clip_at(5.0), Some((1, 3.0)));
        assert_eq!(track.clip_at(5.1), None);
        assert_eq!(track.clip_at(-0.1), None);
        assert_eq!(track.clip_at(1.0), Some((0, 1.0)));
    }
}
