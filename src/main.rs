//! nix-visualizer
//!
//! Captures audio from a PipeWire sink, runs FFT (zero-padded for denser
//! spectral sampling) -> aggregates into log-spaced bands (narrow bands use
//! interpolation, wide bands use RMS to suppress noise) -> normalizes to
//! [0.001, bar_max] -> spatial smoothing across bands (more smoothing at
//! high frequencies) -> frequency/volume-aware attack + gravity-style fall
//! smoothing -> writes one comma-separated line of floats to stdout per
//! frame, flushed immediately for Quickshell's Process + SplitParser.
//!
//! Architecture:
//!   - PipeWire thread: only pushes downmixed samples into a shared ring
//!     buffer (Mutex<VecDeque<f32>>); no heavy compute in the audio callback.
//!   - Main thread: wakes at a fixed rate (--fps), pulls the latest samples,
//!     runs the FFT pipeline, prints one line.
//!
//! Power saving via signals:
//!   - SIGUSR2 -> pause. The main thread truly sleeps on a Condvar in
//!     PauseControl::block_while_paused() (not polling), so CPU drops to
//!     ~0%; the PipeWire process callback also returns early while paused,
//!     skipping the downmix work.
//!   - SIGUSR1 -> resume, wakes the main thread.
//!
//!   Test with:
//!     kill -USR2 <pid>   # pause
//!     kill -USR1 <pid>   # resume

use std::collections::VecDeque;
use std::fmt::Write as FmtWrite;
use std::io::{self, Write, BufWriter};
use std::mem;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use rustfft::{num_complex::Complex32, FftPlanner};

use signal_hook::consts::signal::{SIGUSR1, SIGUSR2};
use signal_hook::iterator::Signals;

use pipewire as pw;
use pw::spa;
use pw::spa::param::audio::{AudioFormat, AudioInfoRaw};
use pw::spa::pod::Pod;
use pw::stream::StreamFlags;

// ---------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------

#[derive(Parser, Debug, Clone)]
#[command(about = "PipeWire audio visualizer -> log-band bars -> stdout text stream")]
struct Args {
    /// PipeWire sink name, e.g. cava_sink or alsa_output.xxx.analog-stereo.
    /// A pactl-style "xxx.monitor" name is also accepted; the ".monitor"
    /// suffix is stripped automatically since target.object needs the
    /// sink's own node name.
    #[arg(long)]
    sink: String,

    /// Total number of output bars
    #[arg(long, default_value_t = 32)]
    num_bars: usize,

    /// Max value per bar (output range is [0.001, bar_max])
    #[arg(long, default_value_t = 1.0)]
    bar_max: f32,

    /// Lowest frequency (Hz), lower bound of log spacing
    #[arg(long, default_value_t = 30.0)]
    min_freq: f32,

    /// Highest frequency (Hz), upper bound of log spacing (clamped to
    /// sample_rate/2). Raise to ~20000 for more hihat/cymbal detail.
    #[arg(long, default_value_t = 18000.0)]
    max_freq: f32,

    /// Band allocation warp exponent. 1.0 = standard log spacing (equal
    /// bars per octave). >1.0 shifts more bars to high frequencies
    /// (finer high-end detail, coarser low end); <1.0 does the opposite.
    #[arg(long, default_value_t = 1.0)]
    freq_warp: f32,

    /// FFT analysis window size (actual sample count used), must be a
    /// power of two. Mainly controls time resolution/latency.
    #[arg(long, default_value_t = 2048)]
    fft_size: usize,

    /// Zero-padding multiplier. Actual FFT length is fft_size * fft_pad,
    /// the rest is zero. Doesn't raise the true spectral resolution limit,
    /// but greatly increases sampling density so log-band mapping doesn't
    /// collapse many bars onto the same bin (the main cause of blocky
    /// low-frequency bars). Cheaper than raising fft_size since it adds
    /// no latency.
    #[arg(long, default_value_t = 4)]
    fft_pad: usize,

    /// Output frame rate (lines per second)
    #[arg(long, default_value_t = 60.0)]
    fps: f64,

    /// Attack smoothing coefficient (0..1). Closer to 1 = faster rise.
    /// Shared by all bars.
    #[arg(long, default_value_t = 0.9)]
    attack: f32,

    /// Base gravity (acceleration added to fall velocity each frame, same
    /// scale as bar output) for the lowest-frequency bar. Replaces the old
    /// exponential release: higher = faster/heavier fall, lower = slower,
    /// more jelly-like fall.
    #[arg(long, default_value_t = 0.5)]
    release: f32,

    /// Extra gravity multiplier for the highest-frequency bar (0..N). 0 =
    /// same base gravity for all bands; higher values give high-frequency
    /// bars (cymbals/hihats) a snappier, less trailing fall.
    #[arg(long, default_value_t = 0.4)]
    release_high_boost: f32,

    /// Gravity boost scaled by the bar's current normalized value (0..1,
    /// roughly this instant's loudness). 0 = gravity independent of
    /// volume; higher = louder hits fall back down faster/harder, while
    /// quiet tails fall slowest, preserving natural decay.
    #[arg(long, default_value_t = 2.0)]
    gravity_db_scale: f32,

    /// Max radius of cross-bar spatial smoothing. Lower (even 0-1) keeps
    /// more bar-to-bar detail; higher gives smoother hill-like shapes at
    /// the cost of detail.
    #[arg(long, default_value_t = 2)]
    spatial_smooth_max_radius: usize,

    /// Peak mix ratio (0..1) blended into RMS when aggregating wide
    /// (high-frequency) bands. 0 = pure RMS (smoothest, least noise, weaker
    /// transients); 1 = pure peak (sharpest detail, most noise).
    #[arg(long, default_value_t = 0.35)]
    peak_mix: f32,

    /// Dynamic range in dB below the running peak that still counts as
    /// "signal" and maps into the visible range. Lower (e.g. 30) = more
    /// contrast, only clearly loud signal lights up; higher (e.g. 60) =
    /// more forgiving but background noise can look constantly "on".
    #[arg(long, default_value_t = 36.0)]
    dynamic_range_db: f32,

    /// Rise speed (0..1, closer to 1 = slower rise) of each band's
    /// adaptive noise-floor tracker. Tracks each band's typical background
    /// level so only signal clearly above it lights up, avoiding a
    /// constant "half-lit" look across many bars.
    #[arg(long, default_value_t = 0.97)]
    noise_gate_rise: f32,

    /// Noise-gate threshold multiplier: signal must exceed the tracked
    /// floor times this factor to count. Higher = cleaner but may eat
    /// quiet detail; lower = more detail but more noise.
    #[arg(long, default_value_t = 1.4)]
    noise_gate_margin: f32,

    /// Number of decimal places in the output
    #[arg(long, default_value_t = 4)]
    precision: usize,

    /// Print diagnostics (raw/FFT/normalized peaks) to stderr once per
    /// second, to help locate where a signal drops to zero.
    #[arg(long, default_value_t = false)]
    debug: bool,

    /// Independence of the normalization reference per band (0..1).
    /// 0 = all bars share one global running peak as the reference
    ///     (bass spikes can dim quiet bands like hihat).
    /// 1 = each bar only compares against its own running peak, so bass
    ///     never affects hihat's reference.
    /// 0.6-0.85 is a good starting range: greatly reduces cross-band
    /// masking while keeping a bit of global feel, without letting quiet
    /// noisy bands sit permanently near full scale. Applies uniformly to
    /// all bands; no explicit frequency ranges needed.
    #[arg(long, default_value_t = 0.75)]
    band_independence: f32,
}

// ---------------------------------------------------------------------
// Pause control: SIGUSR2 -> main loop truly sleeps via Condvar;
//                SIGUSR1 -> wake and resume.
// ---------------------------------------------------------------------

struct PauseControl {
    paused: AtomicBool,
    lock: Mutex<()>,
    cond: Condvar,
}

impl PauseControl {
    fn new() -> Self {
        Self {
            paused: AtomicBool::new(false),
            lock: Mutex::new(()),
            cond: Condvar::new(),
        }
    }

    fn pause(&self) {
        self.paused.store(true, Ordering::SeqCst);
    }

    fn resume(&self) {
        self.paused.store(false, Ordering::SeqCst);
        // Broadcast wake, regardless of whether anyone is waiting.
        self.cond.notify_all();
    }

    fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    /// Called at the top of every frame in the main loop. If paused, this
    /// truly blocks here (Condvar::wait, futex-backed, not a busy loop)
    /// until resume() is called; CPU usage drops to ~0% while paused.
    fn block_while_paused(&self) {
        if !self.is_paused() {
            return;
        }
        let mut guard = self.lock.lock().unwrap();
        while self.is_paused() {
            guard = self.cond.wait(guard).unwrap();
        }
    }
}

/// Listens for SIGUSR1/SIGUSR2 and translates them into PauseControl state
/// changes. Runs a blocking signal iterator on its own thread, costing no
/// CPU while idle.
fn spawn_signal_listener(pause: Arc<PauseControl>) {
    thread::spawn(move || {
        let mut signals = Signals::new([SIGUSR1, SIGUSR2])
            .expect("failed to register SIGUSR1/SIGUSR2 handlers");

        for sig in signals.forever() {
            match sig {
                s if s == SIGUSR2 => {
                    pause.pause();
                }
                s if s == SIGUSR1 => {
                    pause.resume();
                }
                _ => {}
            }
        }
    });
}

// ---------------------------------------------------------------------
// Shared state: PipeWire callback only writes here, main thread only reads
// ---------------------------------------------------------------------

struct SharedAudio {
    buffer: Mutex<VecDeque<f32>>,
    sample_rate: AtomicU32,
    channels: AtomicU32,
}

impl SharedAudio {
    fn new() -> Self {
        Self {
            buffer: Mutex::new(VecDeque::with_capacity(1 << 16)),
            sample_rate: AtomicU32::new(48000),
            channels: AtomicU32::new(2),
        }
    }

    fn push_samples(&self, mono: &[f32], max_len: usize) {
        let mut buf = self.buffer.lock().unwrap();
        buf.extend(mono.iter().copied());
        // Drain any overflow in one shot instead of popping one at a time.
        let len = buf.len();
        if len > max_len {
            buf.drain(0..len - max_len);
        }
    }

    /// Returns the latest n samples (None if not enough yet). Does not
    /// clear the buffer (frames are allowed to overlap).
    fn latest(&self, n: usize) -> Option<Vec<f32>> {
        let mut buf = self.buffer.lock().unwrap();
        if buf.len() < n {
            return None;
        }
        let skip = buf.len() - n;
        // make_contiguous + slice copy avoids the per-element skip/iter
        // overhead of collecting through an iterator.
        Some(buf.make_contiguous()[skip..].to_vec())
    }
}

// ---------------------------------------------------------------------
// PipeWire user data
// ---------------------------------------------------------------------

struct StreamUserData {
    format: AudioInfoRaw,
    shared: Arc<SharedAudio>,
    max_buffer_len: usize,
    pause: Arc<PauseControl>,
}

// ---------------------------------------------------------------------
// Log-band frequency mapping
// ---------------------------------------------------------------------

/// Computes num_bars+1 log-spaced frequency edges.
/// `warp` shifts bar allocation away from "equal bars per octave":
/// - warp = 1.0: standard log spacing
/// - warp > 1.0: more bars allocated to high frequencies
/// - warp < 1.0: more bars allocated to low frequencies
fn log_band_edges(num_bars: usize, min_freq: f32, max_freq: f32, warp: f32) -> Vec<f32> {
    let min_freq = min_freq.max(1.0);
    let max_freq = max_freq.max(min_freq * 1.01);
    let ratio = (max_freq / min_freq).ln();
    let warp = warp.max(0.01);
    (0..=num_bars)
        .map(|i| {
            let t = i as f32 / num_bars as f32;
            // f(t) = 1 - (1-t)^warp: warp>1 flattens the slope near t=1
            // (high freq), meaning smaller frequency gaps there -> more
            // bars allocated to high frequencies.
            let ft = 1.0 - (1.0 - t).powf(warp);
            min_freq * (ratio * ft).exp()
        })
        .collect()
}

/// Linearly interpolates the magnitude spectrum at an arbitrary frequency
/// (smoother than nearest-bin lookup; important for narrow bands to avoid
/// repeated stair-stepped values at low frequencies).
fn interpolate_magnitude(magnitudes: &[f32], bin_hz: f32, freq: f32) -> f32 {
    if magnitudes.is_empty() {
        return 0.0;
    }
    let pos = (freq / bin_hz).max(0.0);
    let i0 = pos.floor() as usize;
    if i0 + 1 >= magnitudes.len() {
        return *magnitudes.last().unwrap();
    }
    let frac = pos - i0 as f32;
    let a = magnitudes[i0];
    let b = magnitudes[i0 + 1];
    a + (b - a) * frac
}

/// Aggregates the FFT magnitude spectrum (positive frequencies only) into
/// num_bars log-spaced bars using precomputed band `edges`.
///
/// - Narrow bands (<=2 raw bins, usually low frequencies): use interpolated
///   magnitude at the band's center frequency for a smooth curve, avoiding
///   many bars sharing one bin.
/// - Wide bands (many raw bins, usually high frequencies): use RMS instead
///   of peak to suppress single-bin noise spikes, while still reacting to
///   wideband transients like drums/cymbals.
fn fft_to_log_bars(
    magnitudes: &[f32],
    bin_hz: f32,
    edges: &[f32],
    peak_mix: f32,
) -> Vec<f32> {
    let num_bars = edges.len() - 1;
    let mut bars = vec![0.0f32; num_bars];
    let peak_mix = peak_mix.clamp(0.0, 1.0);

    for i in 0..num_bars {
        let f_lo = edges[i];
        let f_hi = edges[i + 1];

        let bin_lo = (f_lo / bin_hz).floor() as usize;
        let bin_hi_raw = (f_hi / bin_hz).ceil() as usize;
        let bin_hi = bin_hi_raw.min(magnitudes.len());

        if bin_lo >= magnitudes.len() {
            bars[i] = 0.0;
            continue;
        }

        let width = bin_hi.saturating_sub(bin_lo);

        if width <= 2 {
            // Narrow band: interpolate at the center frequency (geometric
            // mean, matching the log scale).
            let f_center = (f_lo * f_hi).sqrt();
            bars[i] = interpolate_magnitude(magnitudes, bin_hz, f_center);
        } else {
            // Wide band: RMS suppresses noise, with a bit of peak mixed
            // back in to keep transient detail.
            let slice = &magnitudes[bin_lo..bin_hi];
            let sum_sq: f32 = slice.iter().map(|v| v * v).sum();
            let rms = (sum_sq / slice.len() as f32).sqrt();
            let peak = slice.iter().copied().fold(0.0f32, f32::max);
            bars[i] = rms * (1.0 - peak_mix) + peak * peak_mix;
        }
    }

    bars
}

/// Cross-bar spatial smoothing. Radius grows linearly with bar index
/// (frequency): low bars keep sharp shape, high bars get more smoothing
/// to remove jagged noise.
fn spatial_smooth(bars: &[f32], max_radius: usize) -> Vec<f32> {
    let n = bars.len();
    if n == 0 || max_radius == 0 {
        return bars.to_vec();
    }
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let t = i as f32 / (n as f32 - 1.0).max(1.0);
        let radius = 1 + ((max_radius as f32) * t).round() as usize;
        let lo = i.saturating_sub(radius);
        let hi = (i + radius).min(n - 1);

        // Triangular weights: closer to center = higher weight, keeps a
        // bit of the peak's sharpness.
        let mut sum = 0.0f32;
        let mut weight_sum = 0.0f32;
        for j in lo..=hi {
            let dist = (j as isize - i as isize).unsigned_abs() as f32;
            let w = (radius as f32 + 1.0 - dist).max(0.0);
            sum += bars[j] * w;
            weight_sum += w;
        }
        out[i] = if weight_sum > 0.0 { sum / weight_sum } else { bars[i] };
    }
    out
}

/// Per-band adaptive noise gate (minimum follower).
/// - Value below current floor -> floor snaps down immediately (tracks
///   real quiet levels fast).
/// - Value above floor -> floor creeps up slowly (avoids mistaking real
///   music for noise).
/// The effective signal is "value - floor*margin", which prevents quiet
/// bands from looking permanently "half-lit" due to compressed dynamic
/// range.
struct NoiseGate {
    floor: Vec<f32>,
    rise: f32,
    margin: f32,
}

impl NoiseGate {
    fn new(num_bars: usize, rise: f32, margin: f32) -> Self {
        Self {
            floor: vec![1e-6; num_bars],
            rise: rise.clamp(0.0, 0.999),
            margin: margin.max(1.0),
        }
    }

    fn apply(&mut self, raw: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; raw.len()];
        for i in 0..raw.len() {
            let v = raw[i];
            if v < self.floor[i] {
                self.floor[i] = v;
            } else {
                self.floor[i] = self.floor[i] * self.rise + v * (1.0 - self.rise);
            }
            out[i] = (v - self.floor[i] * self.margin).max(0.0);
        }
        out
    }
}

// ---------------------------------------------------------------------
// Normalization: log-compress dynamic range -> map to [0.001, bar_max]
// ---------------------------------------------------------------------

struct Normalizer {
    running_max: f32,            // global running peak (legacy behavior)
    band_running_max: Vec<f32>,  // each bar's own running peak
    decay: f32,
    dynamic_range_db: f32,
    band_independence: f32,
}

impl Normalizer {
    fn new(num_bars: usize, dynamic_range_db: f32, band_independence: f32) -> Self {
        Self {
            running_max: 1e-3,
            band_running_max: vec![1e-3; num_bars],
            decay: 0.995,
            dynamic_range_db: dynamic_range_db.max(1.0),
            band_independence: band_independence.clamp(0.0, 1.0),
        }
    }

    fn normalize(&mut self, raw: &[f32], bar_max: f32) -> Vec<f32> {
        let frame_max = raw.iter().copied().fold(0.0f32, f32::max).max(1e-6);
        self.running_max = (self.running_max * self.decay).max(frame_max);

        let b = self.band_independence;

        raw.iter()
            .enumerate()
            .map(|(i, &v)| {
                let bv = v.max(1e-6);
                self.band_running_max[i] = (self.band_running_max[i] * self.decay).max(bv);

                // Geometric interpolation: b=0 uses the global peak only,
                // b=1 uses only this bar's own peak, in between blends both
                // (natural on a dB/log scale).
                let reference = self.running_max.powf(1.0 - b) * self.band_running_max[i].powf(b);

                let db = 20.0 * (v.max(1e-8) / reference).log10();
                let norm = ((db + self.dynamic_range_db) / self.dynamic_range_db).clamp(0.0, 1.0);
                (norm * bar_max).clamp(0.001, bar_max)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------
// Frequency/volume-aware attack / gravity-style fall smoothing
// ---------------------------------------------------------------------
//
// Attack keeps the original exponential approach (fast reaction, no need
// for physics). Fall now uses a "gravity" model instead of exponential
// release:
//   1. Add gravity to fall velocity each frame (velocity += gravity)
//   2. value -= velocity
//   3. If it overshoots the frame's real target, snap to the target and
//      zero the velocity ("landed"; next fall restarts from rest)
//
// This gives a slow-then-fast falling curve like a real thrown object,
// instead of the old exponential smoothing's fast-then-slow curve.
//
// Gravity strength combines two factors:
//   - Frequency (release_high_boost): higher bars get stronger base
//     gravity for a snappier cymbal/hihat cutoff.
//   - Volume/dB (gravity_db_scale): the louder a bar's current value, the
//     stronger its fall gravity -> loud hits snap back down fast, quiet
//     tails fall slowest, preserving natural decay.
struct BarState {
    value: Vec<f32>,
    velocity: Vec<f32>,
    attack: Vec<f32>,
    /// Each bar's base gravity, already adjusted by release_high_boost.
    base_gravity: Vec<f32>,
    /// Multiplier for gravity boost by volume (0..1, value/bar_max).
    gravity_db_scale: f32,
    bar_max: f32,
}

impl BarState {
    fn new(
        num_bars: usize,
        floor: f32,
        base_attack: f32,
        base_gravity: f32,
        gravity_high_boost: f32,
        gravity_db_scale: f32,
        bar_max: f32,
    ) -> Self {
        let n = num_bars.max(1);
        let attack = vec![base_attack.clamp(0.0, 1.0); num_bars];
        let base_gravity: Vec<f32> = (0..num_bars)
            .map(|i| {
                let t = i as f32 / (n as f32 - 1.0).max(1.0);
                (base_gravity * (1.0 + gravity_high_boost * t)).max(0.0)
            })
            .collect();

        Self {
            value: vec![floor; num_bars],
            velocity: vec![0.0; num_bars],
            attack,
            base_gravity,
            gravity_db_scale: gravity_db_scale.max(0.0),
            bar_max: bar_max.max(1e-6),
        }
    }

    fn update(&mut self, new: &[f32]) {
        for i in 0..self.value.len() {
            let n = new[i];
            let s = self.value[i];

            if n >= s {
                // Attack: approach the new (higher) target quickly and
                // reset fall velocity so the next fall restarts from rest.
                self.value[i] = s + (n - s) * self.attack[i];
                self.velocity[i] = 0.0;
            } else {
                // Fall: gravity scales with how loud the current value is.
                let loudness = (s / self.bar_max).clamp(0.0, 1.0);
                let gravity = self.base_gravity[i] * (1.0 + self.gravity_db_scale * loudness);

                self.velocity[i] += gravity;

                let mut v = s - self.velocity[i];
                if v <= n {
                    // Overshot (or reached) this frame's target -> landed,
                    // zero the velocity.
                    v = n;
                    self.velocity[i] = 0.0;
                }
                self.value[i] = v;
            }
        }
    }
}

// ---------------------------------------------------------------------
// main
// ---------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if !args.fft_size.is_power_of_two() {
        eprintln!("--fft-size must be a power of two (e.g. 1024, 2048, 4096)");
        std::process::exit(1);
    }
    if args.fft_pad == 0 {
        eprintln!("--fft-pad must be at least 1");
        std::process::exit(1);
    }

    let padded_size = args.fft_size * args.fft_pad;

    let shared = Arc::new(SharedAudio::new());
    let max_buffer_len = args.fft_size * 8;
    let pause = Arc::new(PauseControl::new());

    // Signal listener thread: SIGUSR2 pause / SIGUSR1 resume.
    spawn_signal_listener(Arc::clone(&pause));

    // PipeWire main loop on its own thread; process callback only downmixes
    // and pushes samples.
    {
        let shared = Arc::clone(&shared);
        let pause = Arc::clone(&pause);
        let sink_target = args.sink.clone();

        thread::spawn(move || {
            if let Err(e) = run_pipewire_capture(sink_target, shared, max_buffer_len, pause) {
                eprintln!("pipewire capture error: {e}");
                std::process::exit(1);
            }
        });
    }

    // Main thread: wake at a fixed fps, grab latest samples -> zero-padded
    // FFT -> bars -> print.
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(padded_size);

    // Hann window, applied only to the real sample region (first
    // fft_size); the zero-padded tail stays at 0.
    let window: Vec<f32> = (0..args.fft_size)
        .map(|i| {
            0.5 - 0.5
                * ((2.0 * std::f32::consts::PI * i as f32) / (args.fft_size as f32 - 1.0)).cos()
        })
        .collect();

    // Precompute log-band edges once: they only depend on static CLI args,
    // not per-frame data, so this must not be recomputed inside the loop.
    let band_edges = log_band_edges(args.num_bars, args.min_freq, args.max_freq, args.freq_warp);

    let mut normalizer = Normalizer::new(args.num_bars, args.dynamic_range_db, args.band_independence);
    let mut noise_gate = NoiseGate::new(args.num_bars, args.noise_gate_rise, args.noise_gate_margin);
    let mut bar_state = BarState::new(
        args.num_bars,
        0.001,
        args.attack,
        args.release,
        args.release_high_boost,
        args.gravity_db_scale,
        args.bar_max,
    );

    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());

    let frame_interval = Duration::from_secs_f64(1.0 / args.fps);
    let mut fft_buf = vec![Complex32::new(0.0, 0.0); padded_size];
    // Reused output line buffer to avoid a fresh allocation every frame.
    let mut line_buf = String::with_capacity(args.num_bars * 8);

    let mut debug_frame_counter: u64 = 0;
    let debug_every = args.fps.max(1.0) as u64;

    loop {
        // Truly sleeps here (Condvar::wait) while paused, until SIGUSR1
        // wakes the main thread again.
        pause.block_while_paused();

        let frame_start = Instant::now();

        match shared.latest(args.fft_size) {
            None => {
                if args.debug {
                    eprintln!(
                        "[debug] not enough samples yet, waiting (buffer_len={})",
                        shared.buffer.lock().unwrap().len()
                    );
                }
            }
            Some(samples) => {
                let sample_rate = shared.sample_rate.load(Ordering::Relaxed) as f32;
                let bin_hz = sample_rate / padded_size as f32;
                let raw_peak = samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));

                // Window the first fft_size samples; rest (zero-padding)
                // stays 0.
                for (i, &s) in samples.iter().enumerate() {
                    fft_buf[i] = Complex32::new(s * window[i], 0.0);
                }
                for c in &mut fft_buf[args.fft_size..] {
                    *c = Complex32::new(0.0, 0.0);
                }

                fft.process(&mut fft_buf);

                let half = padded_size / 2;
                let magnitudes: Vec<f32> = fft_buf[..half].iter().map(|c| c.norm()).collect();
                let mag_peak = magnitudes.iter().copied().fold(0.0f32, f32::max);

                let raw_bars = fft_to_log_bars(
                    &magnitudes,
                    bin_hz,
                    &band_edges,
                    args.peak_mix,
                );

                let smoothed_bars = spatial_smooth(&raw_bars, args.spatial_smooth_max_radius);
                let gated_bars = noise_gate.apply(&smoothed_bars);

                let normalized = normalizer.normalize(&gated_bars, args.bar_max);
                let norm_peak = normalized.iter().copied().fold(0.0f32, f32::max);
                bar_state.update(&normalized);

                if args.debug {
                    debug_frame_counter += 1;
                    if debug_frame_counter % debug_every == 0 {
                        let floor_avg: f32 =
                            noise_gate.floor.iter().sum::<f32>() / noise_gate.floor.len() as f32;
                        eprintln!(
                            "[debug] sample_rate={} bin_hz={:.3} raw_sample_peak={:.6} fft_mag_peak={:.6} noise_floor_avg={:.6} running_max={:.6} normalized_peak={:.6}",
                            sample_rate, bin_hz, raw_peak, mag_peak, floor_avg, normalizer.running_max, norm_peak
                        );
                    }
                }

                write_frame_text(&mut writer, &mut line_buf, &bar_state.value, args.precision)?;
            }
        }

        let elapsed = frame_start.elapsed();
        if elapsed < frame_interval {
            thread::sleep(frame_interval - elapsed);
        }
    }
}

/// Writes one CSV line of bar values into `writer`, reusing `line_buf`
/// across calls to avoid a per-frame String allocation, and formatting
/// each value directly into the buffer instead of through a temporary
/// `format!` String per value.
fn write_frame_text<W: Write>(
    writer: &mut W,
    line_buf: &mut String,
    bars: &[f32],
    precision: usize,
) -> io::Result<()> {
    line_buf.clear();
    for (i, &v) in bars.iter().enumerate() {
        if i > 0 {
            line_buf.push(',');
        }
        let _ = write!(line_buf, "{:.*}", precision, v);
    }
    line_buf.push('\n');
    writer.write_all(line_buf.as_bytes())?;
    writer.flush()
}

// ---------------------------------------------------------------------
// PipeWire capture
// ---------------------------------------------------------------------

fn run_pipewire_capture(
    sink_target: String,
    shared: Arc<SharedAudio>,
    max_buffer_len: usize,
    pause: Arc<PauseControl>,
) -> Result<(), pw::Error> {
    pw::init();

    let mainloop = pw::main_loop::MainLoop::new(None)?;
    let context = pw::context::Context::new(&mainloop)?;
    let core = context.connect(None)?;

    let user_data = StreamUserData {
        format: Default::default(),
        shared: Arc::clone(&shared),
        max_buffer_len,
        pause,
    };

    // Users typically type PulseAudio-style "xxx.monitor" names (as
    // cava/pactl do), but PipeWire's native node name has no ".monitor"
    // suffix (that's just the monitor_FL/monitor_FR ports on the sink
    // node). Strip it to get the real sink node name.
    let sink_node_name = sink_target
        .strip_suffix(".monitor")
        .unwrap_or(&sink_target)
        .to_string();

    let props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Music",
        // MEDIA_CLASS tells wireplumber this is an audio-output capture
        // stream; without it target.object hints are often ignored and
        // the process callback only ever sees silence.
        *pw::keys::MEDIA_CLASS => "Stream/Input/Audio",
        // target.object / node.target must point at the sink's own node
        // name (no ".monitor" suffix).
        "target.object" => sink_node_name.as_str(),
        "node.target" => sink_node_name.as_str(),
        // Tells wireplumber this Audio Input should capture the sink's
        // monitor output (what it's playing), not a regular mic source.
        "stream.capture.sink" => "true",
    };

    let stream = pw::stream::Stream::new(&core, "pw-visualizer-capture", props)?;

    let _listener = stream
        .add_local_listener_with_user_data(user_data)
        .state_changed(|_stream, _user_data, old, new| {
            eprintln!("[pipewire] stream state changed: {:?} -> {:?}", old, new);
        })
        .param_changed(|_stream, user_data, id, param| {
            let Some(param) = param else { return };
            if id != spa::param::ParamType::Format.as_raw() {
                return;
            }

            let (media_type, media_subtype) = match spa::param::format_utils::parse_format(param) {
                Ok(v) => v,
                Err(_) => return,
            };

            if media_type != spa::param::format::MediaType::Audio
                || media_subtype != spa::param::format::MediaSubtype::Raw
            {
                return;
            }

            user_data
                .format
                .parse(param)
                .expect("failed to parse AudioInfoRaw from param");

            user_data
                .shared
                .sample_rate
                .store(user_data.format.rate(), Ordering::Relaxed);
            user_data
                .shared
                .channels
                .store(user_data.format.channels(), Ordering::Relaxed);

            eprintln!(
                "capturing: rate={} channels={}",
                user_data.format.rate(),
                user_data.format.channels()
            );
        })
        .process(|stream, user_data| {
            let mut buffer = match stream.dequeue_buffer() {
                None => return,
                Some(b) => b,
            };

            // While paused, return the buffer as-is (auto-returned on
            // drop) and skip the downmix work to keep this thread's CPU
            // usage low.
            if user_data.pause.is_paused() {
                return;
            }

            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }
            let data = &mut datas[0];

            let n_channels = user_data.format.channels().max(1) as usize;
            let chunk_size = data.chunk().size() as usize;
            let n_samples_total = chunk_size / mem::size_of::<f32>();

            let Some(samples) = data.data() else { return };

            let n_frames = n_samples_total / n_channels;
            let mut mono = Vec::with_capacity(n_frames);

            for f in 0..n_frames {
                let mut sum = 0.0f32;
                for c in 0..n_channels {
                    let idx = f * n_channels + c;
                    let start = idx * mem::size_of::<f32>();
                    let end = start + mem::size_of::<f32>();
                    if end > samples.len() {
                        break;
                    }
                    let bytes: [u8; 4] = samples[start..end].try_into().unwrap();
                    sum += f32::from_le_bytes(bytes);
                }
                mono.push(sum / n_channels as f32);
            }

            user_data.shared.push_samples(&mono, user_data.max_buffer_len);
        })
        .register()?;

    let mut audio_info = AudioInfoRaw::new();
    audio_info.set_format(AudioFormat::F32LE);

    let obj = spa::pod::Object {
        type_: spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: spa::param::ParamType::EnumFormat.as_raw(),
        properties: audio_info.into(),
    };

    let values: Vec<u8> = spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &spa::pod::Value::Object(obj),
    )
    .unwrap()
    .0
    .into_inner();

    let mut params = [Pod::from_bytes(&values).unwrap()];

    stream.connect(
        spa::utils::Direction::Input,
        None,
        StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS | StreamFlags::RT_PROCESS,
        &mut params,
    )?;

    mainloop.run();

    Ok(())
}
