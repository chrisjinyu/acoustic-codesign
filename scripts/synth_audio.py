"""synthesize_audio.py -- turn the saved FRFs into audible WAV files.

Reads the FRF magnitude curves stored by demo.ipynb in outputs/best_params.npz
and renders them as audio so design results can be evaluated by ear.

Pipeline
--------
For each available FRF |H(jw)| the script:

  1. Reconstructs a minimum-phase complex spectrum from the saved magnitude
     using the cepstral method (real cepstrum + causal window + exp).

  2. Resamples to a full [0, fs/2] grid with half-cosine tapers at the band
     edges to avoid ringing from the abrupt cutoff at 40 and 1500 Hz.

  3. IFFTs to a real impulse response h[n], then renders four output flavors:

       _impulse.wav
           The raw impulse response. Sounds like striking the plate once at
           the bridge. Short by design -- plate damping (zeta=0.02) decays
           each mode in roughly 1/(zeta*f) seconds.

       _pluck.wav
           IR convolved with a 50 ms decaying noise burst.

       _tonal.wav  (primary sustained output)
           A sustained chord of sinusoids at the TARGET PEAK FREQUENCIES is
           filtered through the plate IR for several seconds. This is the
           most useful perceptual comparison: you hear exactly the target
           pitches, and the FRF determines how loudly each comes through.
           The passive plate may pass some weakly; the co-designed plate
           should pass them closer to the target level.

           LQR mode:     frequencies from results.json lqr_target_centers.
           Strings mode: optimised string fundamentals + harmonics.

       _noise.wav  (spectral portrait)
           White noise filtered through the plate IR. Every frequency is
           excited equally; the plate's coloring is entirely responsible for
           the output spectrum. The closest thing to a continuous audio
           portrait of |H(w)|.

Usage
-----
    python scripts/synthesize_audio.py
    python scripts/synthesize_audio.py --sustained-duration 6
    python scripts/synthesize_audio.py --no-impulse --no-pluck
    python scripts/synthesize_audio.py --sample-rate 22050 --n-fft 4096

Inputs (outputs/):
    best_params.npz   -- freqs_lqr, target_lqr, H_passive_lqr, H_lqr,
                         freqs_str, target_str_nominal, H_passive_str, H_str
    results.json      -- lqr_target_centers, actual_pitches_hz

Outputs (outputs/audio/):
    lqr_{target,passive,codesign}_{impulse,pluck,tonal,noise}.wav
    strings_{target,passive,codesign}_{impulse,pluck,tonal,noise}.wav
"""
from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

REPO_ROOT   = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
AUDIO_DIR   = OUTPUTS_DIR / "audio"


# ---------------------------------------------------------------------------
# Spectrum construction
# ---------------------------------------------------------------------------

def build_audio_spectrum(freqs_orig, H_mag, fs, n_fft,
                         taper_width_hz=20.0, mag_floor=1e-6):
    """Resample a band-limited |H| onto a uniform [0, fs/2] grid of length
    n_fft//2 + 1, with smooth half-cosine tapers at both band edges."""
    freqs_full = np.linspace(0.0, fs / 2.0, n_fft // 2 + 1)
    f_lo = float(freqs_orig.min())
    f_hi = float(freqs_orig.max())

    H_full = np.interp(freqs_full, freqs_orig, H_mag,
                       left=mag_floor, right=mag_floor)

    lo_band = (freqs_full >= f_lo - taper_width_hz) & (freqs_full < f_lo)
    if lo_band.any():
        u = (freqs_full[lo_band] - (f_lo - taper_width_hz)) / taper_width_hz
        ramp = 0.5 * (1.0 - np.cos(np.pi * u))
        H_at_flo = float(np.interp(f_lo, freqs_orig, H_mag))
        H_full[lo_band] = mag_floor + (H_at_flo - mag_floor) * ramp

    hi_band = (freqs_full > f_hi) & (freqs_full <= f_hi + taper_width_hz)
    if hi_band.any():
        u = (freqs_full[hi_band] - f_hi) / taper_width_hz
        ramp = 0.5 * (1.0 + np.cos(np.pi * u))
        H_at_fhi = float(np.interp(f_hi, freqs_orig, H_mag))
        H_full[hi_band] = mag_floor + (H_at_fhi - mag_floor) * ramp

    return freqs_full, np.maximum(H_full, mag_floor)


def minimum_phase_impulse_response(H_mag_half, n_fft):
    """Build a real, causal, minimum-phase IR from a half-band magnitude
    spectrum (length n_fft//2 + 1) using the cepstral method."""
    H_full   = np.concatenate([H_mag_half, H_mag_half[-2:0:-1]])
    assert H_full.shape[0] == n_fft
    cepstrum = np.fft.ifft(np.log(H_full)).real
    window              = np.zeros(n_fft)
    window[0]           = 1.0
    window[1:n_fft//2]  = 2.0
    window[n_fft//2]    = 1.0
    return np.fft.ifft(np.exp(np.fft.fft(cepstrum * window))).real


# ---------------------------------------------------------------------------
# Excitation generators
# ---------------------------------------------------------------------------

def make_pluck_excitation(fs, duration_ms=50.0, decay_ratio=0.005, seed=0):
    """Short noise burst with exponential decay."""
    rng   = np.random.default_rng(seed)
    n     = int(fs * duration_ms / 1000.0)
    decay = np.exp(np.linspace(0.0, np.log(decay_ratio), n))
    return rng.standard_normal(n) * decay


def make_tonal_excitation(fs, frequencies_hz, duration_s,
                          attack_s=0.05, release_s=0.3):
    """Sustained sum of sinusoids at the specified frequencies.

    Each sinusoid runs for `duration_s` seconds with a soft attack and release.
    Random initial phases prevent the partials from all peaking at t=0.
    The FRF filter applied downstream does all the spectral shaping -- this
    excitation is intentionally flat across the supplied frequencies.
    """
    n   = int(fs * duration_s)
    t   = np.arange(n) / float(fs)
    sig = np.zeros(n)
    rng = np.random.default_rng(42)
    for f in frequencies_hz:
        if 0 < f < fs / 2.0:
            sig += np.sin(2.0 * np.pi * f * t + rng.uniform(0.0, 2.0 * np.pi))
    if np.max(np.abs(sig)) > 0:
        sig /= np.max(np.abs(sig))
    n_att, n_rel = max(1, int(fs * attack_s)), max(1, int(fs * release_s))
    envelope = np.ones(n)
    envelope[:n_att]  = np.linspace(0.0, 1.0, n_att)
    envelope[-n_rel:] = np.linspace(1.0, 0.0, n_rel)
    return sig * envelope


def make_noise_excitation(fs, duration_s, seed=1):
    """White noise with soft fade-in and fade-out."""
    rng    = np.random.default_rng(seed)
    n      = int(fs * duration_s)
    sig    = rng.standard_normal(n)
    n_fade = int(fs * 0.05)
    sig[:n_fade]  *= np.linspace(0.0, 1.0, n_fade)
    sig[-n_fade:] *= np.linspace(1.0, 0.0, n_fade)
    return sig


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def normalize_for_wav(audio, headroom_db=-1.0):
    peak = np.max(np.abs(audio))
    if peak < 1e-12:
        return audio.astype(np.float32)
    return (audio * (10.0 ** (headroom_db / 20.0) / peak)).astype(np.float32)


def write_wav_int16(path, fs, audio):
    wavfile.write(path, fs, (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16))


# ---------------------------------------------------------------------------
# Core render function
# ---------------------------------------------------------------------------

def render_frf(freqs, H_mag, fs, n_fft,
               transient_duration_s, sustained_duration_s,
               tonal_frequencies,
               do_impulse, do_pluck, do_tonal, do_noise):
    """Build a minimum-phase IR from |H(jw)| and render all four flavors.

    Parameters
    ----------
    transient_duration_s : output length for impulse and pluck.
    sustained_duration_s : output length for tonal and noise.
    tonal_frequencies    : list of Hz values for the sustained tonal excitation.

    Returns
    -------
    dict mapping variant name -> normalized float32 audio array.
    """
    _, H_audio = build_audio_spectrum(freqs, H_mag, fs, n_fft)
    h_ir       = minimum_phase_impulse_response(H_audio, n_fft)

    n_trans = int(fs * transient_duration_s)
    n_sust  = int(fs * sustained_duration_s)
    n_max   = max(n_trans, n_sust)

    # Extend the IR array so convolution with the sustained excitation works
    # correctly without aliasing; pad with zeros beyond the n_fft window.
    if n_max > n_fft:
        h_ir = np.concatenate([h_ir, np.zeros(n_max - n_fft)])

    out = {}

    if do_impulse:
        out["impulse"] = normalize_for_wav(h_ir[:n_trans])

    if do_pluck:
        exc = make_pluck_excitation(fs)
        out["pluck"] = normalize_for_wav(
            fftconvolve(exc, h_ir, mode="full")[:n_trans])

    if do_tonal and tonal_frequencies:
        exc = make_tonal_excitation(fs, tonal_frequencies, sustained_duration_s)
        out["tonal"] = normalize_for_wav(
            fftconvolve(exc, h_ir, mode="full")[:n_sust])

    if do_noise:
        exc = make_noise_excitation(fs, sustained_duration_s)
        out["noise"] = normalize_for_wav(
            fftconvolve(exc, h_ir, mode="full")[:n_sust])

    return out


# ---------------------------------------------------------------------------
# Mode-specific drivers
# ---------------------------------------------------------------------------

def synthesize_lqr(npz, results, args):
    if "freqs_lqr" not in npz.files or len(np.asarray(npz["freqs_lqr"])) == 0:
        print("  LQR FRFs not present in best_params.npz -- skipping.")
        return

    freqs = np.asarray(npz["freqs_lqr"])
    items = [
        ("target",   np.asarray(npz["target_lqr"])),
        ("passive",  np.asarray(npz["H_passive_lqr"])),
        ("codesign", np.asarray(npz["H_lqr"])),
    ]

    # Tonal frequencies from the LQR target Gaussian centers
    tonal_freqs = []
    if results is not None:
        centers = results.get("config", {}).get("lqr_target_centers", [])
        tonal_freqs = [float(f) for f in centers]
    if not tonal_freqs:
        # Fall back: estimate peaks from the saved target FRF
        from scipy.signal import find_peaks
        target_H = np.asarray(npz["target_lqr"])
        peaks, _ = find_peaks(target_H, height=target_H.max() * 0.3)
        tonal_freqs = [float(freqs[p]) for p in peaks]
    print(f"  LQR tonal frequencies: {[f'{f:.0f}' for f in tonal_freqs]} Hz")

    # Pre-build the tonal excitation once so all three FRFs use the same signal.
    tonal_exc = (make_tonal_excitation(
                     args.sample_rate, tonal_freqs, args.sustained_duration)
                 if (not args.no_tonal and tonal_freqs) else None)

    for name, H in items:
        if H.size == 0:
            continue

        rendered = render_frf(
            freqs, H, args.sample_rate, args.n_fft,
            transient_duration_s=args.duration,
            sustained_duration_s=args.sustained_duration,
            tonal_frequencies=tonal_freqs,
            do_impulse=not args.no_impulse,
            do_pluck=not args.no_pluck,
            # Skip FRF-filtered tonal for "target" -- see override below.
            do_tonal=(not args.no_tonal and name != "target"),
            do_noise=not args.no_noise,
        )

        # For the target, the tonal reference is the raw excitation itself --
        # pure sinusoids at exactly the specified frequencies, uncolored by any
        # FRF. This is the ground truth the optimizer is trying to reach.
        # Filtering the target Gaussian through cepstral reconstruction
        # produces ringing artefacts because its peaks are only 20-40 Hz wide.
        if name == "target" and tonal_exc is not None and not args.no_tonal:
            rendered["tonal"] = normalize_for_wav(tonal_exc.copy())

        for variant, audio in rendered.items():
            path = AUDIO_DIR / f"lqr_{name}_{variant}.wav"
            write_wav_int16(path, args.sample_rate, audio)
            print(f"  wrote {path.name}  ({audio.size / args.sample_rate:.2f} s)")


def synthesize_strings(npz, results, args):
    if "freqs_str" not in npz.files or len(np.asarray(npz["freqs_str"])) == 0:
        print("  Strings FRFs not present in best_params.npz -- skipping.")
        return

    freqs = np.asarray(npz["freqs_str"])
    items = [
        ("target",   np.asarray(npz["target_str_nominal"])),
        ("passive",  np.asarray(npz["H_passive_str"])),
        ("codesign", np.asarray(npz["H_str"])),
    ]

    # Tonal frequencies: optimized string fundamentals + harmonics
    tonal_freqs = []
    if results is not None and "actual_pitches_hz" in results:
        fundamentals = results["actual_pitches_hz"]
        for f0 in fundamentals:
            for k in range(1, args.string_harmonics + 1):
                fk = k * float(f0)
                if fk < 0.95 * args.sample_rate / 2.0:
                    tonal_freqs.append(fk)
        print(f"  Strings tonal fundamentals: "
              f"{[f'{f:.1f}' for f in fundamentals]} Hz")
        print(f"  Total partials (fund + {args.string_harmonics - 1} harmonics): "
              f"{len(tonal_freqs)}")
    else:
        print("  Could not determine string fundamentals; "
              "tonal rendering will be skipped.")

    # Pre-build once so all three FRFs convolve with identical excitation.
    tonal_exc_str = (make_tonal_excitation(
                         args.sample_rate, tonal_freqs, args.sustained_duration)
                     if (not args.no_tonal and tonal_freqs) else None)

    for name, H in items:
        if H.size == 0:
            continue

        rendered = render_frf(
            freqs, H, args.sample_rate, args.n_fft,
            transient_duration_s=args.duration,
            sustained_duration_s=args.sustained_duration,
            tonal_frequencies=tonal_freqs,
            do_impulse=not args.no_impulse,
            do_pluck=not args.no_pluck,
            do_tonal=(not args.no_tonal and name != "target"),
            do_noise=not args.no_noise,
        )

        # Same reasoning as LQR: the target tonal is the raw string harmonics
        # at the optimized pitches, not filtered through the target spectrum.
        # The target spectrum is derived from string tensions and has narrow
        # harmonic peaks that cause the same cepstral ringing issue.
        if name == "target" and tonal_exc_str is not None and not args.no_tonal:
            rendered["tonal"] = normalize_for_wav(tonal_exc_str.copy())

        for variant, audio in rendered.items():
            path = AUDIO_DIR / f"strings_{name}_{variant}.wav"
            write_wav_int16(path, args.sample_rate, audio)
            print(f"  wrote {path.name}  ({audio.size / args.sample_rate:.2f} s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Synthesize audio from saved FRFs. "
                    "Primary outputs are _tonal.wav and _noise.wav.")
    parser.add_argument("--npz", type=str,
                        default=str(OUTPUTS_DIR / "best_params.npz"))
    parser.add_argument("--results", type=str,
                        default=str(OUTPUTS_DIR / "results.json"))
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--n-fft", type=int, default=8192)
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Duration in seconds for impulse and pluck outputs.")
    parser.add_argument("--sustained-duration", type=float, default=5.0,
                        help="Duration in seconds for tonal and noise outputs. "
                             "5 s gives a clear impression; use 8-10 for demos.")
    parser.add_argument("--string-harmonics", type=int, default=4,
                        help="Harmonics per string in the strings tonal excitation.")
    parser.add_argument("--no-impulse", action="store_true")
    parser.add_argument("--no-pluck",   action="store_true")
    parser.add_argument("--no-tonal",   action="store_true")
    parser.add_argument("--no-noise",   action="store_true")
    parser.add_argument("--audio-dir",  type=str, default=None)
    args = parser.parse_args()

    global AUDIO_DIR
    if args.audio_dir:
        AUDIO_DIR = Path(args.audio_dir)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found. Run demo.ipynb first.")
        sys.exit(1)

    results = None
    results_path = Path(args.results)
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    npz = np.load(npz_path, allow_pickle=True)

    print("=" * 60)
    print("Audio synthesis from saved FRFs")
    print("=" * 60)
    print(f"Source      : {npz_path}")
    print(f"Audio output: {AUDIO_DIR}")
    print(f"Sample rate : {args.sample_rate} Hz")
    print(f"Transient   : {args.duration:.1f} s  (impulse, pluck)")
    print(f"Sustained   : {args.sustained_duration:.1f} s  (tonal, noise)")
    print()

    print("LQR mode:")
    synthesize_lqr(npz, results, args)
    print()

    print("Strings mode:")
    synthesize_strings(npz, results, args)
    print()

    print(f"Done. All files in {AUDIO_DIR}/")
    print()
    print("Recommended listening order for each mode:")
    print("  1. *_target_tonal.wav    -- target chord filtered by target FRF")
    print("  2. *_passive_tonal.wav   -- same chord through the passive plate")
    print("  3. *_codesign_tonal.wav  -- same chord through the co-designed plate")
    print("  4. *_*_noise.wav         -- spectral portrait of each design")


if __name__ == "__main__":
    main()