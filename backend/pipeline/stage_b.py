import numpy as np
import librosa
from backend.pipeline.models import NoteEvent, ChordEvent, MetaData, AlternativePitch
from typing import List, Tuple
import os
import scipy.signal

# Try imports for heavy ML models
# Basic Pitch removed as per request (Deep Analysis Default)
BASIC_PITCH_AVAILABLE = False

try:
    import crepe
    CREPE_AVAILABLE = True
except ImportError:
    CREPE_AVAILABLE = False
    print("CREPE not available.")

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available.")


def extract_features(y: np.ndarray, sr: int, meta: MetaData) -> Tuple[List[NoteEvent], List[ChordEvent]]:
    """
    Stage B: Analysis & Feature Extraction

    1. Fallback Chain (Basic-Pitch -> CREPE -> HPSS)
    2. Attack / Transient Modeling (Spectral Flux)
    3. Octave Error Correction
    4. Vibrato & Modulation Handling
    5. Chord Detection
    6. Voice Separation (K-Means)
    7. Dissonant / Overtone Suppression
    8. Sliding Window Key Detection (Simplified to global key for now, updated in Meta)
    """

    events: List[NoteEvent] = []
    chords: List[ChordEvent] = []

    # Use Adaptive Window from Stage A
    n_fft = meta.window_size
    hop_length = n_fft // 4

    # B1. Pitch Detection
    # Primary: CREPE
    # Secondary: HPSS/Librosa (for alternatives)

    # 1. Run CREPE (Primary)
    if CREPE_AVAILABLE:
        print("Running CREPE (Primary)...")
        try:
            # B4. Vibrato & Modulation Handling (Pre-processing for Crepe?)
            # Actually Crepe handles vibrato well, but we need to smooth the output pitch curve.
            # "Implement vibrato smoothing: median filter (1/2-semitone window)"

            time, frequency, confidence, _ = crepe.predict(y, sr, viterbi=True, step_size=10)

            # Apply median filter to frequency to smooth vibrato
            # Window size? "1/2-semitone window".
            # If vibrato is ~5-6Hz, and step_size=10ms (100Hz sample rate of pitch),
            # 5Hz is 20 frames period. A median filter of 5-10 frames might smooth it.
            # SciPy median filter
            frequency = scipy.signal.medfilt(frequency, kernel_size=5)

            current_event = None

            for t, f, c in zip(time, frequency, confidence):
                if c > 0.6 and f > 20:
                    midi = librosa.hz_to_midi(f)
                    midi_round = int(round(midi))

                    if current_event is None:
                        current_event = {
                            "start": t,
                            "midi": midi_round,
                            "conf_sum": c,
                            "count": 1
                        }
                    else:
                        if abs(current_event["midi"] - midi_round) <= 1:
                            current_event["conf_sum"] += c
                            current_event["count"] += 1
                        else:
                            # Close event
                            duration = t - current_event["start"]
                            if duration > 0.05:
                                events.append(NoteEvent(
                                    id=f"crepe_{len(events)}",
                                    midi_note=current_event["midi"],
                                    start_sec=current_event["start"],
                                    end_sec=t,
                                    start_beat=0, duration_beat=0,
                                    confidence=current_event["conf_sum"]/current_event["count"],
                                    type="normal",
                                    voice=1
                                ))
                            current_event = {
                                "start": t,
                                "midi": midi_round,
                                "conf_sum": c,
                                "count": 1
                            }
                else:
                    if current_event:
                        duration = t - current_event["start"]
                        if duration > 0.05:
                            events.append(NoteEvent(
                                id=f"crepe_{len(events)}",
                                midi_note=current_event["midi"],
                                start_sec=current_event["start"],
                                end_sec=t,
                                start_beat=0, duration_beat=0,
                                confidence=current_event["conf_sum"]/current_event["count"],
                                type="normal",
                                voice=1
                            ))
                        current_event = None

            meta.processing_mode = "crepe_deep"

        except Exception as e:
            print(f"CREPE failed: {e}")

    # 2. Run HPSS + Librosa (Secondary - for Alternatives)
    # Always run this for "Deep Analysis" to provide alternatives
    print("Running Librosa (HPSS) for alternatives...")
    hpss_events = []
    try:
        y_harmonic, _ = librosa.effects.hpss(y)
        f0, voiced_flag, voiced_probs = librosa.pyin(y_harmonic, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr, frame_length=n_fft)
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

        # Simple segmentation for pyin
        current_event = None
        for i, (f, v) in enumerate(zip(f0, voiced_flag)):
            t = times[i]
            if v and not np.isnan(f):
                midi = librosa.hz_to_midi(f)
                midi_round = int(round(midi))
                if current_event is None:
                    current_event = {"start": t, "midi": midi_round, "count": 1}
                else:
                    if abs(current_event["midi"] - midi_round) <= 1:
                        current_event["count"] += 1
                    else:
                        duration = t - current_event["start"]
                        if duration > 0.05:
                            hpss_events.append({
                                "midi": current_event["midi"],
                                "start": current_event["start"],
                                "end": t,
                                "conf": 0.7 # Librosa doesn't give direct confidence like Crepe per se, use prob if available or constant
                            })
                        current_event = {"start": t, "midi": midi_round, "count": 1}
            else:
                 if current_event:
                        duration = t - current_event["start"]
                        if duration > 0.05:
                             hpss_events.append({
                                "midi": current_event["midi"],
                                "start": current_event["start"],
                                "end": t,
                                "conf": 0.7
                            })
                        current_event = None
    except Exception as e:
        print(f"HPSS failed: {e}")

    # 3. Match HPSS events to CREPE events (Alternatives)
    if events and hpss_events:
        for m_event in events:
            # Find overlapping HPSS events
            for h_event in hpss_events:
                # Check overlap
                start = max(m_event.start_sec, h_event["start"])
                end = min(m_event.end_sec, h_event["end"])
                overlap = end - start

                if overlap > 0.05: # Minimal overlap to consider
                     # Add as alternative if pitch is different
                     if h_event["midi"] != m_event.midi_note:
                         # Check if already added
                         already_exists = any(a.midi == h_event["midi"] and a.source == "hpss" for a in m_event.alternatives)
                         if not already_exists:
                             m_event.alternatives.append(AlternativePitch(
                                 midi=h_event["midi"],
                                 confidence=h_event["conf"],
                                 source="hpss"
                             ))

    # If CREPE failed completely (no events), maybe use HPSS as fallback?
    # User said "Option B. Ignore it" regarding missing notes.
    # But if events list is empty, we have no result.
    # Assuming "Ignore it" means "don't add HPSS notes that don't overlap with CREPE notes".
    # So if CREPE finds nothing, we return nothing.
    # This aligns with "eliminate basic pitch" and "CREPE primary".


    # B2. Attack / Transient Modeling
    # "Compute Spectral centroid, Rise time slope, Flux curve"
    # "Low attack energy + pitch continuity -> slur"
    # "High attack energy -> re-articulation"

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=n_fft)
    onset_times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)

    # We can check the onset strength at the start of each event.
    for e in events:
        # Find index in onset_env
        idx = np.argmin(np.abs(onset_times - e.start_sec))
        attack_energy = onset_env[idx]

        # If attack energy is low, mark as slur?
        # This usually requires context (is it connected to previous note?)
        # For now, we store it or use it for "type"
        if attack_energy < 1.0: # Threshold dependent on normalization
             # Might be slur or grace
             pass

    # B3. Octave Error Correction
    # "Compare energy at f, 2f, 4f. Correct doubling or halving errors."
    S = np.abs(librosa.stft(y, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    for e in events:
        # Get frame index
        t_center = (e.start_sec + e.end_sec) / 2
        frame_idx = int(t_center * sr / hop_length)
        if frame_idx >= S.shape[1]:
            # Try to clamp or just skip if out of bounds (end of file)
            frame_idx = S.shape[1] - 1
            if frame_idx < 0: continue

        f0 = librosa.midi_to_hz(e.midi_note)

        # Check energy at f0, f0/2, 2*f0
        def get_energy(freq):
            bin_idx = np.argmin(np.abs(freqs - freq))
            return S[bin_idx, frame_idx]

        e_f0 = get_energy(f0)
        e_half = get_energy(f0 / 2)
        e_double = get_energy(f0 * 2)

        # If half frequency has significantly more energy, we might have an octave error (too high)
        if e_half > e_f0 * 1.5:
             # Correct down
             e.midi_note -= 12
        elif e_double > e_f0 * 2.0: # Rarer to estimate too low if fundamental is weak
             pass


    # B5. Chord Detection
    # Use beat-aligned chroma clustering + template matching
    # Calculate chroma over the whole track
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    # Define simple templates for Major and Minor triads
    # Shape: (12, 12) or (12, 24)
    # Rows = Chroma bins (C, C#, ...), Cols = Chord roots
    templates = {}
    roots = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Major template: [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0] (Root, Maj3, Perf5) -> indices 0, 4, 7
    # Minor template: [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0] (Root, Min3, Perf5) -> indices 0, 3, 7

    for i, root in enumerate(roots):
        # Major
        vec_maj = np.zeros(12)
        vec_maj[i] = 1
        vec_maj[(i+4)%12] = 1
        vec_maj[(i+7)%12] = 1
        templates[f"{root}"] = vec_maj

        # Minor
        vec_min = np.zeros(12)
        vec_min[i] = 1
        vec_min[(i+3)%12] = 1
        vec_min[(i+7)%12] = 1
        templates[f"{root}m"] = vec_min

    # For simplicity, we sample chords at regular intervals (e.g. every second or beat)
    # Or aggregate chroma over segments.
    # Let's do a simple framing: 1 chord per second.
    # chroma shape is (12, time)
    frames_per_sec = chroma.shape[1] / (len(y)/sr)
    step = int(frames_per_sec) # 1 sec

    if step > 0:
        for t_idx in range(0, chroma.shape[1], step):
            segment = chroma[:, t_idx:t_idx+step]
            if segment.shape[1] == 0: continue

            avg_chroma = np.mean(segment, axis=1)
            # Normalize
            if np.max(avg_chroma) > 0:
                avg_chroma /= np.max(avg_chroma)

            # Match against templates
            best_score = -1
            best_chord = "N.C."

            for name, template in templates.items():
                score = np.dot(avg_chroma, template)
                if score > best_score:
                    best_score = score
                    best_chord = name

            # Record chord event
            beat_est = (t_idx / frames_per_sec) * (120/60) # Approx using 120bpm default
            # Only add if distinct from previous?
            if not chords or chords[-1].symbol != best_chord:
                chords.append(ChordEvent(beat=beat_est, symbol=best_chord))

    # B6. Voice Separation (K-Means)
    if SKLEARN_AVAILABLE and len(events) > 0:
        pitches = np.array([e.midi_note for e in events]).reshape(-1, 1)
        if len(pitches) >= 2:
            try:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(pitches)
                labels = kmeans.labels_
                centers = kmeans.cluster_centers_

                if centers[0][0] > centers[1][0]:
                    map_voice = {0: 1, 1: 2}
                else:
                    map_voice = {0: 2, 1: 1}

                for i, event in enumerate(events):
                    event.voice = map_voice[labels[i]]
            except Exception as e:
                print(f"KMeans failed: {e}")

    # B7. Dissonant / Overtone Suppression
    # "If a detected note is: Harmonic multiple of a stronger bass note AND aligns in time AND has low confidence"
    events.sort(key=lambda x: (x.start_sec, x.midi_note))

    events_to_remove = set()
    for i in range(len(events)):
        for j in range(len(events)):
            if i == j: continue

            note_a = events[i]
            note_b = events[j]

            # Check overlap
            start = max(note_a.start_sec, note_b.start_sec)
            end = min(note_a.end_sec, note_b.end_sec)
            overlap = end - start

            if overlap > 0.1: # Significant overlap
                # Check harmonic relation
                # If note_b is harmonic of note_a
                # e.g., note_b.midi = note_a.midi + 12, +19, +24
                diff = note_b.midi_note - note_a.midi_note
                if diff in [12, 19, 24, 28]: # Octave, Fifth, 2 Octaves, etc.
                    # If note_b is weaker or low confidence
                    if note_b.confidence < 0.6 or (note_b.confidence < note_a.confidence * 0.5):
                        events_to_remove.add(note_b.id)

    final_events = [e for e in events if e.id not in events_to_remove]

    # B8. Sliding Window Key Detection (Simplified to global key for now, updated in Meta)
    # Use music21 to analyze the key of the detected events.
    if len(final_events) > 0:
        try:
            from music21 import stream, note, analysis
            s = stream.Stream()
            for e in final_events:
                n = note.Note(e.midi_note)
                n.quarterLength = 1.0 # arbitrary
                s.append(n)

            # Analyze key
            k = s.analyze('key')
            meta.detected_key = f"{k.tonic.name}{'m' if k.mode == 'minor' else ''}"
        except Exception as e:
            print(f"Key detection failed: {e}")
            meta.detected_key = "C"

    return final_events, chords
