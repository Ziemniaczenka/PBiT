import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_sensor_data(filepath, sensor_key):
    if not os.path.exists(filepath): return None, None, None
    with open(filepath, 'r') as f: data = json.load(f)['data']
    samples, timestamps = [], []
    for row in data:
        if sensor_key in row:
            item = row[sensor_key]
            timestamps.append(item['Timestamp'])
            if sensor_key == 'ecg': 
                samples.extend(item['Samples'])
            elif sensor_key == 'acc': 
                samples.extend([v['y'] for v in item['ArrayAcc']])
                
    if not samples: return None, None, None
    total_time_s = (timestamps[-1] - timestamps[0]) / 1000.0 
    fs_calculated = len(samples) / total_time_s if total_time_s > 0 else 1.0
    time_sec = np.linspace(0, total_time_s, len(samples))
    return np.array(samples), time_sec, fs_calculated

def load_hr_data(filepath):
    if not os.path.exists(filepath): return np.array([])
    with open(filepath, 'r') as f: data = json.load(f)['data']
    return np.array([row['heartRate']['average'] for row in data if 'heartRate' in row])

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ecg_files = glob.glob("**/*_ecg_stream.json", recursive=True)
sessions = sorted([f.replace("_ecg_stream.json", "") for f in ecg_files if OUTPUT_DIR not in f])
print("Dostępne sesje do ekstrakcji:")
for i, s in enumerate(sessions): print(f"[{i}] {s}")

wybor = input("\nWybierz numery sesji (np. 0,1,2 lub 0-2): ")
wybrane_indeksy = []
for part in wybor.split(','):
    if '-' in part:
        start, end = map(int, part.split('-'))
        wybrane_indeksy.extend(range(start, end + 1))
    else:
        try: wybrane_indeksy.append(int(part))
        except ValueError: pass

for idx in wybrane_indeksy:
    if not (0 <= idx < len(sessions)): continue
    session_base = sessions[idx]
    session_name = os.path.basename(session_base)
    print(f"\n--- Przetwarzanie: {session_name} ---")

    ecg_signal, time_ecg, fs_ecg = load_sensor_data(f"{session_base}_ecg_stream.json", 'ecg')
    acc_signal, time_acc, fs_acc = load_sensor_data(f"{session_base}_acc_stream.json", 'acc')
    hr_sensor = load_hr_data(f"{session_base}_heartRate_stream.json")
    fs_ecg_int = int(round(fs_ecg))

    # 1. AKCELEROMETR
    acc_y_no_gravity = acc_signal - np.mean(acc_signal)
    acc_filtered = nk.signal_filter(acc_y_no_gravity, sampling_rate=fs_acc, lowcut=0.5, highcut=10)
    
    raw_acc_peaks, _ = find_peaks(acc_filtered, height=1.5, prominence=0.8, distance=fs_acc/4)
    
    acc_peaks = []
    if len(raw_acc_peaks) > 3:
        step_times = time_acc[raw_acc_peaks]
        diffs = np.diff(step_times)
        valid_diffs = diffs[(diffs > 0.25) & (diffs < 1.5)] 
        median_diff = np.median(valid_diffs) if len(valid_diffs) > 0 else 0.5
        
        valid_peaks = []
        for i in range(len(raw_acc_peaks)):
            dist_prev = step_times[i] - step_times[i-1] if i > 0 else np.inf
            dist_next = step_times[i+1] - step_times[i] if i < len(raw_acc_peaks)-1 else np.inf
            if abs(dist_prev - median_diff) < 0.4 * median_diff or abs(dist_next - median_diff) < 0.4 * median_diff:
                valid_peaks.append(raw_acc_peaks[i])
        acc_peaks = np.array(valid_peaks)
    else:
        acc_peaks = raw_acc_peaks

    num_steps = len(acc_peaks)
    if num_steps > 1:
        t0_shift = time_acc[acc_peaks[0]]
        walk_duration = time_acc[acc_peaks[-1]] - time_acc[acc_peaks[0]]
        cadence = ((num_steps - 1) / walk_duration) * 60 if walk_duration > 0 else 0
    else:
        t0_shift, walk_duration, cadence = 0, 0, 0

    time_ecg -= t0_shift
    time_acc -= t0_shift
    total_time_s = time_ecg[-1] if len(time_ecg) > 0 else 0

    # 2. EKG
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs_ecg_int, method="neurokit")
    peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs_ecg_int, method="neurokit")
    r_peaks = info['ECG_R_Peaks']

    waves, waves_info = nk.ecg_delineate(ecg_cleaned, r_peaks, sampling_rate=fs_ecg_int, method="dwt")

    beats_time, beats_rr, beats_pq, beats_qt = [], [], [], []
    p_onsets = waves_info['ECG_P_Onsets']
    r_onsets = waves_info['ECG_R_Onsets']
    t_offsets = waves_info['ECG_T_Offsets']

    max_idx = min(len(r_peaks), len(p_onsets), len(r_onsets), len(t_offsets))

    for i in range(1, max_idx):
        r_idx = r_peaks[i]
        rr = (r_idx - r_peaks[i-1]) / fs_ecg * 1000
        p_on, r_on, t_off = p_onsets[i], r_onsets[i], t_offsets[i]
        
        pq = (r_on - p_on) / fs_ecg * 1000 if (not np.isnan(p_on) and not np.isnan(r_on) and r_on > p_on) else np.nan
        qt = (t_off - r_on) / fs_ecg * 1000 if (not np.isnan(r_on) and not np.isnan(t_off) and t_off > r_on) else np.nan

        max_qt = min(600, rr * 0.85) 
        if np.isnan(qt) or not (150 < qt < max_qt): qt = np.nan
        if np.isnan(pq) or not (50 < pq < 350): pq = np.nan

        beats_time.append(time_ecg[r_idx])
        beats_rr.append(rr)
        beats_pq.append(pq)
        beats_qt.append(qt)

    df_beats = pd.DataFrame({"Time_s": beats_time, "RR_ms": beats_rr, "PQ_ms": beats_pq, "QT_ms": beats_qt})
    df_beats['PQ_smooth'] = df_beats['PQ_ms'].interpolate(limit_direction='both').rolling(window=15, center=True, min_periods=1).mean()
    df_beats['QT_smooth'] = df_beats['QT_ms'].interpolate(limit_direction='both').rolling(window=15, center=True, min_periods=1).mean()

    # Wyliczanie Tętna z EKG na całą oś czasu
    hr_calculated = nk.signal_rate(r_peaks, sampling_rate=fs_ecg_int, desired_length=len(ecg_cleaned))
    time_hr_sensor = np.linspace(-t0_shift, total_time_s, len(hr_sensor)) if len(hr_sensor) > 0 else []

    # 3. DYNAMIKA HR (Zaawansowane parametry)
    # Wygładzamy tętno oknem 4-sekundowym, aby odrzucić błędy pomiaru przy szukaniu Max HR
    hr_smoothed = pd.Series(hr_calculated).rolling(window=fs_ecg_int*4, min_periods=1, center=True).mean().values
    
    max_hr_idx = np.nanargmax(hr_smoothed)
    max_hr = hr_smoothed[max_hr_idx]
    peak_hr_time = time_ecg[max_hr_idx]

    # HR na starcie (średnia z okna +/- 2 sekundy wokół t=0)
    mask_start = (time_ecg >= -2.0) & (time_ecg <= 2.0)
    hr_start = np.nanmean(hr_smoothed[mask_start]) if np.sum(mask_start) > 0 else np.nan

    # HR na końcu zapisu (średnia z ostatnich 5 sekund zapisu - restytucja)
    mask_end = (time_ecg >= time_ecg[-1] - 5.0)
    hr_end = np.nanmean(hr_smoothed[mask_end]) if np.sum(mask_end) > 0 else np.nan

    # Kalkulacja różnic
    hr_delta = max_hr - hr_start
    hr_recovery = max_hr - hr_end

    # Bezpieczne średnie
    hr_mean = 60000 / np.nanmean(beats_rr) if np.sum(~np.isnan(beats_rr)) > 0 else np.nan
    mean_pq = np.nanmean(beats_pq) if np.sum(~np.isnan(beats_pq)) > 0 else np.nan
    mean_qt = np.nanmean(beats_qt) if np.sum(~np.isnan(beats_qt)) > 0 else np.nan

    # 4. ZAPIS DO PLIKU
    summary_data = {
        "session": session_name, 
        "total_steps": num_steps, 
        "cadence_bpm": cadence, 
        "walk_duration_s": walk_duration,
        "hr_start_bpm": hr_start,
        "hr_max_bpm": max_hr,
        "time_to_peak_s": peak_hr_time,
        "hr_delta_bpm": hr_delta,
        "hr_recovery_bpm": hr_recovery,
        "hr_mean_bpm": hr_mean,
        "rr_mean_ms": np.nanmean(beats_rr) if np.sum(~np.isnan(beats_rr)) > 0 else np.nan, 
        "rr_std_ms": np.nanstd(beats_rr) if np.sum(~np.isnan(beats_rr)) > 0 else np.nan,
        "pq_mean_ms": mean_pq, 
        "qt_mean_ms": mean_qt
    }
    with open(os.path.join(OUTPUT_DIR, f"{session_name}_summary.json"), 'w') as f: json.dump(summary_data, f, indent=4)
    df_beats.to_csv(os.path.join(OUTPUT_DIR, f"{session_name}_beats.csv"), index=False)

    # ==========================================
    # 5. WIZUALIZACJA 
    # ==========================================
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.subplots_adjust(top=0.90, hspace=0.4)
    
    stats_text = (f"Plik: {session_name}   |   Kadencja: {cadence:.1f} kr/min (Czas: {walk_duration:.1f}s)   |   Max HR: {max_hr:.1f} bpm (Peak w {peak_hr_time:.1f}s)\n"
                  f"Przyrost HR: +{hr_delta:.1f} bpm   |   Spadek HR na koniec: -{hr_recovery:.1f} bpm   |   Średnie QT: {mean_qt:.1f} ms")
    fig.suptitle(stats_text, fontsize=12, fontweight='bold', color='darkblue')

    # WYKRES 1: EKG
    plot_start = max(time_ecg[0], peak_hr_time - 2.5)
    plot_end = plot_start + 5.0
    
    mask_ecg = (time_ecg >= plot_start) & (time_ecg <= plot_end)
    axes[0].plot(time_ecg[mask_ecg], ecg_cleaned[mask_ecg], color='black', linewidth=1.2)
    
    r_in_range_idx = r_peaks[(time_ecg[r_peaks] >= plot_start) & (time_ecg[r_peaks] <= plot_end)]
    axes[0].scatter(time_ecg[r_in_range_idx], ecg_cleaned[r_in_range_idx], color='red', s=60, zorder=6, label="Załamek R")

    p_in_range = [int(p) for p in p_onsets if not np.isnan(p) and plot_start <= time_ecg[int(p)] <= plot_end]
    t_in_range = [int(t) for t in t_offsets if not np.isnan(t) and plot_start <= time_ecg[int(t)] <= plot_end]
    axes[0].scatter(time_ecg[p_in_range], ecg_cleaned[p_in_range], color='green', s=50, zorder=5, marker='>', label="Start PQ")
    axes[0].scatter(time_ecg[t_in_range], ecg_cleaned[t_in_range], color='purple', s=50, zorder=5, marker='<', label="Koniec QT")
    
    axes[0].set_title(f"Morfologia EKG w szczytowym wysiłku (Centrum: {peak_hr_time:.1f} s)"); axes[0].legend(loc="upper right"); axes[0].grid(True)

    # WYKRES 2: Akcelerometr
    axes[1].plot(time_acc, acc_filtered, color='blue', alpha=0.7)
    axes[1].scatter(time_acc[raw_acc_peaks], acc_filtered[raw_acc_peaks], color='gray', marker='x', zorder=4, label="Odrzucone szumy")
    if len(acc_peaks) > 0:
        axes[1].scatter(time_acc[acc_peaks], acc_filtered[acc_peaks], color='orange', zorder=5, label=f"Ważne kroki ({num_steps})")
    axes[1].set_title("Detekcja kroków (Tylko oś Y pionowa)"); axes[1].legend(); axes[1].grid(True)

    # WYKRES 3: Tętno
    axes[2].plot(time_ecg, hr_smoothed, color='green', label="HR z EKG (Wygładzone)")
    if len(time_hr_sensor) > 0: axes[2].plot(time_hr_sensor, hr_sensor, color='purple', linestyle='--', label="HR z Sensora")
    axes[2].axvline(x=0, color='red', linestyle='--', label='Start wysiłku')
    axes[2].axvline(x=peak_hr_time, color='orange', linestyle=':', linewidth=2, label='Max HR')
    axes[2].set_title("Profil tętna"); axes[2].set_ylabel("Tętno [bpm]"); axes[2].legend(); axes[2].grid(True)

    # WYKRES 4: Dynamika PQ i QT
    axes[3].plot(df_beats['Time_s'], df_beats['QT_smooth'], color='purple', linewidth=2, label="Odcinek QT (wygładzony)")
    axes[3].plot(df_beats['Time_s'], df_beats['PQ_smooth'], color='green', linewidth=2, label="Odcinek PQ (wygładzony)")
    axes[3].axvline(x=0, color='red', linestyle='--', label='Start wysiłku')
    axes[3].set_title("Zmiany fizjologiczne odcinków PQ i QT w czasie")
    axes[3].set_xlabel("Czas zsynchronizowany [s] (0 = pierwszy krok)"); axes[3].set_ylabel("Czas trwania [ms]")
    axes[3].legend(); axes[3].grid(True)

    output_fig_path = os.path.join(OUTPUT_DIR, f"{session_name}_ekstrakcja_wykres.png")
    plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
    print(f"Zapisano dane i wykres w folderze '{OUTPUT_DIR}'")

    plt.show()