import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import find_peaks

# ==========================================
# 1. FUNKCJE WYSZUKUJĄCE I ŁADUJĄCE DANE
# ==========================================

def find_sessions():
    """Przeszukuje katalogi i znajduje unikalne sesje pomiarowe."""
    # Szuka wszystkich plików EKG we wszystkich podfolderach
    ecg_files = glob.glob("**/*_ecg_stream.json", recursive=True)
    
    if not ecg_files:
        print("Nie znaleziono żadnych plików '*_ecg_stream.json' w katalogu i podkatalogach.")
        return []

    sessions = []
    for file in ecg_files:
        # Wyciąga bazową nazwę sesji (odcina końcówkę)
        base_name = file.replace("_ecg_stream.json", "")
        sessions.append(base_name)
    
    return sorted(sessions)

def select_session(sessions):
    """Wyświetla interaktywne menu w konsoli do wyboru sesji."""
    print("\n" + "="*40)
    print(" ZNALEZIONE SESJE POMIAROWE (Wybierz plik)")
    print("="*40)
    for i, session in enumerate(sessions):
        print(f"[{i}] {session}")
    
    while True:
        try:
            choice = int(input(f"\nWybierz numer sesji (0 - {len(sessions)-1}): "))
            if 0 <= choice < len(sessions):
                return sessions[choice]
            else:
                print("Nieprawidłowy numer. Spróbuj ponownie.")
        except ValueError:
            print("Wpisz poprawną liczbę całkowitą.")

def load_sensor_data(filepath, sensor_key):
    """Wczytuje surowe próbki i oblicza dynamiczne próbkowanie na podstawie Timestamp."""
    if not os.path.exists(filepath):
        return None, None, None

    with open(filepath, 'r') as f:
        data = json.load(f)['data']
    
    samples = []
    timestamps = []
    
    for row in data:
        if sensor_key in row:
            item = row[sensor_key]
            ts = item['Timestamp']
            
            if sensor_key == 'ecg':
                vals = item['Samples']
            elif sensor_key == 'acc':
                # Wyliczamy moduł przyspieszenia (wypadkową) od razu
                vals = [np.sqrt(v['x']**2 + v['y']**2 + v['z']**2) for v in item['ArrayAcc']]
                
            samples.extend(vals)
            # Timestamp z MoveSense podany jest dla paczki danych. 
            timestamps.append(ts) 

    if not samples:
        return None, None, None

    # Obliczanie dynamicznego próbkowania (FS)
    # Różnica czasu między pierwszym a ostatnim pomiarem w sekundach
    total_time_s = (timestamps[-1] - timestamps[0]) / 1000.0 
    fs_calculated = len(samples) / total_time_s if total_time_s > 0 else 1.0
    
    # Tworzenie wektora czasu od 0 do total_time_s
    time_sec = np.linspace(0, total_time_s, len(samples))
    
    return np.array(samples), time_sec, fs_calculated

def load_hr_data(filepath):
    """Wczytuje gotowe uśrednione tętno wyliczone przez czujnik."""
    if not os.path.exists(filepath):
        return np.array([])

    with open(filepath, 'r') as f:
        data = json.load(f)['data']
    hr_vals = [row['heartRate']['average'] for row in data if 'heartRate' in row]
    return np.array(hr_vals)

# ==========================================
# 2. GŁÓWNA LOGIKA PROGRAMU
# ==========================================

# A. Wybór sesji
available_sessions = find_sessions()
if not available_sessions:
    exit()

session_base = select_session(available_sessions)
print(f"\nŁadowanie plików dla sesji: {session_base}...")

FILE_ECG = f"{session_base}_ecg_stream.json"
FILE_ACC = f"{session_base}_acc_stream.json"
FILE_HR  = f"{session_base}_heartRate_stream.json"

# B. Wczytywanie danych
ecg_signal, time_ecg, fs_ecg = load_sensor_data(FILE_ECG, 'ecg')
acc_signal, time_acc, fs_acc = load_sensor_data(FILE_ACC, 'acc')
hr_sensor = load_hr_data(FILE_HR)

print(f"\n--- Parametry sygnałów ---")
print(f"Obliczone próbkowanie EKG: {fs_ecg:.1f} Hz")
print(f"Obliczone próbkowanie ACC: {fs_acc:.1f} Hz")

# C. Analiza EKG (NeuroKit2)
print("Przetwarzanie EKG (to może chwilę potrwać)...")
# Używamy ecg_process, które robi wszystko: filtrację, szukanie pików i analizę jakości
# Zaokrąglamy FS do inta, bo NeuroKit w niektórych miejscach tego wymaga
ecg_signals_df, info = nk.ecg_process(ecg_signal, sampling_rate=int(round(fs_ecg)))

# Wyciągamy wyliczone chwilowe tętno dla całego przebiegu
hr_calculated = ecg_signals_df["ECG_Rate"].values
ecg_cleaned = ecg_signals_df["ECG_Clean"].values

# Analiza Zmienności Rytmu Serca (HRV) - dla dodatkowych parametrów
hrv_indices = nk.hrv_time(info, sampling_rate=int(round(fs_ecg)))
rmssd = hrv_indices['HRV_RMSSD'].values[0] if 'HRV_RMSSD' in hrv_indices else np.nan
sdnn = hrv_indices['HRV_SDNN'].values[0] if 'HRV_SDNN' in hrv_indices else np.nan

# D. Analiza Akcelerometru (Kroki)
print("Przetwarzanie Akcelerometru...")
acc_no_gravity = acc_signal - np.mean(acc_signal)
# Szukamy kroków, minimalny odstęp to (fs_acc/3) czyli ok 0.33s (do 3 kroków na sekundę)
acc_peaks, _ = find_peaks(acc_no_gravity, height=2.0, distance=fs_acc/3)

duration_minutes = time_acc[-1] / 60.0
num_steps = len(acc_peaks)
cadence = num_steps / duration_minutes if duration_minutes > 0 else 0

# Obliczanie czasów pomiędzy krokami w sekundach, aby wyciągnąć statystyki miarowości
steps_time_diffs = np.diff(time_acc[acc_peaks]) 
steps_per_min_instant = 60.0 / steps_time_diffs if len(steps_time_diffs) > 0 else []

# ==========================================
# 3. ROZBUDOWANE WYNIKI W KONSOLI
# ==========================================
print("\n" + "="*50)
print(f" PODSUMOWANIE ANALIZY: {os.path.basename(session_base)}")
print("="*50)

print("\n[ PARAMETRY AKCELEROMETRU I KROKÓW ]")
print(f"Łączny czas pomiaru:      {duration_minutes*60:.1f} s")
print(f"Całkowita liczba kroków:  {num_steps}")
print(f"Średnia kadencja:         {cadence:.1f} kroków/min")
if len(steps_per_min_instant) > 0:
    print(f"  -> Min. chwilowa kadencja: {np.min(steps_per_min_instant):.1f} kr/min")
    print(f"  -> Max. chwilowa kadencja: {np.max(steps_per_min_instant):.1f} kr/min")
    print(f"  -> Odchylenie standardowe: {np.std(steps_per_min_instant):.1f} kr/min (miara równości rytmu)")

print("\n[ PARAMETRY TĘTNA Z SENSORA (Plik HR) ]")
if len(hr_sensor) > 0:
    print(f"Średnia:   {np.mean(hr_sensor):.1f} bpm")
    print(f"Mediana:   {np.median(hr_sensor):.1f} bpm")
    print(f"Min / Max: {np.min(hr_sensor):.1f} / {np.max(hr_sensor):.1f} bpm")
    print(f"Odch. standardowe (SD): {np.std(hr_sensor):.1f} bpm")
else:
    print("Brak danych z pliku heartRate.")

print("\n[ PARAMETRY WYLICZONE Z SUROWEGO EKG (NeuroKit2) ]")
# Usuwamy zera/NaN z początków sygnału do statystyk
valid_hr = hr_calculated[~np.isnan(hr_calculated) & (hr_calculated > 0)]
if len(valid_hr) > 0:
    print(f"Średnia:   {np.mean(valid_hr):.1f} bpm")
    print(f"Mediana:   {np.median(valid_hr):.1f} bpm")
    print(f"Min / Max: {np.min(valid_hr):.1f} / {np.max(valid_hr):.1f} bpm")
    print(f"Odch. standardowe (SD): {np.std(valid_hr):.1f} bpm")
    
    print("\n[ DODATKOWE PARAMETRY ZMIENNOŚCI (HRV) ]")
    print(f"RMSSD (Zmienność krótkoterminowa): {rmssd:.1f} ms")
    print(f"SDNN (Zmienność ogólna):           {sdnn:.1f} ms")
else:
    print("Nie udało się wyliczyć tętna z sygnału EKG.")

print("="*50 + "\n")

# ==========================================
# 4. WIZUALIZACJA (WYKRESY)
# ==========================================
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.tight_layout(pad=6.0)

# 1. EKG - wycinek
limit_ecg = int(5 * fs_ecg) # Pokazujemy 5 sekund dla lepszej czytelności załamków
axes[0].plot(time_ecg[:limit_ecg], ecg_signal[:limit_ecg], label="Surowe", alpha=0.4, color='gray')
axes[0].plot(time_ecg[:limit_ecg], ecg_cleaned[:limit_ecg], label="Oczyszczone", color='red')
r_peaks_indices = info['ECG_R_Peaks']
peaks_in_limit = [p for p in r_peaks_indices if p < limit_ecg]
axes[0].scatter(time_ecg[peaks_in_limit], ecg_cleaned[peaks_in_limit], color='black', zorder=5, label="R-peaks")
axes[0].set_title(f"Sygnał EKG (wycinek 5s) - Próbkowanie {fs_ecg:.1f} Hz")
axes[0].set_xlabel("Czas [s]"); axes[0].set_ylabel("Amplituda")
axes[0].legend(); axes[0].grid(True)

# 2. Akcelerometr - pełny przebieg by widzieć schody
axes[1].plot(time_acc, acc_no_gravity, color='blue', alpha=0.7, label="Przyspieszenie wypadkowe")
axes[1].scatter(time_acc[acc_peaks], acc_no_gravity[acc_peaks], color='orange', zorder=5, label=f"Wykryte kroki ({num_steps})")
axes[1].set_title(f"Sygnał ACC (całość) - Średnia kadencja: {cadence:.1f} kr/min")
axes[1].set_xlabel("Czas [s]"); axes[1].set_ylabel("Przyspieszenie [m/s^2]")
axes[1].legend(loc="upper right"); axes[1].grid(True)

# 3. Porównanie Tętna (Wyliczone z EKG vs Sensor)
# Aby narysować HR sensora potrzebujemy osi czasu. Sensor podaje HR co sekundę (zakładamy równy rozstaw)
total_time_s = time_ecg[-1] if len(time_ecg) > 0 else 0
time_hr_sensor = np.linspace(0, total_time_s, len(hr_sensor)) if len(hr_sensor) > 0 else []

axes[2].plot(time_ecg, hr_calculated, color='green', label="Tętno z surowego EKG (NeuroKit)")
if len(time_hr_sensor) > 0:
    axes[2].plot(time_hr_sensor, hr_sensor, color='purple', linestyle='--', linewidth=2, label="Tętno uśrednione z Sensora")
axes[2].set_title("Profil tętna w trakcie wysiłku")
axes[2].set_xlabel("Czas [s]"); axes[2].set_ylabel("Tętno [bpm]")
axes[2].legend(); axes[2].grid(True)

plt.show()