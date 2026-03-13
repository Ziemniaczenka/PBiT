import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"
summary_files = sorted(glob.glob(f"{OUTPUT_DIR}/*_summary.json"))

if not summary_files:
    print(f"Brak plików w folderze '{OUTPUT_DIR}'. Najpierw użyj programu ekstrakcja.py")
    exit()

print("Dostępne sesje do porównania:")
for i, file in enumerate(summary_files):
    name = os.path.basename(file).replace('_summary.json', '')
    print(f"[{i}] {name}")

wybor = input("\nWpisz numery sesji do porównania (np. 0,1,2 lub 0-2): ")
wybrane_indeksy = []
for part in wybor.split(','):
    if '-' in part:
        start, end = map(int, part.split('-'))
        wybrane_indeksy.extend(range(start, end + 1))
    else:
        try: wybrane_indeksy.append(int(part))
        except ValueError: pass

wybrane_summary = [summary_files[i] for i in wybrane_indeksy if 0 <= i < len(summary_files)]

# 1. TABELA W KONSOLI
print("\n" + "="*100)
print(f"{'SESJA':<20} | {'Kroki':<6} | {'Kadencja':<8} | {'HR Średnie':<10} | {'R-R Śr (ms)':<11} | {'PQ (ms)':<8} | {'QT (ms)':<8}")
print("-" * 100)
for file in wybrane_summary:
    with open(file, 'r') as f: d = json.load(f)
    print(f"{d['session']:<20} | {d.get('total_steps', 0):<6} | {d['cadence_bpm']:<8.1f} | {d['hr_mean_bpm']:<10.1f} | {d['rr_mean_ms']:<11.1f} | {d['pq_mean_ms']:<8.1f} | {d['qt_mean_ms']:<8.1f}")
print("="*100)

# 2. WYKRESY PORÓWNAWCZE
fig, axes = plt.subplots(3, 1, figsize=(12, 14))
fig.tight_layout(pad=6.0)

for file in wybrane_summary:
    session_name = os.path.basename(file).replace("_summary.json", "")
    beats_file = os.path.join(OUTPUT_DIR, f"{session_name}_beats.csv")
    
    if os.path.exists(beats_file):
        df = pd.read_csv(beats_file)
        
        df['HR'] = 60000 / df['RR_ms']
        
        # Odrzucenie błędów medycznych i interpolacja luk dla gładkiego wykresu
        df.loc[~df['QT_ms'].between(200, 600), 'QT_ms'] = np.nan
        df.loc[~df['PQ_ms'].between(100, 300), 'PQ_ms'] = np.nan
        
        df['QT_ms'] = df['QT_ms'].interpolate()
        df['PQ_ms'] = df['PQ_ms'].interpolate()

        # Mocne wygładzanie, aby wyciągnąć czytelny trend z szumu
        df['HR_smooth'] = df['HR'].rolling(window=7, center=True, min_periods=1).mean()
        df['QT_smooth'] = df['QT_ms'].rolling(window=10, center=True, min_periods=1).mean()
        df['PQ_smooth'] = df['PQ_ms'].rolling(window=10, center=True, min_periods=1).mean()

        axes[0].plot(df['Time_s'], df['HR_smooth'], label=session_name, linewidth=2)
        axes[1].plot(df['Time_s'], df['QT_smooth'], label=session_name, linewidth=2)
        axes[2].plot(df['Time_s'], df['PQ_smooth'], label=session_name, linewidth=2)

axes[0].axvline(x=0, color='black', linestyle='--', label='Start chodu')
axes[0].set_title("Porównanie Tętna (HR)")
axes[0].set_ylabel("Tętno [bpm]")
axes[0].legend(); axes[0].grid(True)

axes[1].axvline(x=0, color='black', linestyle='--', label='Start chodu')
axes[1].set_title("Porównanie odcinka QT (Wydolność skurczu komór)")
axes[1].set_ylabel("Długość QT [ms]")
axes[1].legend(); axes[1].grid(True)

axes[2].axvline(x=0, color='black', linestyle='--', label='Start chodu')
axes[2].set_title("Porównanie odcinka PQ (Przewodnictwo przedsionkowo-komorowe)")
axes[2].set_xlabel("Czas [s] (0 = start chodu)")
axes[2].set_ylabel("Długość PQ [ms]")
axes[2].legend(); axes[2].grid(True)

output_fig_path = os.path.join(OUTPUT_DIR, "porownanie_wykres_PQ_QT.png")
plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
print(f"\nZapisano wykres porównawczy w: {output_fig_path}")

plt.show()