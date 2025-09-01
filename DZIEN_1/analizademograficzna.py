import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generowanie danych demograficznych
lata = np.arange(2010, 2025)
populacja_miasto_x = 50000 + np.cumsum(np.random.normal(500, 100, len(lata)))
populacja_miasto_y = 70000 + np.cumsum(np.random.normal(300, 150, len(lata)))

# Tworzenie DataFrame
df = pd.DataFrame({
    'Rok': lata,
    'Miasto X': populacja_miasto_x,
    'Miasto Y': populacja_miasto_y
})

# Obliczanie zmiany procentowej rok do roku
df['Zmiana Miasto X (%)'] = df['Miasto X'].pct_change() * 100
df['Zmiana Miasto Y (%)'] = df['Miasto Y'].pct_change() * 100

# Wizualizacja
fig, ax1 = plt.subplots(figsize=(10, 5))

# Wykres populacji
ax1.set_xlabel('Rok')
ax1.set_ylabel('Populacja', color='black')
ax1.plot(df['Rok'], df['Miasto X'], label='Miasto X', color='blue')
ax1.plot(df['Rok'], df['Miasto Y'], label='Miasto Y', color='orange')
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend(loc='upper left')

# Dodanie drugiej osi y dla zmiany procentowej
ax2 = ax1.twinx()
ax2.set_ylabel('Zmiana procentowa (%)', color='grey')
ax2.plot(df['Rok'], df['Zmiana Miasto X (%)'], color='blue', linestyle='--', alpha=0.5)
ax2.plot(df['Rok'], df['Zmiana Miasto Y (%)'], color='orange', linestyle='--', alpha=0.5)
ax2.tick_params(axis='y', labelcolor='grey')

plt.title('Zmiany populacji w miastach i zmiana procentowa')
plt.show()
