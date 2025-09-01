import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generowanie danych wyników sportowych
sezony = np.arange(2015, 2025)
wyniki_druzyna_a = np.random.randint(50, 100, len(sezony))
wyniki_druzyna_b = np.random.randint(50, 100, len(sezony))

# Tworzenie DataFrame
df = pd.DataFrame({
    'Sezon': sezony,
    'Drużyna A': wyniki_druzyna_a,
    'Drużyna B': wyniki_druzyna_b
})

# Obliczanie średniej z wyników dla każdego sezonu
df['Średnia'] = df[['Drużyna A', 'Drużyna B']].mean(axis=1)

# Wizualizacja
plt.figure(figsize=(10, 5))
plt.plot(df['Sezon'], df['Drużyna A'], marker='o', label='Drużyna A', color='blue')
plt.plot(df['Sezon'], df['Drużyna B'], marker='o', label='Drużyna B', color='red')
plt.plot(df['Sezon'], df['Średnia'], marker='o', label='Średnia', color='green', linestyle='--')
plt.xlabel('Sezon')
plt.ylabel('Wynik')
plt.title('Wyniki drużyn w różnych sezonach')
plt.legend()
plt.show()
