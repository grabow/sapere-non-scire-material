# ppgen – Property-guided Program Generator

Prototyp zur Bachelorarbeit „Generierung von Beispielcode mit definierten Eigenschaften
und Vorhersage der Test-Coverage“.

- `ppgen/spec.py` – Eigenschaftsmodell (harte und weiche Eigenschaften)
- `ppgen/generator.py` – konstruktiver Generator (Budget, Rückgrat, Rendering mit Pfadbedingungen)
- `ppgen/metrics.py` – unabhängige statische Analyse (Validierung, ML-Merkmale)
- `ppgen/runner.py` – Zweig-Instrumentierung und Coverage-Messung
- `ppgen/estimator.py` – analytische Coverage-Vorhersage (probabilistische abstrakte Interpretation)
- `ppgen/casestudy.py`, `ppgen/naive.py` – Lehrbuchfunktionen, naive Vergleichsbasis
- `experiments/` – Versuche 1–7, Modelle, Kennzahlen, Diagramme
- `data/` – Ergebnisse der Versuche (CSV/JSON)

Beispiel:

```python
from ppgen.spec import Spec
from ppgen.generator import generate
from ppgen.estimator import branch_probabilities, expected_coverage

spec = Spec(n_params=2, cc=7, max_depth=3, n_loops=1, domain=100, seed=10)
src = generate(spec, header=True).source
print(src)
print(expected_coverage(branch_probabilities(src, domain=100), [10, 100, 1000]))
```

Reproduktion aller Versuche: siehe Anhang A der Arbeit.
