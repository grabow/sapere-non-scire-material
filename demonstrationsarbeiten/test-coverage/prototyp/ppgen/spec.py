"""Eigenschaftsmodell (Spezifikation) fuer zu generierende Beispielfunktionen.

Harte Eigenschaften werden vom Generator exakt eingehalten, weiche Eigenschaften
steuern Wahrscheinlichkeiten und werden nur im Erwartungswert erreicht.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field


class SpecError(ValueError):
    """Die Spezifikation ist widerspruechlich und kann nicht erfuellt werden."""


@dataclass
class Spec:
    # --- harte Eigenschaften -------------------------------------------
    n_params: int = 2          # Anzahl ganzzahliger Parameter (1..5)
    cc: int = 6                # erweiterte zyklomatische Komplexitaet (Myers)
    max_depth: int = 3         # maximale Schachtelungstiefe der Kontrollstrukturen
    n_loops: int = 1           # Anzahl Schleifen (for/while)
    domain: int = 100          # Eingabebereich je Parameter: [-domain, domain]
    # --- weiche Eigenschaften (Wahrscheinlichkeiten) -------------------
    p_eq: float = 0.15         # Anteil Gleichheitsbedingungen (==, !=)
    p_mod: float = 0.10        # Anteil Modulo-Bedingungen
    p_rel2: float = 0.15       # Anteil Vergleiche zweier Variablen
    p_compound: float = 0.20   # Anteil Budget fuer zusammengesetzte Bedingungen
    p_elif: float = 0.20       # Anteil Budget fuer elif-Zweige
    p_else: float = 0.50       # Wahrscheinlichkeit eines else-Zweigs
    p_return: float = 0.15     # Wahrscheinlichkeit eines vorzeitigen return
    min_stmts: int = 1         # Zuweisungen je Block (min)
    max_stmts: int = 2         # Zuweisungen je Block (max)
    loop_kinds: tuple = ("for_const", "for_mod", "while")
    seed: int = 0

    @property
    def decisions(self) -> int:
        return self.cc - 1

    def validate(self) -> None:
        if not 1 <= self.n_params <= 5:
            raise SpecError("n_params muss zwischen 1 und 5 liegen")
        if self.cc < 1:
            raise SpecError("cc muss >= 1 sein")
        if self.n_loops < 0 or self.max_depth < 0:
            raise SpecError("n_loops und max_depth muessen >= 0 sein")
        if self.n_loops > self.decisions:
            raise SpecError("mehr Schleifen als Entscheidungen (cc - 1)")
        if self.max_depth > self.decisions:
            raise SpecError("Schachtelungstiefe groesser als Anzahl Entscheidungen")
        if self.decisions > 0 and self.max_depth == 0:
            raise SpecError("Entscheidungen erfordern max_depth >= 1")
        if self.n_loops > 4:
            raise SpecError("hoechstens 4 Schleifen unterstuetzt")
        if self.domain < 2:
            raise SpecError("domain muss >= 2 sein")
        if self.min_stmts < 1 or self.max_stmts < self.min_stmts:
            raise SpecError("ungueltige Blockgroesse")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["loop_kinds"] = list(self.loop_kinds)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Spec":
        d = dict(d)
        if "loop_kinds" in d:
            d["loop_kinds"] = tuple(d["loop_kinds"])
        return cls(**d)
