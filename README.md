# Material zum Beitrag „Sapere, non scire“

Ergänzendes Material zum Beitrag *Sapere, non scire. Bildung, Delegation
und Urteil unter der Annahme überlegener künstlicher Intelligenz*.

## Inhalt

### `demonstrationsarbeiten/`

> **Demonstrationsarbeiten – keine Prüfungsleistungen.**
> Beide Arbeiten wurden im Rahmen des Beitrags mit einem KI-Agenten
> erstellt. Sie wurden nie als Prüfungsleistung eingereicht und nicht
> bewertet. Wer sie oder Teile davon als eigene Prüfungsleistung
> einreicht, täuscht.

Jede Arbeit liegt als gekennzeichnetes PDF vor: mit vorangestelltem
Hinweisblatt und dem Vermerk „Demonstrationsarbeit – mit KI erstellt –
keine Prüfungsleistung“ auf jeder Seite. Daneben liegen jeweils die
vollständige, unveränderte Vorgabe an den Agenten (`prompt.txt`) und die
Datei `kennzeichnung.tex`, die das gekennzeichnete PDF aus der
ursprünglichen Ausgabe des Agenten erzeugt.

Beide Arbeiten wurden im Oktober 2026 mit Claude Opus 5.5 (Anthropic),
genutzt über Claude Code, erstellt. Die Vorgaben enthielten den Titel und
formale Vorgaben, aber keine fachliche Gliederung, keine Berechnungen und
keinen Programmcode.

#### `sap-btp/`

*Technische und ökonomische Aspekte der Entwicklung kommerzialisierbarer
Micro-Apps auf SAP BTP* (57 Seiten): Literatur- und Dokumentenanalyse,
Referenzarchitektur, Kosten- und Erlösmodell, Fallstudie.

#### `test-coverage/`

*Generierung von Beispielcode mit definierten Eigenschaften und
Vorhersage der Test-Coverage* (55 Seiten): Der Agent hat dafür auch einen
Prototyp entwickelt, die Versuche ausgeführt und ausgewertet.

- `prototyp/` – der Prototyp `ppgen` (Python), die Versuchsskripte in
  `experiments/` und die Messdaten in `data/`. Die Versuche lassen sich
  nach Anhang A der Arbeit reproduzieren; benötigte Pakete stehen in
  `requirements.txt`. In den Protokolldateien sind lokale Pfade durch
  relative ersetzt.
- Die Vorgabe nennt als Autorenbezeichnung „GPT-6.1 Sol (Hoch)“;
  erstellt wurde die Arbeit jedoch mit Claude Opus 5.5, das auf dem
  Titelblatt steht.
- Die Arbeit enthält die eidesstattliche Erklärung der Hochschulvorlage,
  unterschrieben mit der Modellbezeichnung. Sie hat keinerlei rechtliche
  Bedeutung; ein KI-System kann keine solche Erklärung abgeben.

#### Zum Titelblatt

Die Angaben auf den Titelblättern (Hochschule, Fakultät, Studiengang,
angestrebter Grad, Betreuung) stammen aus der verwendeten LaTeX-Vorlage
für Abschlussarbeiten und beschreiben kein reales Prüfungsverfahren. Die
Betreuenden sind Platzhalter („N. N.“). Das Titelblatt der
Hochschulvorlage wurde bewusst unverändert gelassen. Es zeigt, dass der
Agent eine formal vollständige Arbeit in der realen, öffentlich
verfügbaren Vorlage erzeugt hat.

### `sortiergespraech/`

- `sortieralgorithmen-konversation-ascii.txt` – Transkript eines Gesprächs
  mit einem KI-System über Verfahren zum Sortieren eines Arrays von
  100 000 Zahlen. Es illustriert die im Beitrag vorgeschlagene
  Aufgabenform, bei der der Gesprächsverlauf statt des Quellcodes
  abgegeben wird. Umlaute sind als ae, oe, ue wiedergegeben.

## Lizenz

- Demonstrationsarbeiten (die beiden PDF-Dateien in
  `demonstrationsarbeiten/`):
  [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/deed.de).
  Weitergabe nur unverändert, einschließlich Hinweisblatt und Vermerken.
  Ausgenommen ist das Logo der Hochschule Offenburg.
- Prototyp (`demonstrationsarbeiten/test-coverage/prototyp/`):
  [MIT-Lizenz](demonstrationsarbeiten/test-coverage/prototyp/LICENSE).
- Alle übrigen Inhalte:
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.de).

Siehe auch [`LICENSE.md`](LICENSE.md).
