---
title: Stack-Overflow-Fehler
description: Programme überschreiten den zugewiesenen Stack-Speicher aufgrund exzessiver
  Rekursion oder großer lokaler Variablen, was Anwendungsabstürze verursacht.
category:
- Code
- Performance
related_problems:
- slug: buffer-overflow-vulnerabilities
  similarity: 0.65
- slug: memory-leaks
  similarity: 0.6
- slug: excessive-object-allocation
  similarity: 0.6
- slug: null-pointer-dereferences
  similarity: 0.6
- slug: integer-overflow-underflow
  similarity: 0.55
- slug: memory-fragmentation
  similarity: 0.55
solutions:
- memory-management-optimization
- fuzz-testing
- static-analysis-and-linting
- error-handling
- negative-testing
- stress-testing
- code-reviews
- profiling
- exploratory-testing
- design-by-contract
layout: problem
lang: de
en_slug: stack-overflow-errors
---

## Description

Stack-Overflow-Fehler treten auf, wenn der Aufrufstack eines Programms den zugewiesenen Stack-Speicher überschreitet, typischerweise aufgrund unbegrenzter Rekursion, exzessiv tiefer Funktionsaufrufketten oder Zuweisung sehr großer lokaler Variablen. Der Stack ist ein begrenzter Speicherbereich, der für Funktionsaufrufe, lokale Variablen und Rücksprungadressen genutzt wird. Wenn dieser Speicher erschöpft ist, stürzt das Programm mit einem Stack-Overflow-Fehler ab, was schwierig zu debuggen sein kann und auf fundamentale algorithmische oder architektonische Probleme hindeuten kann.

## Indicators ⟡

- Die Anwendung stürzt mit Stack-Overflow- oder „Stack-Speicher überschritten"-Fehlern ab
- Abstürze treten während rekursiver Operationen oder tief verschachtelter Funktionsaufrufe auf
- Die Performance verschlechtert sich vor Abstürzen aufgrund exzessiver Stack-Nutzung
- Stack-Traces zeigen sehr tiefe Aufrufhierarchien oder unendliche Rekursionsmuster
- Speicherprofiling zeigt schnelles Stack-Wachstum während bestimmter Operationen

## Symptoms ▲

- [Systemausfälle](systemausfaelle.md)
<br/>  Stack-Overflow-Fehler lassen die Anwendung abstürzen, was potenziell Ausfälle für Nutzer verursacht.

## Causes ▼

- [Spaghetticode](spaghetticode.md)
<br/>  Verworrener Code mit unvorhersehbaren Aufrufketten kann tiefe oder zirkuläre Aufrufhierarchien schaffen, die den Stack erschöpfen.
- [Komplexe und unklare Logik](komplexe-und-unklare-logik.md)
<br/>  Übermäßig komplexe rekursive Logik ohne ordentliche Terminierungsbedingungen führt zu unbegrenzter Rekursion.
- [Unzureichendes Code-Review](unzureichendes-code-review.md)
<br/>  Ohne Code-Review bleiben unbegrenzte Rekursion und exzessive Stack-Nutzungsmuster unentdeckt.

## Detection Methods ○

- **Stack-Nutzungs-Monitoring:** Überwachung der Stack-Nutzung während der Anwendungsausführung zur Identifikation von Wachstumsmustern
- **Verfolgung der Rekursionstiefe:** Instrumentierung rekursiver Funktionen zur Verfolgung der maximalen Rekursionstiefe
- **Statische Analyse:** Analyse von Code auf potenzielle unbegrenzte Rekursion oder große Stack-Zuweisungen
- **Stresstests:** Testen mit Eingaben, die tiefe Rekursion oder große Stack-Nutzung verursachen könnten
- **Stack-Trace-Analyse:** Untersuchung von Absturz-Stack-Traces zur Identifikation von Rekursionsmustern
- **Profiling-Werkzeuge:** Nutzung von Speicherprofilern zur Überwachung der Stack-Nutzung während des Betriebs

## Examples

Eine Dateisystem-Verzeichnistraversierungsfunktion nutzt Rekursion zur Erkundung verschachtelter Ordner, hat aber kein Maximaltiefenlimit. Bei der Verarbeitung einer Verzeichnisstruktur mit Hunderten verschachtelter Ebenen (entweder legitim oder bösartig erstellt) erschöpfen die rekursiven Aufrufe den Stack-Speicher und lassen die Anwendung abstürzen. Ein weiteres Beispiel betrifft eine mathematische Berechnungsfunktion, die rekursiv Fakultäten berechnet, ohne auf angemessene Eingabegrenzen zu prüfen. Wenn ein Nutzer eine große Zahl wie 50000 eingibt, erstellt die rekursive Fakultätsberechnung Zehntausende von Stack-Frames und stürzt mit einem Stack-Overflow ab, bevor die Berechnung abgeschlossen ist.
