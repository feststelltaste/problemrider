---
title: Übermäßige Objektallokation
description: Code erzeugt eine große Anzahl temporärer Objekte, was den Garbage
  Collector belastet und die Performance verschlechtert.
category:
- Code
- Performance
related_problems:
- slug: garbage-collection-pressure
  similarity: 0.8
- slug: algorithmic-complexity-problems
  similarity: 0.65
- slug: memory-fragmentation
  similarity: 0.65
- slug: memory-leaks
  similarity: 0.6
- slug: high-client-side-resource-consumption
  similarity: 0.6
- slug: resource-allocation-failures
  similarity: 0.6
solutions:
- memory-management-optimization
- profiling
- resource-pooling
- resource-usage-optimization
- serialization-optimization
- lazy-evaluation
- lazy-loading
- memory-hierarchy
layout: problem
lang: de
en_slug: excessive-object-allocation
---

## Description

Übermäßige Objektallokation entsteht, wenn Code eine unnötig große Anzahl temporärer Objekte erzeugt, besonders in häufig ausgeführten Codepfaden. Dies belastet den Garbage Collector, erhöht den Speicherverbrauch und kann die Anwendungsperformance erheblich verschlechtern. Während Objekterzeugung in der objektorientierten Programmierung normal ist, kann übermäßige Allokation in kritischen Pfaden Performance-Probleme verursachen, die sich verschlimmern, während die Anwendung skaliert oder mehr Daten verarbeitet.

## Indicators ⟡
- Garbage Collection tritt häufig auf und verbraucht erhebliche CPU-Zeit
- Der Speicherverbrauch steigt während des normalen Betriebs sprunghaft an, auch ohne Speicherlecks
- Die Anwendungsperformance verschlechtert sich während Phasen hoher Aktivität
- Profiling zeigt hohe Objektallokationsraten in bestimmten Codebereichen
- Die Performance verbessert sich erheblich, wenn Object Pooling oder Wiederverwendung umgesetzt wird

## Symptoms ▲

- [Garbage-Collection-Druck](garbage-collection-druck.md)
<br/>  Die Erzeugung großer Mengen temporärer Objekte erhöht direkt Häufigkeit und Dauer der Garbage Collection.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Übermäßiger Allokations- und Garbage-Collection-Overhead verringert die für die tatsächliche Anwendungsverarbeitung verfügbare CPU-Zeit.
- [Speicherfragmentierung](speicherfragmentierung.md)
<br/>  Schnelle Allokation und Deallokation vieler Objekte unterschiedlicher Größe fragmentiert den Heap-Speicher.
- [Schleichende Performance-Verschlechterung](schleichende-performance-verschlechterung.md)
<br/>  Während Datenvolumen zunehmen, skaliert übermäßige Objektallokation proportional mit, was zu fortschreitender Performance-Verschlechterung führt.
- [Hoher Ressourcenverbrauch auf Client-Seite](hoher-ressourcenverbrauch-auf-client-seite.md)
<br/>  Client-Anwendungen mit übermäßiger Objektallokation verbrauchen mehr Speicher und CPU als für GC-Overhead nötig.

## Causes ▼

- [Ineffizienter Code](ineffizienter-code.md)
<br/>  Schlecht geschriebener Code, der unnötige temporäre Objekte in kritischen Pfaden erzeugt, ist die direkte Ursache übermäßiger Allokation.
- [Probleme mit algorithmischer Komplexität](probleme-mit-algorithmischer-komplexitaet.md)
<br/>  Algorithmen, die neue Objekte in inneren Schleifen erzeugen, statt sie wiederzuverwenden, vervielfachen Allokationsraten mit der Datengröße.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler, die mit Speicherverwaltung und GC-Implikationen nicht vertraut sind, schreiben allokationsintensiven Code, ohne die Performance-Auswirkung zu berücksichtigen.
- [Missverständnis der Objektorientierung](missverstaendnis-der-objektorientierung.md)
<br/>  Übermäßige Nutzung von Objekterzeugungsmustern ohne Verständnis, wann Wertetypen oder Object Pooling angemessener wären, führt zu übermäßiger Allokation.

## Detection Methods ○
- **Speicher-Profiling:** Nutzung von Profiling-Werkzeugen zur Identifikation von Codebereichen mit hohen Objektallokationsraten
- **Garbage-Collection-Monitoring:** Nachverfolgung von GC-Häufigkeit, -Dauer und Speicherdruck-Metriken
- **Allokationsraten-Analyse:** Messung von Objekterzeugungsraten in unterschiedlichen Teilen der Anwendung
- **Performance-Tests:** Lasttests, die allokationsbezogene Performance-Probleme aufdecken
- **Fokussiertes Code-Review:** Gezielte Untersuchung von Code auf unnötige Objekterzeugungsmuster

## Examples

Eine Datenverarbeitungsanwendung liest CSV-Dateien und verarbeitet jede Zeile, indem sie ein neues `DataRecord`-Objekt erzeugt, dann jedes Feld in geeignete Typen konvertiert, indem zusätzliche temporäre Objekte für Validierung und Transformation erzeugt werden. Für eine Datei mit 1 Million Zeilen und 20 Spalten erzeugt dies über 20 Millionen temporäre Objekte innerhalb einer einzigen Verarbeitungsoperation. Die übermäßige Allokation führt dazu, dass der Garbage Collector kontinuierlich läuft, 60 % der CPU-Zeit verbraucht und die Verarbeitung 10-mal langsamer als nötig macht. Ein Refactoring zur Wiederverwendung von Objekten und Nutzung primitiver Typen wo möglich reduziert die Verarbeitungszeit von 10 Minuten auf 1 Minute. Ein weiteres Beispiel betrifft eine Webanwendung, die JSON-Antworten baut, indem sie wiederholt Strings in einer Schleife verkettet, was Tausende temporärer String-Objekte für jede API-Antwort erzeugt. Während Phasen hohen Verkehrs verbringt der Server mehr Zeit in der Garbage Collection als mit der Verarbeitung tatsächlicher Anfragen. Nutzer erleben langsame Antwortzeiten, und der Server benötigt mehr Speicher- und CPU-Ressourcen als vergleichbare Anwendungen. Der Wechsel zu einem StringBuilder oder Streaming-JSON-Writer beseitigt das Performance-Problem und reduziert die Serverressourcenanforderungen um 70 %.
