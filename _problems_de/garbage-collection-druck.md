---
title: Garbage-Collection-Druck
description: Übermäßige Objektallokation und -deallokation verursacht häufige Garbage-Collection-Zyklen,
  was Performance-Pausen erzeugt und den Anwendungsdurchsatz verringert.
category:
- Code
- Performance
related_problems:
- slug: excessive-object-allocation
  similarity: 0.8
- slug: memory-fragmentation
  similarity: 0.6
- slug: resource-allocation-failures
  similarity: 0.55
- slug: gradual-performance-degradation
  similarity: 0.55
- slug: high-client-side-resource-consumption
  similarity: 0.55
- slug: stack-overflow-errors
  similarity: 0.55
solutions:
- memory-management-optimization
- profiling
- resource-pooling
- resource-usage-optimization
- serialization-optimization
- performance-measurements
- continuous-performance-monitoring
- load-testing
- efficient-algorithms
layout: problem
lang: de
en_slug: garbage-collection-pressure
---

## Description

Garbage-Collection-Druck entsteht, wenn Anwendungen Objekte in einer so hohen Rate erzeugen und verwerfen, dass der Garbage Collector häufig laufen muss, um Speicher zurückzugewinnen, was merkliche Performance-Pausen verursacht und den Gesamtdurchsatz verringert. Dieses Problem ist besonders schwerwiegend bei Anwendungen mit hohen Allokationsraten, großen Objektgraphen oder unpassenden Objektlebensdauer-Mustern, die das Garbage-Collection-System belasten.

## Indicators ⟡

- Häufige Garbage-Collection-Zyklen unterbrechen die Anwendungsausführung
- GC-Pausenzeiten steigen über die Lebensdauer der Anwendung
- Hohe Allokationsraten zeigen sich beim Memory-Profiling
- Der Anwendungsdurchsatz sinkt aufgrund von GC-Overhead
- Speichernutzungsmuster zeigen schnelle Allokations- und Collection-Zyklen

## Symptoms ▲

- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Häufige GC-Pausen verursachen direkt nutzerseitig spürbare Trägheit und unresponsives Verhalten der Anwendung.
- [Schleichende Performance-Verschlechterung](schleichende-performance-verschlechterung.md)
<br/>  Während sich Objektallokationsmuster über die Zeit verschlechtern, steigt der GC-Druck schrittweise, was zu langsamer, aber stetiger Performance-Verschlechterung führt.
- [Hohe API-Latenz](hohe-api-latenz.md)
<br/>  GC-Pausenzeiten addieren sich direkt zu API-Antwortzeiten, was unvorhersehbare Latenzspitzen während Garbage-Collection-Zyklen verursacht.
- [Service-Timeouts](service-timeouts.md)
<br/>  Lange GC-Pausen können dazu führen, dass Anfragen Timeout-Schwellenwerte überschreiten, was zu fehlgeschlagenen Serviceaufrufen führt.

## Causes ▼

- [Übermäßige Objektallokation](uebermaessige-objektallokation.md)
<br/>  Das Erzeugen einer großen Anzahl temporärer Objekte erhöht direkt die Rate, mit der der Garbage Collector laufen muss, um Speicher zurückzugewinnen.
- [Ineffizienter Code](ineffizienter-code.md)
<br/>  Schlecht geschriebener Code, der unnötige Zwischenobjekte erzeugt oder Objekte nicht wiederverwendet, belastet den Garbage Collector übermäßig.
- [Zirkuläre Referenzen](zirkulaere-referenzen.md)
<br/>  Zirkuläre Objektreferenzen verhindern effiziente Garbage Collection und können dazu führen, dass der GC härter arbeiten muss, um rückgewinnbaren Speicher zu identifizieren.
- [Speicherlecks](speicherlecks.md)
<br/>  Speicherlecks verringern den verfügbaren Heap-Speicher und erzwingen häufigere Garbage-Collection-Zyklen auf dem verbleibenden Speicher.

## Detection Methods ○

- **GC-Logging:** Aktivierung des Garbage-Collector-Loggings zur Analyse von Collection-Häufigkeit und -Dauer
- **Memory-Profiling:** Nutzung von Profilern zur Nachverfolgung von Objektallokationsraten und Garbage-Collection-Auswirkungen
- **Anwendungs-Performance-Monitoring:** Überwachung der Korrelation zwischen Durchsatz, Antwortzeit und GC-Aktivität
- **Heap-Analyse:** Analyse von Heap-Dumps zur Identifikation von Objektallokationsmustern und Lebensdauern
- **GC-Tuning-Metriken:** Überwachung GC-spezifischer Metriken wie Collection-Zeitanteil und Pausendauer
- **Allokations-Profiling:** Profiling von Objektallokations-Hotpaths und -Mustern

## Examples

Ein Webservice verarbeitet JSON-Anfragen, indem er sie in Objektgraphen parst, die Daten verarbeitet und Antworten serialisiert. Das Parsen erzeugt Tausende temporärer Objekte pro Anfrage, und bei hoher Last läuft der Garbage Collector alle paar Sekunden, was 100-200ms-Pausen verursacht, die die API unresponsiv machen. Der Durchsatz der Anwendung sinkt um 40 %, weil Zeit in der Garbage Collection statt in der Anfrageverarbeitung verbracht wird. Ein weiteres Beispiel betrifft eine Datenanalyseanwendung, die große Datensätze verarbeitet, indem sie für jeden Datentransformationsschritt Zwischen-Collection-Objekte erzeugt. Die Anwendung erzeugt Millionen temporärer Listen- und Map-Objekte, was dazu führt, dass der Garbage Collector fast durchgehend läuft und die Datenverarbeitung aufgrund des konstanten Speicherverwaltungs-Overheads 10-mal länger dauert als nötig.
