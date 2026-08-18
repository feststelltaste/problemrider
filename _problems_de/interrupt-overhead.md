---
title: Interrupt-Overhead
description: Übermäßige Hardware-Interrupts stören den CPU-Ausführungsfluss, was
  häufige Kontextwechsel verursacht und die Anwendungsperformance verringert.
category:
- Code
- Performance
related_problems:
- slug: context-switching-overhead
  similarity: 0.55
- slug: microservice-communication-overhead
  similarity: 0.55
- slug: atomic-operation-overhead
  similarity: 0.55
- slug: high-client-side-resource-consumption
  similarity: 0.55
- slug: high-resource-utilization-on-client
  similarity: 0.55
- slug: endianness-conversion-overhead
  similarity: 0.5
solutions:
- profiling
- performance-measurements
- asynchronous-processing
- batch-processing
- monitoring-system-utilization
- efficient-algorithms
- load-testing
- continuous-performance-monitoring
- performance-modeling
- static-code-analysis
layout: problem
lang: de
en_slug: interrupt-overhead
---

## Description

Interrupt-Overhead tritt auf, wenn Hardware-Geräte Interrupts mit einer so hohen Frequenz erzeugen, dass die CPU übermäßig viel Zeit mit der Handhabung von Interrupt-Service-Routinen verbringt, statt Anwendungscode auszuführen. Jeder Interrupt erfordert das Speichern des aktuellen Ausführungskontexts, das Ausführen des Interrupt-Handlers und das Wiederherstellen des Kontexts, was in interrupt-lastigen Umgebungen zu einem erheblichen Performance-Engpass werden kann.

## Indicators ⟡

- CPU-Performance-Zähler zeigen hohe Interrupt-Raten
- Die Systemperformance verschlechtert sich unter hoher I/O-Last
- Anwendungen werden während Perioden hoher Interrupt-Aktivität unresponsiv
- Die CPU-Nutzung ist hoch im Interrupt-Kontext statt in Nutzeranwendungen
- Performance-Probleme korrelieren mit spezifischer Hardware-Geräteaktivität

## Symptoms ▲

- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  CPU-Zeit, die für die Handhabung von Interrupts aufgewendet wird, verringert die für Anwendungsverarbeitung verfügbare Zeit, was die Performance verschlechtert.

## Causes ▼

- [Schlechte Systemumgebung](schlechte-systemumgebung.md)
<br/>  Schlecht konfigurierte Hardware-Umgebungen mit suboptimalen Interrupt-Einstellungen tragen zu übermäßigen Interrupt-Raten bei.

## Detection Methods ○

- **Interrupt-Raten-Monitoring:** Überwachung von System-Interrupt-Raten mit Betriebssystem-Performance-Werkzeugen
- **CPU-Interrupt-Zeit-Analyse:** Messung der im Interrupt-Kontext vs. normaler Ausführung verbrachten Zeit
- **Geräteweises Interrupt-Tracking:** Identifikation, welche Geräte die meisten Interrupts erzeugen
- **Interrupt-Verteilungsanalyse:** Überprüfung, wie Interrupts über CPU-Kerne verteilt sind
- **Korrelation der Anwendungsperformance:** Korrelation der Anwendungsperformance mit Interrupt-Aktivität
- **Hardware-Performance-Zähler:** Nutzung von Hardware-Zählern zur Überwachung interruptbezogener Metriken

## Examples

Ein Hochleistungs-Webserver empfängt Netzwerkpakete von einer 10-Gbps-Netzwerkschnittstelle, die für jedes empfangene Paket einen Interrupt erzeugt. Bei Spitzentraffic erzeugt dies 1,5 Millionen Interrupts pro Sekunde, was dazu führt, dass die CPU 40 % ihrer Zeit in Interrupt-Handlern verbringt, statt HTTP-Anfragen zu verarbeiten. Die Aktivierung von Interrupt-Coalescing zum Bündeln mehrerer Pakete pro Interrupt reduziert die Interrupt-Häufigkeit um 90 % und verbessert den Anfrageverarbeitungsdurchsatz um 60 %. Ein weiteres Beispiel betrifft ein Echtzeit-Datenerfassungssystem, bei dem mehrere Sensoren Timer-Interrupts in Mikrosekunden-Intervallen erzeugen. Die kumulierte Interrupt-Last führt dazu, dass die Hauptdatenverarbeitungsschleife kritische Timing-Fristen verpasst, was zu verlorenen Sensormesswerten und verschlechterter Systemgenauigkeit führt. Die Implementierung von Interrupt-Priorisierung und effizienteren Interrupt-Handlern stellt die Echtzeitperformance wieder her.
