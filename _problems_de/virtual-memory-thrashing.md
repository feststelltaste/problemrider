---
title: Virtual-Memory-Thrashing
description: Das System tauscht konstant Seiten zwischen physischem Speicher und
  Festplatte aus, was schwere Performance-Verschlechterung durch exzessive Paging-Aktivität
  verursacht.
category:
- Code
- Performance
related_problems:
- slug: memory-swapping
  similarity: 0.7
- slug: memory-fragmentation
  similarity: 0.55
- slug: garbage-collection-pressure
  similarity: 0.5
- slug: priority-thrashing
  similarity: 0.5
solutions:
- backpressure
- elastic-scaling
- memory-management-optimization
- resource-usage-optimization
- profiling
- performance-measurements
- monitoring-system-utilization
- capacity-planning
- resource-pooling
layout: problem
lang: de
en_slug: virtual-memory-thrashing
---

## Description

Virtual-Memory-Thrashing tritt auf, wenn der aktive Working Set eines Systems den verfügbaren physischen Speicher übersteigt, was das Betriebssystem zwingt, konstant Seiten zwischen RAM und Festplattenspeicher auszutauschen. Dies schafft einen destruktiven Kreislauf, in dem das System mehr Zeit mit der Verwaltung des virtuellen Speichers verbringt als mit der Ausführung von Anwendungscode, was zu schwerer Performance-Verschlechterung und Systemunresponsivität führt.

## Indicators ⟡

- Extrem hohe Festplatten-I/O-Aktivität mit minimaler tatsächlicher Datenverarbeitung
- Die Systemreaktionsfähigkeit sinkt unter Speicherdruck erheblich
- Page-Fault-Raten steigen während speicherintensiver Operationen dramatisch
- Verfügbarer physischer Speicher liegt konsequent nahe null
- Die Swap-Datei-Nutzung wächst schnell und bleibt hoch

## Symptoms ▲

- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Thrashing verursacht, dass das System die meiste Zeit mit dem Austausch von Seiten statt der Verarbeitung verbringt, was zu schwerer Performance-Verschlechterung führt.
- [Übermäßige Festplatten-I/O](uebermaessige-festplatten-io.md)
<br/>  Konstanter Seitenaustausch zwischen RAM und Festplatte generiert extrem hohe Festplatten-I/O-Aktivität.
- [Service-Timeouts](service-timeouts.md)
<br/>  Anwendungen werden während des Thrashings so langsam, dass sie nicht innerhalb der Timeout-Fenster antworten.
- [Systemausfälle](systemausfaelle.md)
<br/>  Schweres Thrashing kann ein System vollständig unresponsiv machen, was effektiv einen Systemausfall verursacht, wenn die Anwendung nicht mehr antworten kann.

## Causes ▼

- [Speicherlecks](speicherlecks.md)
<br/>  Speicherlecks verbrauchen graduell verfügbaren RAM, bis das System stark auf virtuellen Speicher angewiesen sein muss, was Thrashing verursacht.
- [Ressourcenkonkurrenz](ressourcenkonkurrenz.md)
<br/>  Mehrere Prozesse, die um begrenzte Speicherressourcen konkurrieren, verursachen, dass das System die physische Speicherkapazität übersteigt.
- [Unbegrenztes Datenwachstum](unbegrenztes-datenwachstum.md)
<br/>  Wachsende Datensätze, die in den Speicher geladen werden, können die physische RAM-Kapazität übersteigen, was Thrashing auslöst.

## Detection Methods ○

- **System-Speicher-Monitoring:** Überwachung der physischen Speichernutzung, Swap-Nutzung und verfügbaren Speichers
- **Page-Fault-Analyse:** Verfolgung von Page-Fault-Raten und -Arten (Minor vs. Major Faults)
- **Festplatten-I/O-Monitoring:** Analyse von Festplatten-I/O-Mustern zur Identifikation paging-bezogener Aktivität
- **Working-Set-Analyse:** Messung der Working-Set-Größen von Anwendungen im Verhältnis zum verfügbaren Speicher
- **Performance-Profiling:** Profiling von Anwendungen unter Speicherdruck zur Identifikation von Thrashing-Mustern
- **Virtual-Memory-Statistiken:** Überwachung von Virtual-Memory-Systemstatistiken und Swap-Datei-Aktivität

## Examples

Ein Datenbankserver mit 8 GB RAM versucht, einen Datensatz zu verarbeiten, der 12 GB Speicher erfordert. Während Abfragen auf verschiedene Teile des Datensatzes zugreifen, tauscht das Betriebssystem konstant Seiten zwischen Speicher und Festplatte aus. Jede Datenbankoperation, die Millisekunden dauern sollte, dauert jetzt aufgrund von Festplattenzugriffsverzögerungen Sekunden, und das System verbringt 90 % seiner Zeit mit Speicherverwaltung statt Abfrageverarbeitung. Ein weiteres Beispiel betrifft ein Batch-Verarbeitungssystem, das mehrere Worker-Prozesse erstellt, von denen jeder große Datendateien in den Speicher lädt. Wenn die kombinierte Speichernutzung den verfügbaren RAM übersteigt, beginnt das System zu thrashen, während der Speicher jedes Prozesses ausgelagert wird, während andere Prozesse laufen, was einen Kreislauf schafft, in dem kein Prozess seinen Working Set lange genug im Speicher halten kann, um effizient abzuschließen.
