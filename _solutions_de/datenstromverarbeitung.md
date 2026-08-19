---
title: Datenstromverarbeitung
description: Kontinuierliche Verarbeitung von Daten aus Echtzeit-Datenquellen.
category:
- Performance
- Architecture
problems:
- slow-application-performance
- growing-task-queues
- task-queues-backing-up
- gradual-performance-degradation
- scaling-inefficiencies
layout: solution
lang: de
en_slug: data-stream-processing
related_solutions:
- slug: streaming
  similarity: 0.9
- slug: pipelining
  similarity: 0.7
- slug: distributed-processing
  similarity: 0.7
- slug: batch-processing
  similarity: 0.7
- slug: business-event-processing
  similarity: 0.65
- slug: in-memory-processing
  similarity: 0.65
---

## Description

Datenstromverarbeitung ersetzt batchorientierte Datenverarbeitung durch kontinuierliche, inkrementelle Verarbeitung von Datensätzen, während sie eintreffen, typischerweise auf einer Streaming-Plattform (Kafka, Pulsar, Kinesis) aufgebaut, die Events erfasst, und einem Stream-Prozessor, der Logik auf jedes Event einzeln oder über ein begrenztes Zeitfenster anwendet, statt zu warten, um einen angehäuften Batch nach festem Zeitplan zu verarbeiten. Der Mechanismus tauscht die konzentrierte periodische Last des Batch-Modells gegen eine kontinuierliche, verteilte, und lässt dabei die Latenz zwischen dem Auftreten eines Events und der Reaktion des Systems darauf von der Länge des Batch-Intervalls auf Sekunden kollabieren. Diese Unterscheidung ist in Legacy-Systemen folgenreich, die kritische Logik — Betrugserkennung, Alarmierung, Abgleich — immer noch als nächtliche oder stündliche Batch-Jobs ausführen, rein weil das das einzige Verarbeitungsmodell war, das verfügbar war, als der Job ursprünglich geschrieben wurde, mit dem Ergebnis, dass zu dem Zeitpunkt, an dem ein Problem durch den Batch-Job erkannt wird, das Fenster, in dem etwas dagegen hätte unternommen werden können, bereits geschlossen ist. Die Migration zu Stream Processing erfolgt typischerweise schrittweise, wobei der neue Stream-Prozessor parallel zum bestehenden Batch-Job läuft, sodass beide auf Korrektheit verglichen werden können, bevor der Batch-Job stillgelegt wird, und erfordert die Entscheidung für eine spezifische Zustellungsgarantie (mindestens einmal oder genau einmal), die zur Geschäftsanforderung passt, statt anzunehmen, dass die Garantie, die der Legacy-Batch-Code implizit bot, weiterhin gilt. Weil Legacy-Systeme häufig keine Events nativ emittieren, erfordert die Einführung von Streaming oft zunächst einen Adapter — Change Data Capture oder eine Polling-Brücke —, um einen Event-Strom aus einem System zu erzeugen, das nie dafür entworfen wurde.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie Batch-Verarbeitungsjobs im Legacy-System, die von kontinuierlicher, inkrementeller Verarbeitung profitieren könnten
- Führen Sie eine Streaming-Plattform (Kafka, Pulsar, Kinesis) als Rückgrat für ereignisgesteuerten Datenfluss ein
- Konvertieren Sie Batch-ETL-Pipelines in Stream-Prozessoren, die Datensätze verarbeiten, während sie eintreffen
- Implementieren Sie Windowing-Strategien für Aggregationen, die über zeitlich begrenzte Datensegmente operieren müssen
- Gestalten Sie für Exactly-once- oder At-least-once-Semantik basierend auf den Geschäftsanforderungen jedes Streams
- Fügen Sie Backpressure-Mechanismen hinzu, um Traffic-Spitzen zu handhaben, ohne nachgelagerte Konsumenten zu überwältigen
- Führen Sie Stream Processing während der Migration parallel zu bestehenden Batch-Jobs aus, um Korrektheit zu verifizieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Bietet nahezu Echtzeit-Datenverarbeitung, statt auf Batch-Fenster zu warten
- Verteilt die Verarbeitungslast kontinuierlich über die Zeit, statt sie in Batch-Spitzen zu konzentrieren
- Ermöglicht ereignisgesteuerte Architekturen, die auf Änderungen reagieren, während sie geschehen
- Skaliert horizontal durch Hinzufügen weiterer Stream-Processing-Instanzen

**Kosten und Risiken:**
- Stream-Processing-Infrastruktur fügt operative Komplexität hinzu
- Exactly-once-Semantik ist schwer zu erreichen und kann idempotente Konsumenten erfordern
- Das Debuggen von Streaming-Pipelines ist schwieriger als das Debuggen von Batch-Jobs mit klaren Ein- und Ausgaben
- Legacy-Systeme erzeugen möglicherweise keine Events nativ, was Change Data Capture oder Polling-Adapter erfordert

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Betrugserkennungssystem lief als nächtlicher Batch-Job und analysierte die Transaktionen des Tages auf verdächtige Muster. Bis Betrug erkannt wurde, waren 24 Stunden vergangen, und erhebliche Verluste waren bereits entstanden. Das Team führte Kafka ein, um Transaktions-Events in Echtzeit zu erfassen, und deployte eine Stream-Processing-Anwendung, die Betrugserkennungsregeln auf jede Transaktion anwandte, während sie geschah. Verdächtige Transaktionen wurden innerhalb von Sekunden markiert, was dem Operations-Team erlaubte einzugreifen, bevor Gelder transferiert wurden. Der nächtliche Batch-Job wurde zunächst als Sicherheitsnetz beibehalten und schließlich stillgelegt, nachdem sich die Streaming-Lösung über drei Monate als zuverlässig erwiesen hatte.
