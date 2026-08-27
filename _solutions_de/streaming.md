---
title: Streaming
description: Kontinuierliche Verarbeitung und Übertragung von Daten.
category:
- Performance
- Architecture
problems:
- slow-application-performance
- unbounded-data-growth
- growing-task-queues
- bottleneck-formation
- scaling-inefficiencies
- work-queue-buildup
- unoptimized-file-access
layout: solution
lang: de
en_slug: streaming
related_solutions:
- slug: data-stream-processing
  similarity: 0.9
- slug: pipelining
  similarity: 0.75
- slug: batch-processing
  similarity: 0.75
- slug: distributed-processing
  similarity: 0.75
- slug: data-replication
  similarity: 0.7
- slug: in-memory-processing
  similarity: 0.7
---

## Description

Streaming ersetzt batch-orientierte Datenverarbeitung — bei der sich Datensätze in einem Staging-Bereich ansammeln und periodisch in geplanten Läufen verarbeitet werden — durch kontinuierliche, ereignisweise Verarbeitung, während Daten ankommen, typischerweise gebaut auf einer Plattform wie Kafka, Kinesis oder RabbitMQ Streams, oft kombiniert mit Change Data Capture, um Events aus einer Legacy-Datenbank zu extrahieren, ohne deren Code zu modifizieren. Batch-Verarbeitung war häufig die einzig praktikable Architektur, die verfügbar war, als viele Legacy-Systeme gebaut wurden, aber Batch-Fenster skalieren nicht elegant: Während das Datenvolumen wächst, braucht ein stündlicher Job, der einst komfortabel innerhalb seines Fensters fertig wurde, irgendwann länger als eine Stunde, um zu laufen, und der Rückstand, den er produziert, häuft sich unbegrenzt an, statt sich von selbst aufzulösen. Streaming adressiert dies strukturell statt durch Feinabstimmung des bestehenden Batch-Jobs, weil es jedes Ereignis verarbeitet, während es auftritt, statt darauf zu warten, einen großen Zwischendatensatz anzusammeln und dann zu verarbeiten, was sowohl den Rückstands-Fehlermodus beseitigt als auch die End-to-End-Latenz von Stunden auf Sekunden zusammenbrechen lässt. Dies ist besonders relevant für Legacy-Modernisierung, weil es erlaubt, Echtzeitfähigkeiten — live kundenseitige Statusaktualisierungen, nahezu-Echtzeit-Analytik — auf die Daten eines Legacy-Systems zu schichten, ohne notwendigerweise das System neu zu schreiben, das die Daten produziert, vorausgesetzt ein Change-Data-Capture-Mechanismus kann es beobachten. Die Kosten sind eine echte Zunahme betrieblicher Komplexität: Streaming-Pipelines erfordern spezialisiertes Tooling zum Debuggen, Exactly-once-Verarbeitungssemantik ist schwer zu garantieren, und Legacy-Systeme, die vollständig um ein Batch-Denkmodell herum gebaut sind, könnten echtes Refactoring brauchen, bevor sie überhaupt Events produzieren können.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie batch-orientierte Prozesse, die von kontinuierlicher Verarbeitung profitieren könnten (z. B. Dateneingabe, Ereignisverarbeitung, ETL-Pipelines)
- Führen Sie eine Streaming-Plattform (Kafka, RabbitMQ Streams, Kinesis) als Rückgrat für Echtzeit-Datenfluss ein
- Refaktorieren Sie Batch-Dateiübertragungsintegrationen in Event-Streams, wobei Events produziert werden, während sie auftreten, statt sie anzusammeln
- Implementieren Sie Stream-Verarbeitung für Legacy-Berichte, die derzeit von Tagesabschluss-Batch-Läufen abhängen
- Nutzen Sie Change Data Capture (CDC), um Datenbankänderungen aus Legacy-Systemen zu streamen, ohne deren Code zu modifizieren
- Wenden Sie Windowing und Aggregation auf Stream-Ebene an, um nahezu-Echtzeit-Analytik zu produzieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Ermöglicht Echtzeit-Datenverarbeitung und reduziert die End-to-End-Latenz von Stunden auf Sekunden
- Handhabt kontinuierliche Datenflüsse, ohne große Zwischendatensätze anzusammeln
- Entkoppelt Produzenten und Konsumenten und erlaubt unabhängige Skalierung und Evolution
- Handhabt Backpressure natürlich durch Consumer-Gruppen-Verwaltung

**Kosten und Risiken:**
- Streaming-Infrastruktur fügt betriebliche Komplexität im Vergleich zu einfacher Batch-Verarbeitung hinzu
- Exactly-once-Verarbeitungssemantik ist schwer zu erreichen und zu verifizieren
- Das Debuggen von Streaming-Pipelines erfordert spezialisiertes Tooling und Expertise
- Legacy-Systeme, die um Batch-Paradigmen herum gestaltet sind, könnten erhebliches Refactoring brauchen, um Events zu produzieren
- Reihenfolgegarantien über Partitionen hinweg erfordern sorgfältiges Design

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Das Legacy-System eines Logistikunternehmens sammelte Sendungsverfolgungsereignisse in einer Staging-Tabelle an und verarbeitete sie in stündlichen Batch-Läufen. Während das Sendungsvolumen wuchs, brauchte der stündliche Batch länger als eine Stunde zum Abschluss, was einen wachsenden Rückstand verursachte. Das Team implementierte Kafka-basiertes Event-Streaming mit CDC auf der Legacy-Datenbank und verarbeitete Verfolgungsereignisse, während sie ankamen. Dies beseitigte den Batch-Rückstand, reduzierte die Verfolgungsaktualisierungslatenz von bis zu zwei Stunden auf unter 10 Sekunden und ermöglichte Echtzeit-Kundenbenachrichtigungen, die zuvor unmöglich gewesen waren.
