---
title: In-Memory-Verarbeitung
description: Vorhaltung aller Daten im Hauptspeicher statt auf langsamen Speichermedien.
category:
- Performance
problems:
- slow-application-performance
- slow-database-queries
- excessive-disk-io
- gradual-performance-degradation
- high-database-resource-utilization
- unoptimized-file-access
layout: solution
lang: de
en_slug: in-memory-processing
related_solutions:
- slug: distributed-caching
  similarity: 0.75
- slug: distributed-processing
  similarity: 0.75
- slug: lazy-loading
  similarity: 0.75
- slug: memory-hierarchy
  similarity: 0.75
- slug: data-replication
  similarity: 0.75
- slug: data-archiving
  similarity: 0.75
---

## Description

In-Memory-Verarbeitung verschiebt häufig genutzte Daten aus plattenbasiertem Speicher in den Hauptspeicher — über ein In-Memory-Daten-Grid, eine In-Memory-Datenbankfunktion oder anwendungsseitige Datenstrukturen —, sodass Lese- und Schreibvorgänge Festplatten-Suchlatenz vollständig vermeiden und in Mikrosekunden statt Millisekunden abgeschlossen werden. Viele Legacy-Systeme wurden um die Annahme herum konstruiert, dass plattenbasierte relationale Speicherung zur Zeit ihrer Erstellung die einzig wirtschaftlich tragfähige Option war, eine Annahme, die inzwischen von den fallenden Speicherkosten überholt wurde, sodass heiße, häufig gelesene Referenzdaten oder Arbeitsmengen aus keinem anderen Grund als historischer Trägheit auf der Festplatte liegen. Dies wird zu einem kritischen Engpass speziell bei Workloads mit plötzlicher konzentrierter Nachfrage — Echtzeit-Risikoberechnungen bei Markteröffnung, Sitzungsabfragen während einer Verkehrsspitze —, wo Festplatten-I/O-Sättigung kaskadierende Verzögerungen verursacht, die keine Menge an Abfrageoptimierung auf dem plattenbasierten Pfad vollständig lösen kann. Die relevante Arbeitsmenge in den Speicher zu verschieben, typischerweise zusammen mit einem Persistenzmechanismus wie Snapshotting oder Write-Ahead-Logging zum Schutz vor Datenverlust bei Ausfall, wandelt diese Berechnungen von I/O-gebunden zu CPU-gebunden um und ermöglicht Echtzeit-Reaktionsfähigkeit, die ein plattenbasiertes Legacy-Design strukturell nicht bieten kann. Die entsprechende Einschränkung ist, dass Speicherkapazität weit teurer und begrenzter ist als Festplatte, sodass dieser Ansatz bewusst auf die spezifische heiße Teilmenge der Daten beschränkt wird, die davon profitiert, statt als vollständiger Ersatz für den persistenten Speicher des Systems angewendet zu werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie Workloads, bei denen Festplatten-I/O der primäre Engpass ist: häufig genutzte Referenzdaten, Sitzungsspeicher, Echtzeit-Analytik
- Verschieben Sie heiße Daten in In-Memory-Datenstrukturen oder In-Memory-Datenbanken (Redis, Apache Ignite, Hazelcast)
- Ziehen Sie für relationale Workloads In-Memory-Tabellenfunktionen in Betracht, die von Datenbanken wie SAP HANA, SQL Server oder PostgreSQL-Erweiterungen angeboten werden
- Gestalten Sie Datenstrukturen, die für Speicherzugriffsmuster statt plattenbasierte Layouts optimiert sind
- Implementieren Sie Persistenzstrategien (Snapshots, Write-Ahead-Logs), um vor Datenverlust bei Ausfällen zu schützen
- Bemessen Sie die Speicherzuweisung basierend auf der Arbeitsmenge plus Wachstumsprognosen, nicht nur dem aktuellen Datenvolumen
- Überwachen Sie Speichernutzung und Garbage-Collection-Overhead, um Swapping und OOM-Situationen zu vermeiden

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Beseitigt Festplatten-I/O-Latenz und bietet um Größenordnungen schnelleren Datenzugriff
- Ermöglicht komplexe Berechnungen und Abfragen, die mit plattenbasiertem Speicher unpraktikabel sind
- Bietet vorhersagbare, geringvariante Antwortzeiten, die keinen Festplatten-Suchmustern unterliegen
- Ermöglicht Echtzeit-Analytik und -Verarbeitung, die batch-orientierte Plattensysteme nicht unterstützen können

**Kosten und Risiken:**
- Speicher ist erheblich teurer als Plattenspeicher, was das In-Memory haltbare Datenvolumen begrenzt
- Datenverlustrisiko bei Ausfällen, sofern Persistenzmechanismen nicht ordentlich konfiguriert sind
- Garbage-Collection-Pausen in verwalteten Speicher-Laufzeitumgebungen können Latenzspitzen verursachen
- Speicherkapazitätsgrenzen erfordern sorgfältiges Datenlebenszyklusmanagement

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-Handelsplattform führte Echtzeit-Risikoberechnungen durch, indem sie eine plattenbasierte relationale Datenbank nach Positionsdaten abfragte. Bei Markteröffnung, wenn sich Tausende Positionen gleichzeitig änderten, wurde die Festplatten-I/O gesättigt, und Risikoberechnungen hinkten den Marktbewegungen um mehrere Minuten hinterher. Das Team migrierte die Positionsdaten zu einem In-Memory-Daten-Grid, lud die Positionen des aktuellen Tages beim Start in den Speicher und aktualisierte sie über Marktdaten-Events. Risikoberechnungen, die zuvor Sekunden pro Position brauchten, wurden nun in Mikrosekunden abgeschlossen, was dem System erlaubte, selbst unter den volatilsten Marktbedingungen Echtzeit-Risikosichtbarkeit zu bewahren.
