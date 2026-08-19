---
title: Datenpartitionierung
description: Aufteilung großer Datensätze über mehrere Computer oder Speichereinheiten.
category:
- Database
- Performance
problems:
- unbounded-data-growth
- slow-database-queries
- scaling-inefficiencies
- high-database-resource-utilization
- gradual-performance-degradation
layout: solution
lang: de
en_slug: data-partitioning
related_solutions:
- slug: data-replication
  similarity: 0.8
- slug: data-archiving
  similarity: 0.8
- slug: distributed-processing
  similarity: 0.8
- slug: materialized-views
  similarity: 0.8
- slug: denormalization
  similarity: 0.8
- slug: distributed-caching
  similarity: 0.75
---

## Description

Datenpartitionierung teilt einen großen Datensatz in kleinere, unabhängig handhabbare Segmente — nach Datumsbereich, Hash, Geografie oder einem anderen zu den Abfragemustern passenden Schlüssel —, sodass Operationen, die nur eine Teilmenge der Daten berühren, auf die relevanten Partitionen beschränkt werden können, statt die gesamte Tabelle zu scannen. Der Mechanismus beruht auf Partition Pruning: Wenn der Filter einer Abfrage den Partitionierungsschlüssel enthält, kann die Datenbank-Engine jede Partition überspringen, die keine übereinstimmenden Zeilen enthalten kann, was einen sonst vollständigen Tabellenscan in einen Scan verwandelt, der auf die Datenmenge beschränkt ist, die tatsächlich in den angefragten Bereich fällt. Dies ist eine direkte Antwort auf eine häufige Legacy-System-Entwicklung, bei der eine einzelne Tabelle über Jahre Transaktionshistorie anhäuft, bis routinemäßige Abfragen — selbst Jahresendberichte oder tägliche Abgleiche — sich durch Hunderte Millionen Zeilen wühlen müssen, die größtenteils für die gestellte Frage irrelevant sind. Über die Abfrageperformance hinaus macht Partitionierung auch Wartungsoperationen wie Backups und Index-Neuaufbauten wieder handhabbar, indem sie auf einzelnen Partitionen statt dem gesamten Datensatz operieren können, und sie gibt dem Datenlebenszyklusmanagement (Archivierung oder Löschung alter Partitionen) einen sauberen, kostengünstigen Mechanismus, auf dem gehandelt werden kann. Das Hauptrisiko ist, dass der Partitionierungsschlüssel vorab gut gewählt werden muss, da er im Nachhinein schwer zu ändern ist, und jede Abfrage, die ihn auslässt, verliert den Pruning-Vorteil und kann schlechter performen als vor der Einführung der Partitionierung.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Analysieren Sie Abfragemuster, um den besten Partitionierungsschlüssel zu bestimmen (Datumsbereiche, geografische Regionen, Kundensegmente)
- Implementieren Sie Tabellenpartitionierung innerhalb der Datenbank für Zeitreihendaten mittels Range-Partitionierung
- Nutzen Sie Hash-Partitionierung, um Daten gleichmäßig über Partitionen zu verteilen, wenn kein natürlicher Bereichsschlüssel existiert
- Stellen Sie sicher, dass Abfragen den Partitionierungsschlüssel in WHERE-Klauseln enthalten, um Partition Pruning zu ermöglichen
- Planen Sie Partitionswartung: Automatisieren Sie die Erstellung neuer Partitionen und die Archivierung alter
- Testen Sie die Abfrageperformance mit partitionierten Daten, um zu verifizieren, dass Partition Pruning wie erwartet funktioniert
- Erwägen Sie horizontales Sharding über Datenbankinstanzen für extreme Skalierungsanforderungen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Ermöglicht Abfragen, nur relevante Partitionen statt des gesamten Datensatzes zu scannen
- Macht Wartungsoperationen (Backups, Index-Neuaufbauten) handhabbar, indem auf einzelnen Partitionen operiert wird
- Vereinfacht Datenlebenszyklusmanagement: Alte Partitionen können effizient archiviert oder gelöscht werden
- Erlaubt unabhängige Skalierung des Speichers für unterschiedliche Datensegmente

**Kosten und Risiken:**
- Abfragen, die den Partitionierungsschlüssel nicht enthalten, können wegen partitionsübergreifender Scans schlechter performen
- Die Wahl des Partitionierungsschlüssels ist kritisch und schwer zu ändern, nachdem Daten partitioniert wurden
- Anwendungslogik benötigt möglicherweise Aktualisierungen, um partitionsbewusst zu sein
- Partitionsübergreifende Transaktionen und Joins sind komplexer und potenziell langsamer

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Finanztransaktionssystem speicherte alle Transaktionen in einer einzigen Tabelle, die über acht Jahre auf 800 Millionen Zeilen angewachsen war. Jahresendberichtsabfragen dauerten Stunden, und selbst routinemäßiger täglicher Abgleich war langsam. Das Team implementierte Range-Partitionierung nach Monat, was tägliche Abgleichsabfragen erlaubte, nur die Partition des aktuellen Monats (etwa 8 Millionen Zeilen) statt der gesamten Tabelle zu scannen. Jahresendberichte konnten auf spezifische Jahrespartitionen zielen. Das Team automatisierte auch die Partitionserstellung für zukünftige Monate und richtete vierteljährliche Archivierung von Partitionen ein, die älter als zwei Jahre waren. Die Abfrageperformance verbesserte sich um zwei Größenordnungen für zeitbegrenzte Abfragen.
