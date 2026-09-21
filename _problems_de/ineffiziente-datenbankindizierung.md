---
title: Ineffiziente Datenbankindizierung
description: Der Datenbank fehlen angemessene Indizes für gängige Abfragemuster,
  was langsame, vollständige Tabellen-Scans für Datenabrufoperationen erzwingt.
category:
- Database
- Performance
related_problems:
- slug: incorrect-index-type
  similarity: 0.85
- slug: slow-database-queries
  similarity: 0.8
- slug: queries-that-prevent-index-usage
  similarity: 0.75
- slug: database-query-performance-issues
  similarity: 0.7
- slug: high-number-of-database-queries
  similarity: 0.7
- slug: index-fragmentation
  similarity: 0.7
solutions:
- query-optimization-process
- data-modeling
- performance-measurements
- load-testing
- profiling
- continuous-performance-monitoring
- monitoring
- index-lifecycle-management
layout: problem
lang: de
en_slug: inefficient-database-indexing
---

## Description
Ineffiziente Datenbankindizierung ist eine verbreitete Ursache langsamer Datenbankabfragen. Dies kann durch verschiedene Faktoren verursacht werden, von fehlenden Indizes auf häufig abgefragten Spalten bis zur Nutzung des falschen Index-Typs für die Daten. Eine wirksame Indizierungsstrategie ist essenziell, um sicherzustellen, dass die Datenbank Daten schnell und effizient abrufen kann. Dies erfordert ein tiefes Verständnis der Daten, der ausgeführten Abfragen und der verschiedenen verfügbaren Index-Typen.

## Indicators ⟡
- Abfragen sind langsam, obwohl sie gegen eine kleine Datenmenge laufen.
- Die Datenbank nutzt einen vollständigen Tabellen-Scan, obwohl ein Index verfügbar ist.
- Die Datenbank nutzt einen weniger effizienten Index als erwartet.
- Die Datenbank nutzt nicht den Index, den man erwarten würde.

## Symptoms ▲

- [Langsame Datenbankabfragen](langsame-datenbankabfragen.md)
<br/>  Fehlende oder unpassende Indizes erzwingen vollständige Tabellen-Scans, was direkt langsame Abfrageausführungszeiten verursacht.
- [Performance-Probleme bei Datenbankabfragen](performance-probleme-bei-datenbankabfragen.md)
<br/>  Schlechte Indizierung ist ein primärer Treiber der allgemeinen Verschlechterung der Datenbankabfrageperformance.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Langsame Datenbankabfragen durch fehlende Indizes kaskadieren zu langsamen Anwendungsantwortzeiten.
- [Hohe Datenbank-Ressourcennutzung](hohe-datenbank-ressourcennutzung.md)
<br/>  Vollständige Tabellen-Scans verbrauchen weit mehr CPU- und I/O-Ressourcen als indizierte Lookups, was den Datenbankserver belastet.

## Causes ▼

- [Probleme im Datenbankschema-Design](probleme-im-datenbankschema-design.md)
<br/>  Die Vernachlässigung regelmäßiger Überprüfung und Optimierung von Datenbankindizes erlaubt es ineffizienter Indizierung, fortzubestehen.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler, die mit Datenbank-Performance-Tuning nicht vertraut sind, versäumen es, angemessene Indizes für ihre Abfragemuster zu erstellen.
- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Ohne Dokumentation von Abfragemustern und Indizierungsstrategien sind ordentliche Indizierungsentscheidungen schwer zu treffen, während sich das System weiterentwickelt.
- [Falscher Index-Typ](falscher-index-typ.md)
<br/>  Falsche Datenbank-Index-Typen verschlechtern die Abfrageperformance.

## Detection Methods ○

- **Analyse des Abfrageausführungsplans:** Nutzung des `EXPLAIN`- oder `EXPLAIN ANALYZE`-Befehls, um zu sehen, wie die Datenbank eine Abfrage ausführt. Achten auf "Sequential Scan" oder "Table Scan" auf großen Tabellen, was auf einen fehlenden Index hindeutet.
- **Datenbank-Indizierungsberater:** Viele Datenbanksysteme haben eingebaute Werkzeuge oder Berater, die Abfragehistorie analysieren und neue Indizes vorschlagen können.
- **Monitoring-Werkzeuge:** Nutzung von Datenbank-Monitoring-Werkzeugen zur Identifikation von Abfragen mit hoher I/O und Überprüfung, ob sie angemessene Indizes nutzen.
- **Manuelle Schema-Überprüfung:** Manuelle Untersuchung des Datenbankschemas und Vergleich der Indizes mit den häufigsten und wichtigsten Abfragemustern im Anwendungscode.

## Examples
Ein Nutzersuchfeature in einer Anwendung ist sehr langsam. Die `users`-Tabelle ist auf der `id`-Spalte indiziert, aber Nutzer werden nach ihrer `email`-Adresse gesucht. Die Ausführung von `EXPLAIN` auf der Suchabfrage bestätigt, dass die Datenbank einen vollständigen Tabellen-Scan auf der `users`-Tabelle durchführt. Das Hinzufügen eines Index auf der `email`-Spalte löst das Problem. In einem anderen Fall ist eine Abfrage wie `SELECT * FROM orders WHERE YEAR(order_date) = 2023;` langsam, obwohl es einen Index auf `order_date` gibt. Die Nutzung der `YEAR()`-Funktion verhindert, dass die Datenbank den Index direkt nutzt. Die Abfrage könnte als `SELECT * FROM orders WHERE order_date >= '2023-01-01' AND order_date < '2024-01-01';` umgeschrieben werden, um die Nutzung des Index zu erlauben. Dies ist ein sehr verbreitetes Problem in Legacy-Anwendungen, in denen im Laufe der Zeit neue Features und Abfragemuster hinzugefügt wurden, ohne eine entsprechende Überprüfung der Datenbank-Indizierungsstrategie.
