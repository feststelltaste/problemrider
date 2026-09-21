---
title: Index-Fragmentierung
description: Über die Zeit, während Daten eingefügt, aktualisiert und gelöscht werden,
  werden Datenbankindizes unorganisiert, was ihre Effizienz verringert.
category:
- Database
- Performance
related_problems:
- slug: unused-indexes
  similarity: 0.7
- slug: inefficient-database-indexing
  similarity: 0.7
- slug: incorrect-index-type
  similarity: 0.65
- slug: database-query-performance-issues
  similarity: 0.6
- slug: queries-that-prevent-index-usage
  similarity: 0.6
- slug: slow-database-queries
  similarity: 0.6
solutions:
- query-optimization-process
- regular-maintenance-and-updates
- monitoring
- performance-measurements
- production-environment-maintenance
- continuous-performance-monitoring
- index-lifecycle-management
- profiling
- capacity-planning
- observability-and-monitoring
layout: problem
lang: de
en_slug: index-fragmentation
---

## Description
Index-Fragmentierung ist ein Problem, das über die Zeit in Datenbankindizes auftritt, während Daten eingefügt, aktualisiert und gelöscht werden. Dies kann zu einer Performance-Verschlechterung führen, da die Datenbank mehr Arbeit leisten muss, um den Index zu lesen. Es gibt zwei Arten von Fragmentierung: interne Fragmentierung, bei der es verschwendeten Platz innerhalb der Index-Seiten gibt, und externe Fragmentierung, bei der die logische Reihenfolge der Index-Seiten nicht der physischen Reihenfolge auf der Festplatte entspricht. Regelmäßiges Neuaufbauen oder Reorganisieren von Indizes kann helfen, Fragmentierung zu reduzieren und die Performance zu verbessern.

## Indicators ⟡
- Abfragen, die früher schnell waren, sind jetzt langsam.
- Die Datenbank nutzt mehr Festplattenspeicher als erwartet.
- Die Datenbank führt viele I/O-Operationen durch.
- Die Datenbank nutzt viel CPU.

## Symptoms ▲

- [Langsame Datenbankabfragen](langsame-datenbankabfragen.md)
<br/>  Fragmentierte Indizes erfordern mehr I/O-Operationen zum Durchqueren, was direkt dazu führt, dass Abfragen langsamer ausgeführt werden.
- [Performance-Probleme bei Datenbankabfragen](performance-probleme-bei-datenbankabfragen.md)
<br/>  Index-Fragmentierung verschlechtert die Effizienz des Abfrageplans, was zu einer Verschlechterung der gesamten Datenbankabfrageperformance führt.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Langsamere Datenbankabfragen aufgrund fragmentierter Indizes übersetzen sich in langsamere Antwortzeiten auf Anwendungsebene.
- [Hohe Datenbank-Ressourcennutzung](hohe-datenbank-ressourcennutzung.md)
<br/>  Fragmentierte Indizes verursachen, dass die Datenbank mehr CPU, Speicher und Festplatten-I/O verbraucht als nötig.

## Causes ▼
- [Unbegrenztes Datenwachstum](unbegrenztes-datenwachstum.md)
<br/>  Index-Fragmentierung wird primär durch laufende Datenmodifikationen (Einfügungen, Aktualisierungen, Löschungen) verursacht.

## Detection Methods ○

- **Datenbankmanagementsystem (DBMS)-Werkzeuge:** Die meisten DBMS bieten Werkzeuge oder Views zur Überprüfung von Index-Fragmentierungsniveaus (z. B. `sys.dm_db_index_physical_stats` in SQL Server, `pg_stat_user_tables` und `pg_stat_user_indexes` in PostgreSQL, `ANALYZE TABLE` in MySQL).
- **Performance-Monitoring:** Überwachung von Festplatten-I/O, Abfrageausführungszeiten und Cache-Trefferraten auf Anzeichen von Verschlechterung.
- **Regelmäßige Audits:** Periodische Überprüfung der Index-Gesundheit und Performance-Metriken.

## Examples
Die `posts`-Tabelle eines Online-Forums nutzt einen geclusterten Index auf `post_id` (einer automatisch inkrementierenden Ganzzahl). Nutzer können jedoch ihre Beiträge bearbeiten, und diese Aktualisierungen lassen die Datenzeilen wachsen. Über die Zeit wird der Index fragmentiert, was zu langsamerem Abruf von Beiträgen führt, besonders beim Durchsuchen älterer Threads. In einem anderen Fall fügt eine Finanzanwendung häufig neue Transaktionen mit einer `transaction_id` ein, die eine UUID ist. Weil UUIDs nicht sequenziell sind, verursachen neue Einfügungen zufällige Seitenaufteilungen im Primärschlüssel-Index, was zu hoher Fragmentierung und verschlechterter Einfügeperformance über die Zeit führt. Index-Fragmentierung ist ein verbreitetes Problem in Datenbanken mit hoher Schreiblast. Während moderne Datenbanksysteme besser darin sind, Fragmentierung zu verwalten, ist regelmäßige Wartung (Neuaufbau oder Reorganisation von Indizes) oft nötig, um optimale Performance zu erhalten.
