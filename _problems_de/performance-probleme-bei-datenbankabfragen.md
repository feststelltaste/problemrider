---
title: Performance-Probleme bei Datenbankabfragen
description: Schlecht optimierte Datenbankabfragen verursachen langsame Antwortzeiten,
  hohen Ressourcenverbrauch und Skalierungsprobleme.
category:
- Architecture
- Code
- Performance
related_problems:
- slug: high-number-of-database-queries
  similarity: 0.8
- slug: slow-database-queries
  similarity: 0.75
- slug: inefficient-database-indexing
  similarity: 0.7
- slug: n-plus-one-query-problem
  similarity: 0.7
- slug: database-schema-design-problems
  similarity: 0.7
- slug: algorithmic-complexity-problems
  similarity: 0.7
solutions:
- query-optimization-process
- cqrs
- data-aggregation
- denormalization
- graph-databases
- mass-test-data-generation
- materialized-views
- object-relational-mapping-orm
- read-replicas
layout: problem
lang: de
en_slug: database-query-performance-issues
---

## Description

Performance-Probleme bei Datenbankabfragen entstehen, wenn SQL-Abfragen ineffizient geschrieben, schlecht optimiert sind oder gegen unzureichend strukturierte Datenbanken ausgeführt werden, was zu langsamen Antwortzeiten, hoher CPU- und Speichernutzung und Skalierungsengpässen führt. Diese Probleme werden oft ausgeprägter, während Datenvolumen wachsen und Nutzerlasten zunehmen.

## Indicators ⟡

- Datenbankabfragen brauchen erheblich länger als erwartet zur Ausführung
- Hohe CPU-Nutzung auf Datenbankservern während der Abfrageausführung
- Anwendungen erreichen ein Timeout, während sie auf Datenbankantworten warten
- Datenbank-Connection-Pools sind aufgrund langsamer Abfragen erschöpft
- Abfrageausführungspläne zeigen vollständige Tabellenscans oder ineffiziente Operationen

## Symptoms ▲

- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Ineffiziente Abfragen verursachen direkt langsame Reaktionen nutzerseitiger Features, während sie auf Datenbankergebnisse warten.
- [Hohe Datenbank-Ressourcennutzung](hohe-datenbank-ressourcennutzung.md)
<br/>  Schlecht optimierte Abfragen verbrauchen übermäßig viel CPU und Speicher auf dem Datenbankserver, was die Ressourcenauslastung auf gefährliche Niveaus treibt.
- [Hohe Anzahl an Verbindungen](hohe-anzahl-an-verbindungen.md)
<br/>  Langsame Abfragen halten Verbindungen länger als nötig offen, was Druck auf den Connection-Pool und hohe Anzahlen aktiver Verbindungen verursacht.
- [Negatives Nutzerfeedback](negatives-nutzerfeedback.md)
<br/>  Nutzer erleben langsame Seitenladezeiten und Timeouts, die durch Datenbank-Performance-Probleme verursacht werden, was zu Beschwerden und negativen Bewertungen führt.
- [Ineffizienzen bei der Skalierung](ineffizienzen-bei-der-skalierung.md)
<br/>  Abfragen, die vollständige Tabellenscans durchführen oder keine ordentliche Indizierung haben, werden exponentiell langsamer, während Datenvolumen wachsen, was effektive Skalierung verhindert.

## Causes ▼

- [Ineffiziente Datenbankindizierung](ineffiziente-datenbankindizierung.md)
<br/>  Fehlende oder schlecht entworfene Indizes zwingen die Datenbank dazu, vollständige Tabellenscans statt effizienter Index-Lookups durchzuführen.
- [Probleme im Datenbankschema-Design](probleme-im-datenbankschema-design.md)
<br/>  Schlechtes Schema-Design zwingt Abfragen dazu, komplexe Multi-Table-Joins durchzuführen und unnötig breite Zeilen zu scannen, was die Performance verschlechtert.
- [N+1-Abfrageproblem](n-plus-1-abfrageproblem.md)
<br/>  Anwendungscode, der verwandte Daten in Schleifen abruft, erzeugt viele einzelne Abfragen statt effizienter Batch-Operationen.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler ohne Kenntnisse zur Datenbankoptimierung schreiben naive Abfragen, die für kleine Datensätze funktionieren, aber im Produktionsumfang versagen.

## Detection Methods ○

- **Abfrage-Performance-Monitoring:** Überwachung von Ausführungszeiten und Ressourcennutzung von Datenbankabfragen
- **Abfrageausführungsplan-Analyse:** Analyse von Abfrageausführungsplänen auf ineffiziente Operationen
- **Datenbank-Performance-Profiling:** Profiling der Datenbankperformance unter unterschiedlichen Lastbedingungen
- **Slow-Query-Log-Analyse:** Überprüfung von Datenbank-Slow-Query-Protokollen auf problematische Abfragen
- **Index-Nutzungsanalyse:** Analyse, welche Indizes genutzt werden und welchen Abfragen ordentliche Indizierung fehlt

## Examples

Die Produktsuchabfrage einer E-Commerce-Anwendung führt einen vollständigen Tabellenscan über eine Produkttabelle mit 10 Millionen Datensätzen durch, weil sie Produktbeschreibungen mit einer LIKE-Klausel ohne ordentliche Textindizierung durchsucht. Jede Suche dauert 15 Sekunden und verbraucht erhebliche Datenbankressourcen, was das Suchfeature bei Spitzenverkehr unbrauchbar macht. Das Hinzufügen eines Volltextindex und die Umstrukturierung der Abfrage reduziert die Suchzeit auf unter 100 ms. Ein weiteres Beispiel betrifft eine Reporting-Abfrage, die fünf große Tabellen ohne ordentliche Indizes auf den Join-Spalten verknüpft. Die Abfrage benötigt 45 Minuten zur Ausführung und sperrt Datenbankressourcen, was andere Operationen an der Fertigstellung hindert. Die Analyse zeigt, dass die Abfrage Nested-Loop-Joins statt effizienterer Hash-Joins durchführt, aufgrund fehlender Indizes auf Fremdschlüsselspalten.
