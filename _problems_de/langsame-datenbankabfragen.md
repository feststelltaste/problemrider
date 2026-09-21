---
title: Langsame Datenbankabfragen
description: Die Anwendungsperformance verschlechtert sich aufgrund ineffizienten
  Datenabrufs aus der Datenbank.
category:
- Performance
related_problems:
- slug: high-database-resource-utilization
  similarity: 0.8
- slug: inefficient-database-indexing
  similarity: 0.8
- slug: slow-application-performance
  similarity: 0.8
- slug: high-number-of-database-queries
  similarity: 0.8
- slug: excessive-disk-io
  similarity: 0.75
- slug: n-plus-one-query-problem
  similarity: 0.75
solutions:
- query-optimization-process
- approximation-methods
- cqrs
- data-aggregation
- data-archiving
- data-partitioning
- data-replication
- datensparsamkeit
- denormalization
- distributed-caching
- graph-databases
- in-memory-processing
- mass-test-data-generation
- materialized-views
- nosql-databases
- pagination
- parallelization
- probabilistic-data-structures
- read-replicas
- sampling
- vertical-scaling
- index-lifecycle-management
- typed-schema-extraction
- attribute-usage-analysis
layout: problem
lang: de
en_slug: slow-database-queries
---

## Description
Langsame Datenbankabfragen sind eine primäre Ursache für schlechte Anwendungsperformance. Wenn eine Abfrage zu lange zur Ausführung braucht, kann sie Anwendungs-Threads blockieren, andere Abfragen aufhalten und zu einer frustrierenden Nutzererfahrung führen. Diese langsamen Abfragen sind oft das Ergebnis ineffizienten Abfragedesigns, fehlender oder unsachgemäßer Indizes, oder eines Datenbankschemas, das nicht für die Arten der ausgeführten Abfragen optimiert ist. Die Identifikation und Optimierung langsamer Abfragen ist eine kritische Aufgabe für eine gesunde und performante Anwendung.

## Indicators ⟡
- Die Anwendung ist langsam, und Sie vermuten, dass dies an langsamen Datenbankabfragen liegt.
- Sie sehen eine hohe Anzahl langsamer Abfragen in Ihren Datenbank-Logs.
- Die Datenbank nutzt viel CPU oder Speicher.
- Sie erhalten Beschwerden von Nutzern über langsame Performance.

## Symptoms ▲

- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Langsame Abfragen verursachen direkt, dass die Anwendung langsam auf Nutzeranfragen antwortet.
- [Langsame Antwortzeiten für Listen](langsame-antwortzeiten-fuer-listen.md)
<br/>  Listenseiten sind besonders von langsamen Abfragen betroffen, weil sie mehrere Abfragen ausführen oder große Ergebnismengen verarbeiten.
- [Hohe API-Latenz](hohe-api-latenz.md)
<br/>  API-Endpunkte, die von Datenbankabfragen abhängen, erben die Langsamkeit, was die Gesamt-API-Antwortzeiten erhöht.

## Causes ▼

- [Ineffiziente Datenbankindizierung](ineffiziente-datenbankindizierung.md)
<br/>  Ohne ordentliche Indizes führt die Datenbank vollständige Tabellenscans durch, was Abfragen dramatisch verlangsamt.
- [Hohe Anzahl an Datenbankabfragen](hohe-anzahl-an-datenbankabfragen.md)
<br/>  N+1-Abfragemuster und exzessive Abfrageanzahlen verstärken sich zu bedeutenden Performance-Problemen.
- [Imperative Datenabruflogik](imperative-datenabruflogik.md)
<br/>  Manuell konstruierte Datenabruflogik produziert oft ineffiziente Abfragemuster statt optimierte Datenbankoperationen zu nutzen.
- [Lazy Loading](lazy-loading.md)
<br/>  Lazy Loading löst zusätzliche Datenbankabfragen bei Bedarf aus, was zu unvorhersehbarer und oft exzessiver Abfrageausführung führt.

## Detection Methods ○

- **Datenbankabfrage-Logging:** Aktivierung des Loggings langsamer Abfragen in der Datenbankkonfiguration.
- **Application Performance Monitoring (APM) Werkzeuge:** Nutzung von Werkzeugen wie New Relic, Datadog oder Prometheus zur Überwachung der Abfrageperformance und Identifikation von Engpässen.
- **Datenbankspezifische Werkzeuge:** Nutzung von Werkzeugen wie `EXPLAIN` in PostgreSQL oder `EXPLAIN PLAN` in Oracle zur Analyse von Abfrageausführungsplänen.
- **Code-Reviews:** Suche nach häufigen Antipatterns wie N+1-Abfragen oder ineffizienter Abfragelogik.
- **Lasttests:** Simulation hohen Traffics zur Identifikation von Abfragen, die nicht gut skalieren.

## Examples
Die Profilseite eines Nutzers in einer Webanwendung braucht lange zum Laden. Bei der Untersuchung wird entdeckt, dass die Seite eine separate Datenbankabfrage für jeden Freund des Nutzers ausführt, um dessen Profilbilder abzurufen. In einem anderen Fall läuft ein Berichts-Dashboard, das Daten aus mehreren Tabellen aggregiert, in Timeout, weil die Abfragen nicht die korrekten Indizes nutzen. Dieses Problem ist häufig in Anwendungen mit großen Datenmengen oder komplexen Datenmodellen. Es wird oft durch mangelnde Datenbank-Expertise im Entwicklungsteam verschärft.
