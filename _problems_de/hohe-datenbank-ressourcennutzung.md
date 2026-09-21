---
title: Hohe Datenbank-Ressourcennutzung
description: Der Datenbankserver arbeitet durchgängig mit hoher CPU- oder Speichernutzung,
  was Instabilität riskiert und alle abhängigen Dienste verlangsamt.
category:
- Code
- Performance
related_problems:
- slug: slow-database-queries
  similarity: 0.8
- slug: high-connection-count
  similarity: 0.75
- slug: excessive-disk-io
  similarity: 0.75
- slug: high-number-of-database-queries
  similarity: 0.7
- slug: misconfigured-connection-pools
  similarity: 0.7
- slug: high-api-latency
  similarity: 0.7
solutions:
- query-optimization-process
- approximation-methods
- batch-processing
- connection-pooling
- cqrs
- data-aggregation
- data-archiving
- data-deduplication
- data-partitioning
- data-replication
- datensparsamkeit
- distributed-caching
- elastic-resource-utilization
- in-memory-processing
- materialized-views
- monitoring-system-utilization
- nosql-databases
- probabilistic-data-structures
- read-replicas
- sampling
- vertical-scaling
layout: problem
lang: de
en_slug: high-database-resource-utilization
---

## Description
Hohe Datenbank-Ressourcennutzung kann eine Hauptursache für schlechte Anwendungsperformance und Instabilität sein. Dies kann durch verschiedene Faktoren verursacht werden, von ineffizienten Abfragen und fehlender ordentlicher Indizierung bis zu einer hohen Anzahl an Verbindungen und lang laufenden Transaktionen. Wenn die Datenbank unter Belastung steht, kann dies zu einer Performance-Verschlechterung, Timeouts und sogar einem vollständigen Systemausfall führen. Ein robustes Monitoring- und Alerting-System ist essenziell, um hohe Datenbank-Ressourcennutzung zeitnah zu erkennen und darauf zu reagieren.

## Indicators ⟡
- Der Datenbankserver läuft durchgängig mit hoher CPU- oder Speichernutzung.
- Es zeigt sich eine hohe Anzahl langsamer Abfragen in den Datenbank-Logs.
- Die Anwendung ist langsam, und der Verdacht besteht, dass dies an einer hohen Anzahl an Datenbankverbindungen liegt.
- Es kommen Beschwerden von Nutzern über langsame Performance.

## Symptoms ▲

- [Langsame Datenbankabfragen](langsame-datenbankabfragen.md)
<br/>  Wenn die Datenbank unter hoher Ressourcenlast steht, steigen die Abfrageausführungszeiten erheblich, da CPU- und Speicherkonkurrenz die Verarbeitung verzögert.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Hohe Datenbank-Ressourcennutzung verschlechtert direkt die Antwortzeiten der Anwendung, da die meisten Operationen von Datenbankinteraktionen abhängen.
- [Hohe API-Latenz](hohe-api-latenz.md)
<br/>  API-Endpunkte, die von Datenbankabfragen abhängen, erleben erhöhte Latenz, wenn der Datenbankserver ressourcenbeschränkt ist.
- [Systemausfälle](systemausfaelle.md)
<br/>  Datenbankinstabilität durch anhaltend hohe Ressourcennutzung kann zu Abstürzen und vollständigen Serviceausfällen führen.
- [Ressourcenkonkurrenz](ressourcenkonkurrenz.md)
<br/>  Hohe Datenbank-Ressourcennutzung schafft Konkurrenz, bei der mehrere Abfragen um begrenzte CPU- und Speicherressourcen konkurrieren.

## Causes ▼

- [Hohe Anzahl an Datenbankabfragen](hohe-anzahl-an-datenbankabfragen.md)
<br/>  Ein hohes Volumen an Abfragen pro Anfrage vervielfacht die Last auf CPU- und Speicherressourcen der Datenbank.
- [Ineffiziente Datenbankindizierung](ineffiziente-datenbankindizierung.md)
<br/>  Fehlende oder schlecht gestaltete Indizes zwingen die Datenbank zu vollständigen Tabellen-Scans, was übermäßig CPU und I/O verbraucht.
- [Hohe Anzahl an Verbindungen](hohe-anzahl-an-verbindungen.md)
<br/>  Zu viele offene Datenbankverbindungen verbrauchen Speicher- und CPU-Ressourcen auf dem Datenbankserver.
- [Performance-Probleme bei Datenbankabfragen](performance-probleme-bei-datenbankabfragen.md)
<br/>  Schlecht optimierte Abfragen verbrauchen übermäßig CPU und I/O auf dem Datenbankserver, was direkt zu hoher Ressourcennutzung beiträgt.
- [Probleme mit algorithmischer Komplexität](probleme-mit-algorithmischer-komplexitaet.md)
<br/>  Wenn ineffiziente Algorithmen innerhalb der Datenbankschicht implementiert werden (komplexe Stored Procedures, schlecht strukturierte Abfragen oder zeilenweise Verarbeitung) statt im Anwendungscode, verbrauchen sie übermäßig Datenbank-CPU und -Speicher.

## Detection Methods ○

- **Datenbank-Monitoring-Werkzeuge:** Nutzung spezialisierter Datenbank-Monitoring-Werkzeuge (z. B. pgAdmin für PostgreSQL, MySQL Workbench oder Drittanbieter-Werkzeuge wie Percona Monitoring and Management), um Ressourcennutzung, laufende Abfragen und Konfiguration zu untersuchen.
- **Cloud-Provider-Metriken:** Bei Nutzung eines verwalteten Datenbankdienstes (wie AWS RDS oder Google Cloud SQL), Nutzung der Monitoring-Dashboards des Cloud-Anbieters zur Nachverfolgung von CPU-, Speicher- und I/O-Metriken.
- **Abfrageanalyse:** Nutzung der `EXPLAIN`- oder `EXPLAIN ANALYZE`-Befehle der Datenbank zur Untersuchung der Ausführungspläne langsamer oder häufiger Abfragen und Identifikation von Ineffizienzen.
- **System-Performance-Werkzeuge:** Nutzung standardmäßiger Linux/Windows-Kommandozeilenwerkzeuge (`top`, `htop`, `iostat`, `vmstat`) auf dem Datenbankserver, um einen Echtzeitüberblick über den Ressourcenverbrauch zu erhalten.

## Examples
Die Hauptanwendung eines Unternehmens wird jeden Tag um die Mittagszeit langsam. Eine Untersuchung zeigt, dass ein täglicher Bericht, der eine Reihe komplexer, nicht optimierter Abfragen ausführt, zu dieser Zeit startet und die gesamte verfügbare Datenbank-CPU verbraucht. In einem anderen Fall ist eine Webanwendung, die einen Connection Pool nutzt, falsch konfiguriert, sodass sie weit mehr Verbindungen öffnet, als die Datenbank ausgelegt ist. Über die Zeit steigt die Speichernutzung der Datenbank, bis sie instabil wird, obwohl die Abfragelast selbst nicht besonders hoch ist. Dies ist ein kritisches Problem in Legacy-Systemen, in denen die Datenbank seit vielen Jahren im Einsatz ist. Über die Zeit wächst das Datenvolumen, ändern sich Abfragemuster, und Indizes, die einst effektiv waren, sind möglicherweise nicht mehr optimal, was zu einem schrittweisen Anstieg der Ressourcennutzung führt.
