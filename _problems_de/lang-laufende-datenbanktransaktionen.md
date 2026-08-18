---
title: Lang laufende Datenbanktransaktionen
description: Datenbanktransaktionen bleiben über längere Zeiträume offen, halten
  Locks und verbrauchen Ressourcen, was andere Operationen blockieren kann.
category:
- Code
- Performance
related_problems:
- slug: long-running-transactions
  similarity: 0.95
- slug: slow-database-queries
  similarity: 0.7
- slug: high-database-resource-utilization
  similarity: 0.7
- slug: high-number-of-database-queries
  similarity: 0.65
- slug: high-connection-count
  similarity: 0.65
- slug: lazy-loading
  similarity: 0.65
solutions:
- evolutionary-database-design
- query-optimization-process
- transactions
- write-ahead-logging
- concurrency-control
- monitoring
- profiling
- batch-processing
- performance-measurements
- index-lifecycle-management
layout: problem
lang: de
en_slug: long-running-database-transactions
---

## Description
Lang laufende Datenbanktransaktionen sind eine spezifische Art lang laufender Transaktionen, die auf Datenbankebene auftritt. Diese Transaktionen können besonders problematisch sein, da sie Locks auf Datenbankressourcen über einen längeren Zeitraum halten können, was andere Abfragen an der Ausführung hindert und potenziell zu Deadlocks führt. Sie werden oft durch ineffiziente Abfragen, fehlende ordentliche Indizierung oder Anwendungslogik verursacht, die Transaktionen offen hält, während andere Aufgaben ausgeführt werden. Die Minimierung der Dauer von Datenbanktransaktionen ist ein Schlüsselprinzip guten Datenbankdesigns.

## Indicators ⟡
- Die Datenbank ist langsam, auch wenn es keine offensichtlichen Anzeichen hoher CPU- oder Speichernutzung gibt.
- Sie sehen eine hohe Anzahl von Deadlocks in Ihren Datenbank-Logs.
- Sie erhalten Beschwerden von Nutzern über langsame Performance.
- Sie sehen eine hohe Anzahl von Timeout-Fehlern in Ihren Logs.

## Symptoms ▲

- [Lock Contention](lock-contention.md)
<br/>  Lang gehaltene Datenbank-Locks blockieren andere Abfragen, die versuchen, auf dieselben Zeilen oder Tabellen zuzugreifen, was Konkurrenz erzeugt.
- [Deadlock-Zustände](deadlock-zustaende.md)
<br/>  Transaktionen, die Locks über längere Zeiträume halten, erhöhen das Zeitfenster für die Bildung zirkulärer Lock-Abhängigkeiten.
- [Langsame Datenbankabfragen](langsame-datenbankabfragen.md)
<br/>  Andere Abfragen werden gezwungen, auf Locks zu warten, die von lang laufenden Transaktionen gehalten werden, was ihre Ausführungszeit erhöht.
- [Hohe Datenbank-Ressourcennutzung](hohe-datenbank-ressourcennutzung.md)
<br/>  Lang laufende Transaktionen verbrauchen Verbindungsslots, Speicher und Transaktionslog-Speicherplatz über längere Zeiträume.
- [Service-Timeouts](service-timeouts.md)
<br/>  Anwendungsanfragen, die auf durch lang laufende Transaktionen blockierte Datenbankoperationen warten, überschreiten Timeout-Schwellenwerte.
- [Datenbankverbindungslecks](datenbankverbindungslecks.md)
<br/>  Lang laufende Datenbanktransaktionen binden Verbindungen über längere Zeiträume, und verlassene Transaktionen können Verbindungen vollständig verlieren.

## Causes ▼

- [Ineffiziente Datenbankindizierung](ineffiziente-datenbankindizierung.md)
<br/>  Fehlende oder schlechte Indizes verursachen, dass Abfragen innerhalb von Transaktionen viel länger dauern, was die Transaktionsdauer verlängert.
- [Verzögerungen durch externe Services](verzoegerungen-durch-externe-services.md)
<br/>  Das Aufrufen externer Services, während eine Datenbanktransaktion offen ist, bedeutet, dass die Transaktion auf langsame externe Antworten wartet.
- [Probleme im Datenbankschema-Design](probleme-im-datenbankschema-design.md)
<br/>  Schlechtes Schema-Design kann zu übermäßigem Sperrbereich führen oder komplexe Mehrtabellen-Operationen erfordern, die die Transaktionsdauer verlängern.

## Detection Methods ○

- **Datenbank-Monitoring-Werkzeuge:** Nutzung datenbankspezifischer Werkzeuge (z. B. `pg_stat_activity` in PostgreSQL, `SHOW PROCESSLIST` in MySQL, `sys.dm_tran_active_transactions` in SQL Server) zur Identifikation aktiver Transaktionen, ihrer Dauer und worauf sie warten.
- **Transaktionslog-Überwachung:** Überwachung der Größe und Wachstumsrate der Datenbank-Transaktionslogs.
- **Anwendungs-Logging:** Hinzufügen von Logging zur Anwendung, um Start- und Endzeiten von Datenbanktransaktionen nachzuverfolgen.
- **Alerting:** Einrichtung von Alarmen für Transaktionen, die eine bestimmte Dauer überschreiten.

## Examples
Eine E-Commerce-Anwendung verarbeitet eine Bestellung. Sie startet eine Datenbanktransaktion, aktualisiert den Bestand und ruft dann ein Drittanbieter-Zahlungsgateway auf. Wenn das Zahlungsgateway langsam ist, bleibt die Datenbanktransaktion offen und hält einen Lock auf der Bestandstabelle. Dies blockiert andere Nutzer daran, Bestellungen für dasselbe Produkt aufzugeben. In einem anderen Fall umschließt ein Batch-Job, der Millionen von Datensätzen in eine Datenbank importiert, den gesamten Import in einer einzigen Transaktion. Wenn der Import auf halbem Weg fehlschlägt, wird die Transaktion zurückgerollt, aber das Rollback selbst dauert Stunden, während dieser Zeit die Datenbank stark beeinträchtigt ist. Dieses Problem ist besonders kritisch in Systemen mit hoher Nebenläufigkeit, wo selbst kurzlebige Locks erhebliche Auswirkungen haben können. Es erfordert oft sorgfältiges Design von Transaktionsgrenzen und asynchrone Verarbeitung für lang laufende Aufgaben.
