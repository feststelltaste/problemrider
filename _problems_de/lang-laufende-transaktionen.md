---
title: Lang laufende Transaktionen
description: Datenbanktransaktionen, die über lange Zeit offen bleiben, können Locks
  halten, Ressourcen verbrauchen und andere Operationen blockieren.
category:
- Code
- Database
- Performance
related_problems:
- slug: long-running-database-transactions
  similarity: 0.95
- slug: slow-database-queries
  similarity: 0.7
- slug: high-database-resource-utilization
  similarity: 0.65
- slug: high-connection-count
  similarity: 0.65
- slug: high-number-of-database-queries
  similarity: 0.65
- slug: n-plus-one-query-problem
  similarity: 0.65
solutions:
- concurrency-control
- profiling
- resource-pooling
- resource-usage-optimization
- transactions
- saga-pattern
- monitoring
- index-lifecycle-management
- query-optimization-process
- batch-processing
- performance-measurements
layout: problem
lang: de
en_slug: long-running-transactions
---

## Description
Lang laufende Transaktionen sind Datenbanktransaktionen, die über einen längeren Zeitraum offen bleiben. Dies kann durch eine Vielzahl von Faktoren verursacht werden, von ineffizienten Abfragen und fehlender ordentlicher Indizierung bis hin zu Anwendungslogik, die Transaktionen offen hält, während andere Aufgaben ausgeführt werden. Lang laufende Transaktionen können eine Reihe von Problemen verursachen, einschließlich des Haltens von Locks auf Datenbankressourcen, der Verhinderung der Ausführung anderer Abfragen und der Erhöhung des Deadlock-Risikos. Sie sind eine häufige Quelle für Performance- und Stabilitätsprobleme in datenbankgetriebenen Anwendungen.

## Indicators ⟡
- Die Datenbank ist langsam, auch wenn es keine offensichtlichen Anzeichen hoher CPU- oder Speichernutzung gibt.
- Sie sehen eine hohe Anzahl von Deadlocks in Ihren Datenbank-Logs.
- Sie erhalten Beschwerden von Nutzern über langsame Performance.
- Sie sehen eine hohe Anzahl von Timeout-Fehlern in Ihren Logs.

## Symptoms ▲

- [Lock Contention](lock-contention.md)
<br/>  Transaktionen, die Locks über längere Zeiträume halten, verursachen, dass andere Operationen blockieren und auf die Freigabe dieser Locks warten.
- [Deadlock-Zustände](deadlock-zustaende.md)
<br/>  Je länger Transaktionen Locks halten, desto größer ist die Wahrscheinlichkeit, dass sich zirkuläre Abhängigkeiten zwischen gleichzeitigen Transaktionen bilden.
- [Hohe Datenbank-Ressourcennutzung](hohe-datenbank-ressourcennutzung.md)
<br/>  Lang laufende Transaktionen verbrauchen Datenbankverbindungen, Speicher und Transaktionslog-Speicherplatz über längere Zeiträume.
- [Datenbankverbindungslecks](datenbankverbindungslecks.md)
<br/>  Transaktionen, die über längere Zeiträume laufen, binden Ressourcen des Connection Pools, und verlassene Transaktionen können Verbindungen vollständig verlieren.
- [Service-Timeouts](service-timeouts.md)
<br/>  Operationen, die durch Locks lang laufender Transaktionen blockiert sind, können Anwendungs-Timeout-Schwellenwerte überschreiten, was zu Fehlern führt.

## Causes ▼

- [Langsame Datenbankabfragen](langsame-datenbankabfragen.md)
<br/>  Langsame Abfragen innerhalb einer Transaktion verlängern direkt ihre Dauer und halten Locks länger.
- [Ineffiziente Datenbankindizierung](ineffiziente-datenbankindizierung.md)
<br/>  Fehlende Indizes verursachen, dass Abfragen vollständige Tabellenscans innerhalb von Transaktionen durchführen, was die Transaktionsdauer erheblich verlängert.
- [Verzögerungen durch externe Services](verzoegerungen-durch-externe-services.md)
<br/>  Anwendungslogik, die langsame externe Services aufruft, während eine offene Transaktion gehalten wird, verlängert deren Lebensdauer.
- [Falsche maximale Connection-Pool-Größe](falsche-maximale-connection-pool-groesse.md)
<br/>  Unterdimensionierte Connection Pools unter Last können dazu führen, dass Transaktionen sich anstellen, und wenn sie schließlich ausgeführt werden, umfasst ihre effektive Dauer die Wartezeit.

## Detection Methods ○

- **Datenbank-Monitoring-Werkzeuge:** Nutzung datenbankspezifischer Befehle (z. B. `pg_stat_activity` in PostgreSQL, `SHOW PROCESSLIST` in MySQL) zur Identifikation aktiver Transaktionen und ihrer Dauer.
- **Transaktionslog-Überwachung:** Überwachung der Größe und Wachstumsrate von Transaktionslogs.
- **Lock-Überwachung:** Nutzung von Datenbankwerkzeugen zur Identifikation aktuell gehaltener Locks und welche Transaktionen sie halten.
- **Anwendungs-Logging:** Hinzufügen von Logging zur Anwendung, um Start- und Endzeiten von Transaktionen nachzuverfolgen.

## Examples
Eine E-Commerce-Anwendung hat einen Checkout-Prozess, der zu Beginn des Prozesses eine Datenbanktransaktion startet. Wenn der Nutzer den Checkout auf halbem Weg abbricht, bleibt die Transaktion offen, bis die Sitzung abläuft, hält Locks auf Bestandstabellen und verhindert, dass andere Nutzer diese Artikel kaufen. In einem anderen Fall umschließt ein nächtlicher Batch-Job zur Datensynchronisation seine gesamte Operation in einer einzigen Transaktion. Wenn der Job Millionen von Datensätzen verarbeitet, kann diese einzelne Transaktion Stunden laufen, erhebliche Ressourcen verbrauchen und potenziell andere Datenbankoperationen blockieren. Dieses Problem ist oft das Ergebnis unzureichenden Verständnisses der Datenbank-Transaktionssemantik oder schlechten Anwendungsdesigns. Es kann zu schweren Performance-Engpässen und Datenkonsistenzproblemen führen, insbesondere in Umgebungen mit hoher Nebenläufigkeit.
