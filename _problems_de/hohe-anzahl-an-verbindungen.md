---
title: Hohe Anzahl an Verbindungen
description: Eine große Anzahl offener Datenbankverbindungen, selbst wenn diese inaktiv
  sind, kann erhebliche Speicherressourcen verbrauchen und zu Verbindungsablehnungen
  führen.
category:
- Code
- Performance
related_problems:
- slug: incorrect-max-connection-pool-size
  similarity: 0.8
- slug: misconfigured-connection-pools
  similarity: 0.8
- slug: high-database-resource-utilization
  similarity: 0.75
- slug: high-number-of-database-queries
  similarity: 0.75
- slug: database-connection-leaks
  similarity: 0.7
- slug: slow-database-queries
  similarity: 0.7
solutions:
- backpressure
- capacity-planning
- elastic-scaling
- resource-pooling
- resource-usage-optimization
- connection-pooling
- reactive-programming
- timeout-management
layout: problem
lang: de
en_slug: high-connection-count
---

## Description
Eine hohe Anzahl an Verbindungen tritt auf, wenn eine Datenbank von einer großen Anzahl offener Verbindungen überwältigt wird, sowohl aktiv als auch inaktiv. Jede Verbindung verbraucht Speicher und andere Ressourcen auf dem Datenbankserver, und das Überschreiten des konfigurierten Limits kann zu Verbindungsablehnungen und Anwendungsausfällen führen. Dieses Problem ist oft ein Symptom von falsch konfiguriertem Connection Pooling, ineffizientem Anwendungscode, der Verbindungen nicht freigibt, oder plötzlichen Traffic-Spitzen. Ordentliches Verbindungsmanagement ist entscheidend für die Aufrechterhaltung der Stabilität und Performance jeder datenbankgetriebenen Anwendung.

## Indicators ⟡
- Es zeigt sich eine hohe Anzahl an Verbindungen in den Datenbank-Monitoring-Werkzeugen.
- Die Anwendung ist langsam, und der Verdacht besteht, dass dies an einer hohen Anzahl an Datenbankverbindungen liegt.
- Es kommen Beschwerden von Nutzern über langsame Performance.
- Es zeigt sich eine hohe Anzahl an Timeout-Fehlern in den Logs.

## Symptoms ▲

- [Hohe Datenbank-Ressourcennutzung](hohe-datenbank-ressourcennutzung.md)
<br/>  Jede offene Verbindung verbraucht Speicher und CPU auf dem Datenbankserver, was die Gesamtressourcennutzung in die Höhe treibt.
- [Service-Timeouts](service-timeouts.md)
<br/>  Wenn das Verbindungslimit erreicht ist, werden neue Verbindungsversuche abgelehnt oder in eine Warteschlange gestellt, was Service-Timeouts verursacht.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Ressourcenkonkurrenz durch zu viele Verbindungen verschlechtert die Antwortzeiten der Datenbank und verlangsamt die gesamte Anwendung.
- [Erhöhte Fehlerraten](erhoehte-fehlerraten.md)
<br/>  Verbindungsablehnungen, wenn Limits erreicht werden, verursachen Anwendungsfehler und fehlgeschlagene Anfragen.
- [Kaskadierende Ausfälle](kaskadierende-ausfaelle.md)
<br/>  Die Erschöpfung von Datenbankverbindungen verursacht Ausfälle, die zu allen von dieser Datenbank abhängigen Diensten kaskadieren.

## Causes ▼

- [Falsch konfigurierte Connection Pools](falsch-konfigurierte-connection-pools.md)
<br/>  Unangemessen konfigurierte Connection-Pool-Einstellungen erlauben es, dass zu viele Verbindungen erzeugt oder inaktiv gehalten werden.
- [Datenbankverbindungslecks](datenbankverbindungslecks.md)
<br/>  Verbindungen, die geöffnet, aber nie ordentlich geschlossen werden, häufen sich über die Zeit an und erhöhen die Verbindungsanzahl stetig.
- [Falsche maximale Connection-Pool-Größe](falsche-maximale-connection-pool-groesse.md)
<br/>  Das Setzen der maximalen Pool-Größe zu hoch erlaubt es jeder Anwendungsinstanz, mehr Verbindungen zu halten, als die Datenbank effizient handhaben kann.
- [Fehler bei der Ressourcenzuweisung](fehler-bei-der-ressourcenzuweisung.md)
<br/>  Code, der Datenbankverbindungen nach Nutzung nicht ordentlich freigibt, verursacht, dass sich Verbindungen anhäufen, ohne an den Pool zurückgegeben zu werden.

## Detection Methods ○

- **Datenbank-Monitoring-Werkzeuge:** Nutzung datenbankspezifischer Werkzeuge (z. B. `SHOW STATUS` in MySQL, `pg_stat_activity` in PostgreSQL) zur Überwachung der Anzahl aktiver und inaktiver Verbindungen.
- **Anwendungsmetriken:** Überwachung von Connection-Pool-Metriken innerhalb der Anwendung (z. B. aktive Verbindungen, inaktive Verbindungen, Wartezeiten).
- **System-Monitoring:** Beobachtung der Speichernutzung und Prozessanzahl des Datenbankservers.
- **Log-Analyse:** Suche nach Datenbank-Fehlerlogs, die auf Verbindungsablehnungen hindeuten.

## Examples
Eine Webanwendung erlebt intermittierende "Zu viele Verbindungen"-Fehler während Spitzenlast. Die Untersuchung zeigt, dass der Connection Pool der Anwendung mit einer sehr hohen `max_idle_connections`-Einstellung konfiguriert ist, was dazu führt, dass sich Tausende inaktiver Verbindungen auf dem Datenbankserver anhäufen. In einem anderen Fall läuft ein Batch-Job jede Stunde und öffnet für jeden verarbeiteten Datensatz eine neue Datenbankverbindung, ohne sie zu schließen. Über die Zeit führt dies zu einem schrittweisen Anstieg der Verbindungsanzahl, bis die Datenbank ihr Limit erreicht. Dieses Problem ist verbreitet in Anwendungen, die nicht mit Verbindungsmanagement im Blick entworfen wurden, oder wo Standard-Connection-Pool-Einstellungen ohne ordentliches Tuning für die spezifische Arbeitslast genutzt werden. Es kann besonders problematisch in Microservices-Architekturen sein, in denen viele Dienste unabhängig voneinander Verbindungen zur selben Datenbank öffnen könnten.
