---
title: Datenbankverbindungslecks
description: Datenbankverbindungen werden geöffnet, aber nicht ordnungsgemäß geschlossen,
  was zur Erschöpfung des Connection-Pools und zu Anwendungsausfällen führt.
category:
- Code
- Database
- Performance
related_problems:
- slug: misconfigured-connection-pools
  similarity: 0.7
- slug: high-connection-count
  similarity: 0.7
- slug: resource-allocation-failures
  similarity: 0.7
- slug: incorrect-max-connection-pool-size
  similarity: 0.65
- slug: unreleased-resources
  similarity: 0.65
- slug: database-query-performance-issues
  similarity: 0.65
solutions:
- query-optimization-process
- connection-pooling
- resource-pooling
- monitoring
- static-analysis-and-linting
- load-testing
- observability-and-monitoring
- code-reviews
- error-handling
- production-readiness-criteria
layout: problem
lang: de
en_slug: database-connection-leaks
---

## Description

Datenbankverbindungslecks entstehen, wenn Anwendungen Datenbankverbindungen öffnen, es aber versäumen, sie ordentlich zu schließen, wenn sie nicht mehr benötigt werden. Dies führt zur schrittweisen Erschöpfung des Connection-Pools, was letztlich dazu führt, dass neue Datenbankoperationen fehlschlagen, wenn keine Verbindungen mehr verfügbar sind. Verbindungslecks sind besonders problematisch in Anwendungen mit hohem Datenverkehr und können zu vollständigen Dienstausfällen führen, die einen Neustart der Anwendung erfordern, um behoben zu werden.

## Indicators ⟡

- Die Anwendung kann Datenbankabfragen mit Fehlern wie "Connection Pool erschöpft" nicht ausführen
- Datenbank-Monitoring zeigt eine stetig steigende Anzahl aktiver Verbindungen
- Die Anwendungsperformance verschlechtert sich im Laufe der Zeit, während verfügbare Verbindungen abnehmen
- Datenbankoperationen erreichen ein Timeout oder schlagen fehl, nachdem die Anwendung eine Zeit lang gelaufen ist
- Connection-Pool-Metriken zeigen hohe Auslastung bei geringem Durchsatz

## Symptoms ▲

- [Systemausfälle](systemausfaelle.md)
<br/>  Die Erschöpfung des Connection-Pools durch geleakte Verbindungen verursacht vollständige Anwendungsausfälle, die einen Neustart erfordern, um den Dienst wiederherzustellen.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Während verfügbare Verbindungen abnehmen, stauen sich Datenbankoperationen an und erreichen ein Timeout, was die Anwendung zunehmend langsamer macht.
- [Schleichende Performance-Verschlechterung](schleichende-performance-verschlechterung.md)
<br/>  Verbindungslecks führen dazu, dass sich die Performance im Laufe der Zeit langsam verschlechtert, während der Connection-Pool schrittweise erschöpft wird.
- [Hohe Anzahl an Verbindungen](hohe-anzahl-an-verbindungen.md)
<br/>  Geleakte Verbindungen häufen sich als offene, aber ungenutzte Verbindungen an, was die Gesamtanzahl der Verbindungen auf dem Datenbankserver in die Höhe treibt.
- [Fehler bei der Ressourcenzuweisung](fehler-bei-der-ressourcenzuweisung.md)
<br/>  Wenn der Connection-Pool durch geleakte Verbindungen erschöpft ist, schlagen neue Datenbankoperationen fehl, weil keine Ressourcen zugewiesen werden können.

## Causes ▼

- [Unzureichende Fehlerbehandlung](unzureichende-fehlerbehandlung.md)
<br/>  Verbindungen, die in Try-Blöcken geöffnet, aber in Ausnahmepfaden nicht ordentlich geschlossen werden, lecken, wenn während Datenbankoperationen Fehler auftreten.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler, die mit dem Lebenszyklus-Management von Verbindungen nicht vertraut sind, nutzen keine Try-with-Resources-Muster oder ordentliche Cleanup-Logik.
- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Ohne Tests, die Fehlerpfade und lang laufende Szenarien durchspielen, bleiben Muster von Verbindungslecks bis zur Produktion unentdeckt.
- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Verbindungslecks äußern sich typischerweise nur unter anhaltender Last oder Fehlerbedingungen, die von oberflächlichem Testen nicht abgedeckt werden.

## Detection Methods ○

- **Connection-Pool-Monitoring:** Überwachung der Connection-Pool-Nutzung, aktiver Verbindungen und Pool-Erschöpfungsereignisse
- **Datenbankverbindungs-Tracking:** Nachverfolgung des Lebenszyklus von Datenbankverbindungen von der Erstellung bis zum Schließen
- **Application-Performance-Monitoring:** Überwachung von Antwortzeiten und Fehlschlagraten bei Datenbankoperationen
- **Ressourcenleck-Erkennung:** Nutzung von Profiling-Werkzeugen zur Identifikation nicht freigegebener Datenbankverbindungen
- **Lasttests:** Durchführung anhaltender Lasttests zur Identifikation von Mustern bei Verbindungslecks
- **Datenbankserver-Monitoring:** Überwachung aktiver Verbindungen auf Ebene des Datenbankservers

## Examples

Eine Webanwendung öffnet Datenbankverbindungen in einem Try-Block, um Abfragen auszuführen, schließt sie aber nur im Hauptausführungspfad, nicht in den Fehlerbehandlungspfaden. Wenn Datenbankabfragen aufgrund vorübergehender Netzwerkprobleme fehlschlagen, bleiben die Verbindungen offen und werden nie an den Pool zurückgegeben. Nach mehreren Stunden intermittierender Datenbankfehler ist der Connection-Pool erschöpft, und die Anwendung kann keine Anfragen mehr bedienen, die Datenbankzugriff erfordern. Ein weiteres Beispiel betrifft ein Batch-Verarbeitungssystem, das Datenbankverbindungen innerhalb von Schleifen öffnet, sie aber außerhalb der Schleife schließt. Wenn die Schleife Tausende von Datensätzen verarbeitet, werden Tausende von Verbindungen geöffnet, aber nur eine wird geschlossen, was den Connection-Pool schnell erschöpft und den Batch-Prozess zum Scheitern bringt.
