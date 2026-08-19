---
title: Nicht freigegebene Ressourcen
description: Objekte, Verbindungen, Dateihandles oder andere Systemressourcen werden
  zugewiesen, aber nie ordentlich freigegeben oder geschlossen.
category:
- Code
- Performance
related_problems:
- slug: resource-allocation-failures
  similarity: 0.8
- slug: memory-leaks
  similarity: 0.7
- slug: database-connection-leaks
  similarity: 0.65
- slug: misconfigured-connection-pools
  similarity: 0.6
- slug: unbounded-data-growth
  similarity: 0.6
- slug: resource-contention
  similarity: 0.6
solutions:
- static-analysis-and-linting
- resource-pooling
- monitoring-system-utilization
- code-reviews
- profiling
- error-handling
- connection-pooling
- load-testing
- observability-and-monitoring
- exploratory-testing
layout: problem
lang: de
en_slug: unreleased-resources
---

## Description

Nicht freigegebene Ressourcen treten auf, wenn Anwendungen Systemressourcen wie Speicher, Dateihandles, Datenbankverbindungen, Netzwerk-Sockets oder andere begrenzte Ressourcen erwerben, es aber versäumen, sie ordentlich freizugeben, wenn sie nicht mehr benötigt werden. Dies führt über die Zeit zu Ressourcenerschöpfung, verschlechterter Performance und schließlich Systeminstabilität. Anders als einfache Speicherlecks umfasst dieses Problem alle Arten von Systemressourcen und kann sich je nachdem, welche Ressourcen nicht ordentlich verwaltet werden, auf verschiedene Weisen äußern.

## Indicators ⟡
- Die Systemressourcennutzung steigt kontinuierlich während der Anwendungslaufzeit
- Anwendungen stürzen schließlich mit Fehlern für „Speicher voll" oder „zu viele offene Dateien" ab
- Datenbank-Connection-Pools werden erschöpft
- Netzwerkverbindungen bleiben für längere Zeit im TIME_WAIT-Zustand
- Die Performance verschlechtert sich, während die Anwendung länger läuft

## Symptoms ▲

- [Speicherlecks](speicherlecks.md)
<br/>  Nicht freigegebene Speicherzuweisungen sind eine direkte Form von Speicherlecks, was wachsenden Speicherverbrauch über die Zeit verursacht.
- [Datenbankverbindungslecks](datenbankverbindungslecks.md)
<br/>  Das Versäumnis, Datenbankverbindungen zu schließen, ist eine spezifische Form nicht freigegebener Ressourcen, die Connection-Pools erschöpft.
- [Fehler bei der Ressourcenzuweisung](fehler-bei-der-ressourcenzuweisung.md)
<br/>  Während sich nicht freigegebene Ressourcen anhäufen, kann das System schließlich keine neuen Ressourcen zuweisen, was Fehler verursacht.
- [Release-Instabilität](release-instabilitaet.md)
<br/>  Allmähliche Ressourcenerschöpfung durch nicht freigegebene Ressourcen führt zu Abstürzen und unvorhersehbarem Systemverhalten.
- [Service-Timeouts](service-timeouts.md)
<br/>  Ressourcenerschöpfung durch nicht freigegebene Verbindungen und Handles verursacht, dass Services unresponsiv werden und in Timeout laufen.

## Causes ▼

- [Unzureichende Fehlerbehandlung](unzureichende-fehlerbehandlung.md)
<br/>  Wenn Ausnahmepfade keinen ordentlichen Bereinigungscode enthalten, werden Ressourcen, die vor dem Fehler zugewiesen wurden, nie freigegeben.
- [Unzureichendes Code-Review](unzureichendes-code-review.md)
<br/>  Der Mangel an gründlichen Code-Reviews erlaubt es Ressourcenmanagement-Fehlern, unentdeckt die Produktion zu erreichen.
- [Inkonsistente Coding-Standards](inkonsistente-coding-standards.md)
<br/>  Ohne Coding-Standards, die Ressourcenbereinigungsmuster vorschreiben, verwalten Entwickler Ressourcenlebenszyklen inkonsistent.

## Detection Methods ○
- **Ressourcen-Monitoring-Werkzeuge:** System-Level-Monitoring von Speicher, Dateihandles, Netzwerkverbindungen und anderen Ressourcen
- **Anwendungs-Profiling:** Speicher- und Ressourcenprofiler, die Ressourcenzuweisung und -freigabe verfolgen können
- **Statische Codeanalyse:** Werkzeuge, die potenzielle Ressourcenlecks im Code identifizieren können
- **Lasttests:** Erweiterte Tests, die Ressourcenlecks über die Zeit offenbaren können
- **Systemlogsanalyse:** Überwachung von Systemlogs auf Fehler oder Warnungen zur Ressourcenerschöpfung

## Examples

Eine Webanwendung öffnet Datenbankverbindungen zur Generierung von Berichten, versäumt es aber, sie ordentlich zu schließen, wenn Ausnahmen während der Berichtsverarbeitung auftreten. Über die Zeit wird der Connection-Pool erschöpft, und neue Nutzer können nicht auf die Anwendung zugreifen, weil keine Datenbankverbindungen verfügbar sind. Die Verbindungen bleiben auf dem Datenbankserver zugewiesen, bis er neu gestartet wird, obwohl die Anwendung sie nicht mehr nutzt. Ein weiteres Beispiel betrifft einen Dateiverarbeitungsservice, der Dateihandles öffnet, um Konfigurationsdateien zu lesen, sie aber nie schließt. Während die Anwendung mehr Anfragen verarbeitet, häuft sie offene Dateihandles an, bis sie das Systemlimit erreicht. Zu diesem Zeitpunkt kann die Anwendung keine Dateien mehr öffnen, einschließlich Log-Dateien, was sie mit „zu viele offene Dateien"-Fehlern abstürzen lässt. Das Problem ist besonders schwer zu diagnostizieren, weil es sich erst äußert, nachdem die Anwendung für längere Zeiträume gelaufen ist und viele Anfragen verarbeitet hat.
