---
title: Fehler bei der Ressourcenzuweisung
description: Objekte, Verbindungen, Dateihandles oder andere Systemressourcen werden
  zugewiesen, aber nie ordentlich freigegeben oder geschlossen, was zu Ressourcenerschöpfung
  führt.
category:
- Code
- Performance
related_problems:
- slug: unreleased-resources
  similarity: 0.8
- slug: database-connection-leaks
  similarity: 0.7
- slug: memory-leaks
  similarity: 0.6
- slug: misconfigured-connection-pools
  similarity: 0.6
- slug: excessive-object-allocation
  similarity: 0.6
- slug: high-connection-count
  similarity: 0.6
solutions:
- change-management-process
- monitoring-system-utilization
- capacity-planning
- resource-pooling
- resource-usage-optimization
- profiling
- elastic-resource-utilization
- load-testing
- observability-and-monitoring
- production-readiness-criteria
layout: problem
lang: de
en_slug: resource-allocation-failures
---

## Description

Fehler bei der Ressourcenzuweisung treten auf, wenn Anwendungen Systemressourcen wie Dateihandles, Datenbankverbindungen, Netzwerk-Sockets oder Speicherzuweisungen erwerben, es aber versäumen, sie ordentlich freizugeben, wenn sie nicht mehr benötigt werden. Dies führt zu Ressourcenerschöpfung, bei der dem System verfügbare Ressourcen ausgehen, was Anwendungsfehlschläge, Performance-Verschlechterung oder Systeminstabilität verursacht. Das Problem ist besonders schwerwiegend in lang laufenden Anwendungen und Servern.

## Indicators ⟡

- Die Anwendung schafft es nicht, Dateien zu öffnen oder Verbindungen aufzubauen, nachdem sie eine Weile gelaufen ist
- Das System meldet „zu viele offene Dateien" oder ähnliche Ressourcenlimitfehler
- Der Datenbank-Connection-Pool erschöpft verfügbare Verbindungen
- Netzwerk-Socket-Operationen scheitern mit „Ressource nicht verfügbar"-Fehlern
- System-Monitoring zeigt stetig steigende Ressourcennutzung ohne entsprechende Freigabe

## Symptoms ▲

- [Schleichende Performance-Verschlechterung](schleichende-performance-verschlechterung.md)
<br/>  Während Ressourcen über die Zeit lecken, verschlechtert sich die Anwendungsperformance stetig aufgrund zunehmender Ressourcenknappheit.
- [Erschöpfung des Thread-Pools](erschoepfung-des-thread-pools.md)
<br/>  Threads, die aufgrund unsachgemäßen Ressourcenmanagements nie an den Pool zurückgegeben werden, erschöpfen schließlich alle verfügbaren Threads.
- [Ressourcenkonkurrenz](ressourcenkonkurrenz.md)
<br/>  Lecke Ressourcen verringern den Pool verfügbarer Ressourcen, was verbleibende Prozesse zwingt, intensiver um das Verbleibende zu konkurrieren.

## Causes ▼

- [Unzureichende Fehlerbehandlung](unzureichende-fehlerbehandlung.md)
<br/>  Schlechte Fehlerbehandlung versäumt es, Ressourcen in Ausnahmepfaden zu bereinigen, was Ressourcen lecken lässt, wenn Fehler auftreten.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler, die mit Ressourcenmanagement-Mustern nicht vertraut sind, versäumen es, Ressourcenbereinigung ordentlich zu implementieren, besonders in komplexen Fehlerszenarien.
- [Schlechte Testabdeckung](schlechte-testabdeckung.md)
<br/>  Fehlende Tests für Fehlerbedingungen und lang laufende Szenarien bedeuten, dass Ressourcenlecks unentdeckt bleiben, bis sie Produktionsfehlschläge verursachen.

## Detection Methods ○

- **Ressourcen-Monitoring-Werkzeuge:** Nutzung von System-Monitoring zur Nachverfolgung von Dateihandles, Verbindungen und anderer Ressourcennutzung über die Zeit
- **Anwendungs-Profiling:** Profiling von Anwendungen zur Identifikation von Ressourcenerwerb- und -freigabemustern
- **Statische Codeanalyse:** Analyse von Code auf Ressourcenerwerb ohne entsprechende Bereinigung
- **Lasttests:** Durchführung anhaltender Lasttests zur Identifikation von Ressourcenlecks unter operativen Bedingungen
- **System-Ressourcenlimits:** Überwachung des Ressourcenverbrauchs und der Limits auf Systemebene
- **Connection-Pool-Überwachung:** Nachverfolgung der Nutzung von Datenbank- und Netzwerk-Connection-Pools

## Examples

Ein Web-Service öffnet Datenbankverbindungen zur Verarbeitung von Anfragen, schließt sie aber bei Fehlerbedingungen nicht. Nach der Handhabung mehrerer tausend Anfragen mit intermittierenden Fehlern ist der Connection Pool erschöpft, und neue Anfragen können nicht verarbeitet werden, was einen Anwendungsneustart erfordert. Ein weiteres Beispiel betrifft eine Dateiverarbeitungsanwendung, die Dateihandles öffnet, um Konfigurationsdateien zu lesen, sie aber nicht in einem Finally-Block schließt. Über die Zeit erreicht die Anwendung das Dateihandle-Limit des Betriebssystems und stürzt ab, wenn sie versucht, zusätzliche Dateien zu öffnen, obwohl die Dateien, auf die sie zuzugreifen versuchte, von der Anwendungslogik nicht mehr genutzt werden.
