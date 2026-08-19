---
title: Erschöpfung des Thread-Pools
description: Alle verfügbaren Threads im Thread-Pool werden von lang laufenden oder
  blockierten Operationen verbraucht, was die Verarbeitung neuer Aufgaben verhindert.
category:
- Code
- Performance
related_problems:
- slug: insufficient-worker-capacity
  similarity: 0.6
- slug: lock-contention
  similarity: 0.6
- slug: resource-contention
  similarity: 0.6
- slug: deadlock-conditions
  similarity: 0.6
- slug: resource-allocation-failures
  similarity: 0.6
- slug: database-connection-leaks
  similarity: 0.6
solutions:
- backpressure
- capacity-planning
- concurrency-control
- elastic-scaling
- resource-pooling
- asynchronous-operations
- asynchronous-processing
- bulkhead
- circuit-breaker
- reactive-programming
- timeout-management
layout: problem
lang: de
en_slug: thread-pool-exhaustion
---

## Description

Erschöpfung des Thread-Pools tritt auf, wenn alle verfügbaren Threads im Thread-Pool einer Anwendung von lang laufenden, blockierten oder feststeckenden Operationen verbraucht werden, sodass keine Threads verfügbar bleiben, um neue eingehende Anfragen oder Aufgaben zu verarbeiten. Dies schafft eine Situation, in der die Anwendung zu hängen oder unresponsiv zu werden scheint, obwohl das zugrunde liegende System über verfügbare CPU- und Speicherressourcen verfügt. Erschöpfung des Thread-Pools ist häufig in Serveranwendungen und kann vollständige Serviceausfälle verursachen.

## Indicators ⟡

- Die Anwendung hört auf, auf neue Anfragen zu antworten, während sie normal zu laufen scheint
- Thread-Pool-Monitoring zeigt alle Threads in Nutzung ohne verfügbare für neue Aufgaben
- Neue Operationen stauen sich unbegrenzt auf, ohne verarbeitet zu werden
- Die CPU-Nutzung kann niedrig sein, obwohl die Anwendung beschäftigt erscheint
- Antwortzeiten steigen dramatisch, oder Operationen laufen in Timeout

## Symptoms ▲

- [Kaskadierende Ausfälle](kaskadierende-ausfaelle.md)
<br/>  Wenn ein Service seinen Thread-Pool erschöpft, scheitern auch abhängige Services, während ihre Anfragen in Timeout laufen, was kaskadierende Ausfälle verursacht.
- [Systemausfälle](systemausfaelle.md)
<br/>  Vollständige Thread-Pool-Erschöpfung verursacht effektiv Serviceausfälle, während die Anwendung vollständig unresponsiv wird.
- [Unvorhersehbares Systemverhalten](unvorhersehbares-systemverhalten.md)
<br/>  Das System scheint mit niedriger CPU-Nutzung zu hängen oder sich unvorhersehbar zu verhalten, was die Grundursache schwer zu diagnostizieren macht.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Vor vollständiger Erschöpfung verursacht teilweise Thread-Pool-Erschöpfung langsame Anwendungsperformance, während weniger Threads verfügbar sind.

## Causes ▼

- [Ressourcenkonkurrenz](ressourcenkonkurrenz.md)
<br/>  Konkurrenz um begrenzte Thread-Pool-Ressourcen zwischen verschiedenen Operationen führt zu Erschöpfung unter Last.
- [Deadlock-Zustände](deadlock-zustaende.md)
<br/>  Deadlockte Threads verbrauchen dauerhaft Thread-Pool-Ressourcen, was den verfügbaren Pool graduell erschöpft.
- [Nicht freigegebene Ressourcen](nicht-freigegebene-ressourcen.md)
<br/>  Threads, die nach Abschluss oder Timeout nicht ordentlich freigegeben werden, verringern dauerhaft den verfügbaren Thread-Pool.
- [Service-Timeouts](service-timeouts.md)
<br/>  Ohne ordentliche Timeout-Einstellungen blockieren Threads unbegrenzt, während sie auf langsame oder unresponsive externe Services warten.

## Detection Methods ○

- **Thread-Pool-Monitoring:** Überwachung der Thread-Pool-Nutzung, aktiver Threads und Warteschlangentiefen
- **Thread-Dump-Analyse:** Analyse von Thread-Dumps zur Identifikation, was Threads tun, wenn Erschöpfung auftritt
- **Application Performance Monitoring:** Verfolgung von Antwortzeiten und Durchsatz zur Identifikation von Thread-Pool-Engpässen
- **Ressourcennutzungs-Monitoring:** Überwachung von CPU-, Speicher- und I/O-Nutzung während der Thread-Pool-Erschöpfung
- **Lasttests:** Testen der Anwendung unter verschiedenen Lastbedingungen zur Identifikation von Thread-Pool-Grenzen
- **Timeout-Konfigurationsanalyse:** Überprüfung der Timeout-Einstellungen für Operationen, die Thread-Pool-Threads verbrauchen

## Examples

Ein Web-Service verarbeitet Datei-Uploads, indem er den gesamten Dateiinhalt innerhalb des Anfrage-Threads in den Speicher liest. Wenn Nutzer sehr große Dateien hochladen, brauchen diese Operationen mehrere Minuten zur Fertigstellung, was Anfrage-Handhabungs-Threads für längere Zeiträume verbraucht. Während Spitzennutzung werden alle verfügbaren Anfrage-Threads mit Datei-Upload-Verarbeitung belegt, was den Server daran hindert, andere HTTP-Anfragen einschließlich einfacher Health-Checks zu handhaben. Ein weiteres Beispiel betrifft eine Anwendung, die synchrone Aufrufe an externe Web-Services ohne Timeout-Konfiguration macht. Wenn die externen Services langsam oder unresponsiv werden, werden alle Thread-Pool-Threads blockiert, während sie auf Antworten warten, die vielleicht nie kommen, was effektiv die gesamte Anwendung einfriert, bis sich die externen Services erholen oder Verbindungen auf TCP-Ebene in Timeout laufen.
