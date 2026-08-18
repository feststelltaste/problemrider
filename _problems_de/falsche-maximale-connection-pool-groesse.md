---
title: Falsche maximale Connection-Pool-Größe
description: Die maximale Anzahl an Verbindungen in einem Datenbank-Connection-Pool
  ist falsch eingestellt, was entweder zu verschwendeten Ressourcen oder Verbindungserschöpfung
  führt.
category:
- Code
- Performance
related_problems:
- slug: misconfigured-connection-pools
  similarity: 0.85
- slug: high-connection-count
  similarity: 0.8
- slug: high-database-resource-utilization
  similarity: 0.7
- slug: database-connection-leaks
  similarity: 0.65
- slug: high-number-of-database-queries
  similarity: 0.65
- slug: slow-database-queries
  similarity: 0.65
solutions:
- query-optimization-process
- connection-pooling
- load-testing
- capacity-planning
- monitoring-system-utilization
- performance-measurements
- externalized-configuration
- stress-testing
- observability-and-monitoring
- production-readiness-criteria
layout: problem
lang: de
en_slug: incorrect-max-connection-pool-size
---

## Description
Das Setzen der maximalen Größe eines Connection Pools ist ein heikler Balanceakt. Wenn die Größe zu klein ist, kann der Anwendung Verbindungen fehlen, was zu Timeouts und schlechter Performance führt. Wenn die Größe zu groß ist, kann sie die Datenbank mit zu vielen Verbindungen überwältigen, was zu einer Performance- und Stabilitätsverschlechterung führt. Die optimale Größe für einen Connection Pool hängt von einer Vielzahl von Faktoren ab, einschließlich der Anzahl der Anwendungsinstanzen, der Anzahl der Threads in jeder Instanz und der Kapazität der Datenbank.

## Indicators ⟡
- Es zeigt sich eine hohe Anzahl an Verbindungsfehlern in den Logs.
- Die Anwendung ist langsam, und der Verdacht besteht, dass dies an einer hohen Anzahl an Datenbankverbindungen liegt.
- Es kommen Beschwerden von Nutzern über langsame Performance.
- Es zeigt sich eine hohe Anzahl an Timeout-Fehlern in den Logs.

## Symptoms ▲

- [Service-Timeouts](service-timeouts.md)
<br/>  Wenn der Pool zu klein ist, warten Anfragen auf verfügbare Verbindungen und laufen schließlich in ein Timeout.
- [Hohe Anzahl an Verbindungen](hohe-anzahl-an-verbindungen.md)
<br/>  Ein überdimensionierter Pool schafft unnötig viele Verbindungen zur Datenbank, was Ressourcen verschwendet.
- [Hohe Datenbank-Ressourcennutzung](hohe-datenbank-ressourcennutzung.md)
<br/>  Zu viele Verbindungen aus einem überdimensionierten Pool verbrauchen Datenbank-Speicher und -CPU-Ressourcen.
- [Erhöhte Fehlerraten](erhoehte-fehlerraten.md)
<br/>  Verbindungserschöpfung durch einen unterdimensionierten Pool oder Datenbankablehnung durch einen überdimensionierten Pool produzieren beide Anwendungsfehler.
- [Falsch konfigurierte Connection Pools](falsch-konfigurierte-connection-pools.md)
<br/>  Ein falsch dimensionierter Connection Pool ist ein wesentlicher Beitragender zu allgemeinen Connection-Pool-Fehlkonfigurationsproblemen.

## Causes ▼

- [Unvollständiges Wissen](unvollstaendiges-wissen.md)
<br/>  Entwickler verstehen möglicherweise nicht die Beziehung zwischen Anwendungs-Nebenläufigkeit, Datenbankkapazität und optimaler Pool-Dimensionierung.
- [Unzureichendes Konfigurationsmanagement](unzureichendes-konfigurationsmanagement.md)
<br/>  Schlechtes Konfigurationsmanagement bedeutet, dass Pool-Größen nicht ordentlich abgestimmt oder über Umgebungen hinweg nachverfolgt werden.

## Detection Methods ○

- **Anwendungsmetriken:** Überwachung von Connection-Pool-Metriken (z. B. aktive Verbindungen, inaktive Verbindungen, Wartezeiten, Verbindungserwerbsraten, Pool-Größe), die vom Anwendungs-Framework oder einem Monitoring-Agenten bereitgestellt werden.
- **Datenbank-Monitoring-Werkzeuge:** Beobachtung der Anzahl aktiver und inaktiver Verbindungen auf dem Datenbankserver und Vergleich mit der `max_connections`-Einstellung.
- **Log-Analyse:** Suche nach verbindungsbezogenen Fehlern in Anwendungs- und Datenbank-Logs.
- **Lasttests:** Systematische Erhöhung der Last unter Beobachtung von Connection-Pool- und Datenbankmetriken, um die optimale `max_pool_size` zu finden.

## Examples
Eine Webanwendung wird mit einer Standard-Connection-Pool-Größe von 10 deployt. Während einer Marketingkampagne steigt die Anzahl gleichzeitiger Nutzer sprunghaft auf 100. Die Anwendung beginnt "Connection Pool erschöpft"-Fehler zu werfen, weil sie nicht genügend Datenbankverbindungen erwerben kann, um alle Anfragen zu bedienen. In einem anderen Fall ist ein Microservice mit einer `max_pool_size` von 200 konfiguriert, aber die Datenbank, mit der er sich verbindet, erlaubt nur maximal 100 Verbindungen. Dies führt zu intermittierenden Verbindungsfehlern und verschwendeten Anwendungsressourcen beim Versuch, Verbindungen zu öffnen, die die Datenbank ablehnen wird. Ordentliche Konfiguration von Datenbank-Connection-Pools ist entscheidend für die Performance und Stabilität jeder Anwendung, die mit einer relationalen Datenbank interagiert. Sie erfordert das Verständnis sowohl der Nebenläufigkeitsbedürfnisse der Anwendung als auch der Kapazität der Datenbank.
