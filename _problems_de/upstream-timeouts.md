---
title: Upstream-Timeouts
description: Services, die eine API konsumieren, scheitern, weil sie keine Antwort
  innerhalb ihres konfigurierten Timeout-Fensters erhalten.
category:
- Code
- Performance
related_problems:
- slug: service-timeouts
  similarity: 0.9
- slug: high-api-latency
  similarity: 0.8
- slug: external-service-delays
  similarity: 0.75
- slug: network-latency
  similarity: 0.75
- slug: slow-application-performance
  similarity: 0.65
- slug: high-database-resource-utilization
  similarity: 0.65
solutions:
- event-driven-architecture
- circuit-breaker
- timeout-management
- retry
- bulkhead
- graceful-degradation
- monitoring
- service-level-agreements
layout: problem
lang: de
en_slug: upstream-timeouts
---

## Description
Upstream-Timeouts sind ein häufiges Problem in verteilten Systemen, bei dem ein Service es versäumt, innerhalb einer festgelegten Zeitgrenze eine Antwort von einem anderen Service (einem „Upstream"-Service) zu erhalten, von dem er abhängt. Dies ist nicht nur ein einfacher Fehler; es ist ein Versagen eines Teils des Systems, die Performance-Erwartungen eines anderen zu erfüllen. Diese Timeouts können kaskadieren, was Fehler in nachgelagerten Services verursacht und letztlich die Endnutzererfahrung beeinflusst. Das Verständnis und die Minderung von Upstream-Timeouts ist entscheidend für den Bau resilienter und zuverlässiger Microservices-Architekturen.

## Indicators ⟡
- Sie sehen eine hohe Anzahl an Timeout-Fehlern in Ihren Logs.
- Ihre Anwendung ist langsam, und Sie vermuten, dass dies an einer hohen Anzahl von Timeouts liegt.
- Sie erhalten Beschwerden von Nutzern über langsame Performance.
- Ihr Monitoring-System löst Alerts für Timeout-Fehler aus.

## Symptoms ▲

- [Kaskadierende Ausfälle](kaskadierende-ausfaelle.md)
<br/>  Upstream-Timeouts pflanzen sich durch Service-Ketten fort, was verursacht, dass nachgelagerte Services ebenfalls scheitern oder in Timeout laufen.
- [Erhöhte Fehlerraten](erhoehte-fehlerraten.md)
<br/>  Timeout-Fehler erhöhen direkt die Gesamtfehlerrate des Systems, während Anfragen scheitern, ohne Antworten zu erhalten.
- [Nutzerfrustration](nutzerfrustration.md)
<br/>  Endnutzer erleben langsame Antworten oder Fehler, die durch Upstream-Timeouts verursacht werden, was zu Unzufriedenheit führt.
- [Hohe Anzahl an Verbindungen](hohe-anzahl-an-verbindungen.md)
<br/>  Wartende Verbindungen häufen sich an, wenn Upstream-Services langsam sind, während aufrufende Services Verbindungen bis zum Timeout offen halten.

## Causes ▼

- [Hohe API-Latenz](hohe-api-latenz.md)
<br/>  Langsame API-Antwortzeiten sind die direkte Ursache für Upstream-Timeouts, wenn Antworten konfigurierte Timeout-Fenster überschreiten.
- [Verzögerungen durch externe Services](verzoegerungen-durch-externe-services.md)
<br/>  Verzögerungen in externen Services, von denen die API abhängt, pflanzen sich nach oben fort, was verursacht, dass Upstream-Aufrufer in Timeout laufen.
- [Falsch konfigurierte Connection Pools](falsch-konfigurierte-connection-pools.md)
<br/>  Falsch konfigurierte Connection-Pools können Verbindungen erschöpfen und Verzögerungen verursachen, die Upstream-Timeouts auslösen.
- [Ressourcenkonkurrenz](ressourcenkonkurrenz.md)
<br/>  Ressourcenkonkurrenz im Upstream-Service verursacht, dass er Anfragen langsam verarbeitet, was Aufrufer-Timeout-Schwellen überschreitet.

## Detection Methods ○

- **Distributed Tracing:** Nutzung von Distributed Tracing zur Verfolgung einer Anfrage über mehrere Services hinweg und Lokalisierung, wo der Timeout auftritt.
- **Log-Analyse:** Zentralisiertes Logging kann genutzt werden, um Timeout-Fehler in einem Service mit langsamen Antworten in einem anderen zu korrelieren.
- **Metriken und Alerting:** Überwachung von Timeout-Metriken sowohl im aufrufenden Service als auch in der API. Einrichtung von Alerts für ungewöhnliche Spitzen.
- **Chaos Engineering:** Absichtliches Einschleusen von Verzögerungen in Services, um zu testen, wie sich das System verhält, und sicherzustellen, dass Timeouts anmutig gehandhabt werden.

## Examples
Ein `UserService` ruft einen `AuthService` auf, um einen Nutzer zu authentifizieren. Der `AuthService` erlebt hohe Latenz. Der `UserService` hat ein 2-Sekunden-Timeout für den Aufruf des `AuthService`. Wenn der `AuthService` länger als 2 Sekunden zum Antworten braucht, läuft der `UserService` in Timeout und gibt einen Fehler an den Nutzer zurück. In einem anderen Fall besteht eine Datenverarbeitungspipeline aus mehreren Services, die sich in Sequenz gegenseitig aufrufen. Einer der Services in der Mitte der Pipeline ist langsam. Dies verursacht, dass alle nachfolgenden Services in der Pipeline in Timeout laufen, obwohl sie nicht die Grundursache des Problems sind. Dies ist ein häufiges Problem in Microservices-Architekturen, wo eine einzelne Nutzeranfrage eine Kaskade von Aufrufen an mehrere Services auslösen kann. Ein Timeout in einem dieser Services kann verursachen, dass die gesamte Anfrage scheitert.
