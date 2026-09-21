---
title: Service-Timeouts
description: Services schaffen es nicht, Anfragen innerhalb einer akzeptablen Zeitgrenze
  abzuschließen, was Fehler, kaskadierende Ausfälle und Systeminstabilität verursacht.
category:
- Code
- Performance
related_problems:
- slug: upstream-timeouts
  similarity: 0.9
- slug: external-service-delays
  similarity: 0.8
- slug: high-api-latency
  similarity: 0.8
- slug: network-latency
  similarity: 0.7
- slug: system-outages
  similarity: 0.65
- slug: service-discovery-failures
  similarity: 0.65
solutions:
- blue-green-canary-deployments
- event-driven-architecture
- circuit-breaker
- cold-start-mitigation
- failover-mechanisms
- nonstop-forwarding
- retry
- service-mesh
- timeout-management
layout: problem
lang: de
en_slug: service-timeouts
---

## Description
Service-Timeouts treten auf, wenn ein Service es versäumt, innerhalb einer festgelegten Zeitspanne auf eine Anfrage zu antworten. Dies ist ein häufiges Problem in verteilten Systemen, wo Services oft voneinander abhängen, um Anfragen zu erfüllen. Timeouts können durch verschiedene Faktoren verursacht werden, einschließlich Netzwerkprobleme, hohe Latenz in einem nachgelagerten Service oder ein Service, der schlicht überlastet ist. Die ordentliche Handhabung von Timeouts ist entscheidend für den Bau resilienter und zuverlässiger Systeme.

## Indicators ⟡
- Sie sehen eine hohe Anzahl an Timeout-Fehlern in Ihren Logs.
- Ihre Anwendung ist langsam, und Sie vermuten, dass dies an einer hohen Anzahl von Timeouts liegt.
- Sie erhalten Beschwerden von Nutzern über langsame Performance.
- Ihr Monitoring-System löst Alerts für Timeout-Fehler aus.

## Symptoms ▲

- [Kaskadierende Ausfälle](kaskadierende-ausfaelle.md)
<br/>  Wenn ein Service in Timeout läuft, laufen Aufrufer möglicherweise ebenfalls in Timeout, während sie darauf warten, was eine Kettenreaktion von Fehlern im gesamten System schafft.
- [Kundenunzufriedenheit](kundenunzufriedenheit.md)
<br/>  Nutzer erleben langsame Antworten oder Fehlermeldungen, wenn Services in Timeout laufen, was zu Frustration und Unzufriedenheit führt.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Anfragen, die auf Services im Timeout warten, tragen zur allgemeinen Anwendungsträgheit bei, während Threads und Verbindungen offen gehalten werden.
- [Erschöpfung des Thread-Pools](erschoepfung-des-thread-pools.md)
<br/>  Threads, die auf nachgelagerte Services im Timeout warten, bleiben blockiert und erschöpfen allmählich den Thread-Pool, was die Verarbeitung neuer Anfragen verhindert.

## Causes ▼

- [Netzwerklatenz](netzwerklatenz.md)
<br/>  Hohe Netzwerklatenz zwischen Services erhöht Round-Trip-Zeiten, was Anfragen die Timeout-Schwellen überschreiten lässt.
- [Verzögerungen durch externe Services](verzoegerungen-durch-externe-services.md)
<br/>  Langsame Antworten von externen oder Drittanbieter-Services pflanzen sich durch das System fort, während vorgelagerte Services warten und schließlich in Timeout laufen.
- [Performance-Probleme bei Datenbankabfragen](performance-probleme-bei-datenbankabfragen.md)
<br/>  Langsame Datenbankabfragen in nachgelagerten Services verursachen, dass die Anfrageverarbeitung Timeout-Limits überschreitet.
- [Ressourcenkonkurrenz](ressourcenkonkurrenz.md)
<br/>  Überlastete Services, die um begrenzte CPU-, Speicher- oder I/O-Ressourcen konkurrieren, verarbeiten Anfragen zu langsam, was Timeouts verursacht.

## Detection Methods ○

- **Distributed Tracing:** Nutzung von Werkzeugen wie Jaeger oder Zipkin zur Verfolgung von Anfragen über Service-Grenzen hinweg zur Identifikation, welcher Service-Aufruf in Timeout läuft.
- **Log-Analyse:** Aggregation und Durchsuchung von Logs aller Services zur Suche nach Timeout-Fehlermeldungen und deren Korrelation mit anderen Ereignissen.
- **Monitoring und Alerting:** Einrichtung von Alerts für Timeout-Fehlerraten (sowohl clientseitig als auch serverseitig) zur proaktiven Problemerkennung.
- **Chaos Engineering:** Absichtliches Einschleusen von Verzögerungen oder Fehlern in das System, um zu testen, wie es sich verhält, und sicherzustellen, dass Timeout- und Retry-Mechanismen wie erwartet funktionieren.

## Examples
In einem Microservices-basierten Bestellsystem ruft der `Order`-Service den `Payment`-Service auf. Der `Payment`-Service ist langsam, sodass der `Order`-Service in Timeout läuft. Dem Nutzer wird ein generischer Fehler angezeigt, aber die Zahlung könnte tatsächlich erfolgreich gewesen sein, was zu einer verwirrenden Nutzererfahrung und inkonsistenten Daten führt. In einem anderen Fall hat ein Webserver ein Standard-Timeout von 30 Sekunden. Ein datenintensiver Berichts-Endpunkt kann manchmal länger als 30 Sekunden brauchen, um einen Bericht zu generieren. Nutzer, die versuchen, auf diesen Bericht zuzugreifen, erhalten häufig einen 504-Gateway-Timeout-Fehler. Dieses Problem ist besonders häufig in komplexen, verteilten Systemen, wo eine einzige Nutzeranfrage Kommunikation zwischen Dutzenden von Services beinhalten kann. Ohne sorgfältiges Design von Timeouts und Retry-Logik können diese Systeme sehr fragil sein.
