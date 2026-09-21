---
title: Netzwerklatenz
description: Verzögerungen bei der Datenübertragung über das Netzwerk erhöhen erheblich
  die Antwortzeiten und beeinträchtigen die Anwendungsperformance.
category:
- Performance
related_problems:
- slug: high-api-latency
  similarity: 0.8
- slug: external-service-delays
  similarity: 0.75
- slug: upstream-timeouts
  similarity: 0.75
- slug: service-timeouts
  similarity: 0.7
- slug: slow-application-performance
  similarity: 0.7
- slug: slow-database-queries
  similarity: 0.65
solutions:
- caching-strategy
- serialization-optimization
- api-calls-optimization
- compression
- optimistic-ui-updates
- predictive-loading
- predictive-prefetching
- progressive-loading
- service-mesh
- virtual-networks
layout: problem
lang: de
en_slug: network-latency
---

## Description
Netzwerklatenz ist die Zeit, die Daten benötigen, um von einem Punkt zu einem anderen in einem Netzwerk zu gelangen. Während etwas Latenz unvermeidlich ist, kann hohe Netzwerklatenz die Anwendungsperformance erheblich beeinträchtigen, besonders in verteilten Systemen, in denen Services über das Netzwerk kommunizieren. Dies kann sich als langsame Antwortzeiten, Timeouts und eine generell träge Nutzererfahrung äußern. Das Verständnis und die Minderung der Auswirkung von Netzwerklatenz ist eine Schlüsselüberlegung beim Design verteilter Systeme.

## Indicators ⟡
- Ihre Anwendung ist langsam, aber Ihre Server sind nicht stark belastet.
- Sie sehen eine hohe Anzahl von Timeout-Fehlern in Ihren Logs.
- Die Performance Ihrer Anwendung ist inkonsistent.
- Sie erhalten Beschwerden von Nutzern über langsame Performance.

## Symptoms ▲

- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Netzwerkverzögerungen tragen direkt zur Anfrageverarbeitungszeit bei und verschlechtern die Gesamtreaktionsfähigkeit der Anwendung.
- [Service-Timeouts](service-timeouts.md)
<br/>  Hohe Netzwerklatenz verursacht, dass die Kommunikation zwischen Services Timeout-Schwellenwerte überschreitet, was Timeout-Fehler auslöst.
- [Negatives Nutzerfeedback](negatives-nutzerfeedback.md)
<br/>  Nutzer erleben träge Interaktionen aufgrund von Netzwerkverzögerungen und beschweren sich über schlechte Performance.
- [Hohe API-Latenz](hohe-api-latenz.md)
<br/>  Netzwerkübertragungsverzögerungen erhöhen direkt die beim Client gemessenen API-Antwortzeiten.
- [Verzögerungen durch externe Services](verzoegerungen-durch-externe-services.md)
<br/>  Netzwerkübertragungsverzögerungen tragen direkt zu langsamen Antworten von externen Services bei, von denen die Anwendung abhängt.

## Causes ▼

- [Kommunikations-Overhead zwischen Microservices](kommunikations-overhead-zwischen-microservices.md)
<br/>  APIs, die viele Roundtrips benötigen, um Operationen abzuschließen, verstärken die Auswirkung der Netzwerklatenz auf die Gesamtperformance.

## Detection Methods ○

- **Ping/Traceroute:** Nutzung von `ping` zur Messung der Roundtrip-Zeit zu einem Host und `traceroute` (oder `tracert` unter Windows) zur Identifikation des Pfads und der Latenz bei jedem Hop.
- **Netzwerk-Monitoring-Werkzeuge:** Nutzung von Werkzeugen wie Wireshark, tcpdump oder Netzwerk-Performance-Monitoring-Lösungen zur Analyse von Netzwerkverkehr und Identifikation von Engpässen.
- **Distributed Tracing:** Nachverfolgung von Anfragen über Services hinweg, um zu sehen, wie viel Zeit in Netzwerkkommunikation versus tatsächlicher Verarbeitung verbracht wird.
- **Real User Monitoring (RUM):** RUM-Werkzeuge können die von tatsächlichen Nutzern von verschiedenen Standorten aus erlebte Netzwerklatenz messen.
- **Cloud-Anbieter-Metriken:** Bei Nutzung von Cloud-Services Überwachung von Netzwerk-I/O- und Latenzmetriken, die vom Cloud-Anbieter bereitgestellt werden.

## Examples
Ein Unternehmen hat seine Hauptanwendungsserver in Nordamerika, aber ein erheblicher Teil seiner Nutzerbasis ist in Europa. Europäische Nutzer berichten konsequent von langsamer Anwendungsperformance, obwohl serverseitige Metriken niedrige Latenz zeigen. Netzwerk-Traces offenbaren hohe Latenz zwischen Europa und Nordamerika. In einem anderen Fall sind zwei Microservices, `Service A` und `Service B`, in verschiedenen virtuellen Netzwerken innerhalb derselben Cloud-Region deployt. Eine falsch konfigurierte Netzwerksicherheitsgruppe oder Routing-Tabelle verursacht, dass Verkehr zwischen ihnen über ein On-Premise-Rechenzentrum geroutet wird, was erhebliche Latenz einführt. Netzwerklatenz ist eine fundamentale Einschränkung in verteilten Systemen. Während sie nicht eliminiert werden kann, kann sie durch Strategien wie Content Delivery Networks (CDNs), Edge Computing, Optimierung von Netzwerkpfaden und Design von Anwendungen, die weniger empfindlich auf Latenz reagieren, gemindert werden.
