---
title: Verzögerungen durch externe Services
description: Eine API hängt von anderen Services (Drittanbieter oder intern) ab,
  die langsam antworten, was die API selbst langsam macht.
category:
- Code
- Performance
related_problems:
- slug: high-api-latency
  similarity: 0.85
- slug: service-timeouts
  similarity: 0.8
- slug: upstream-timeouts
  similarity: 0.75
- slug: network-latency
  similarity: 0.75
- slug: slow-application-performance
  similarity: 0.7
- slug: slow-database-queries
  similarity: 0.65
solutions:
- caching-strategy
- serialization-optimization
- asynchronous-operations
- asynchronous-processing
- circuit-breaker
- cold-start-mitigation
- optimistic-ui-updates
- retry
- timeout-management
layout: problem
lang: de
en_slug: external-service-delays
---

## Description
Verzögerungen durch externe Services sind ein verbreitetes Problem in verteilten Systemen, in denen Services oft von Drittanbieter-APIs abhängen, um Anfragen zu erfüllen. Wenn ein externer Service langsam antwortet, kann dies einen kaskadierenden Effekt haben, der Verzögerungen in nachgelagerten Services und eine schlechte Nutzererfahrung verursacht. Verzögerungen durch externe Services können durch eine Vielzahl von Faktoren verursacht werden, von Netzwerkproblemen und fehlendem ordentlichem Caching bis hin zu einem Problem mit dem Drittanbieter-Service selbst. Ein robustes Monitoring- und Alarmierungssystem ist wesentlich, um Verzögerungen durch externe Services rechtzeitig zu erkennen und darauf zu reagieren.

## Indicators ⟡
- Ihre Anwendung ist langsam, aber Ihre Server sind nicht stark ausgelastet.
- Sie sehen eine hohe Anzahl von Timeout-Fehlern in Ihren Protokollen.
- Die Performance Ihrer Anwendung ist inkonsistent.
- Sie erhalten Beschwerden von Nutzern über langsame Performance.

## Symptoms ▲

- [Hohe API-Latenz](hohe-api-latenz.md)
<br/>  Wenn externe Services langsam sind, erbt die API, die von ihnen abhängt, deren Latenz, was direkt hohe API-Antwortzeiten verursacht.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Verzögerungen durch externe Services pflanzen sich zur Anwendungsschicht fort, was nutzerseitige Features träge und träge reagierend wirken lässt.
- [Upstream-Timeouts](upstream-timeouts.md)
<br/>  Wenn externe Services zu lange zum Antworten brauchen, können vorgelagerte Aufrufer ihre konfigurierten Timeout-Fenster überschreiten und fehlschlagen.
- [Service-Timeouts](service-timeouts.md)
<br/>  Langsame externe Abhängigkeiten verursachen, dass der abhängige Service selbst ein Timeout erreicht, wenn er Anfragen nicht innerhalb akzeptabler Grenzen abschließen kann.
- [Kaskadierende Ausfälle](kaskadierende-ausfaelle.md)
<br/>  Ein langsamer externer Service kann Thread-Pool-Erschöpfung und Ressourcenmangel im aufrufenden Service verursachen, was kaskadierende Ausfälle im gesamten System auslöst.
- [Kundenunzufriedenheit](kundenunzufriedenheit.md)
<br/>  Nutzer erleben langsame oder fehlschlagende Operationen aufgrund von Verzögerungen durch externe Services, was zu Frustration und Beschwerden führt.

## Causes ▼

- [Netzwerklatenz](netzwerklatenz.md)
<br/>  Übertragungsverzögerungen im Netzwerk zwischen Services tragen direkt zu langsamen Antwortzeiten externer Services bei.
- [Schlechte Caching-Strategie](schlechte-caching-strategie.md)
<br/>  Ohne ordentliches Caching trifft jede Anfrage den externen Service, was die Auswirkung jeder Langsamkeit verstärkt, statt zwischengespeicherte Antworten zu liefern.
- [Kommunikations-Overhead zwischen Microservices](kommunikations-overhead-zwischen-microservices.md)
<br/>  Übermäßige Interservice-Kommunikation in einer Microservices-Architektur vervielfacht die Wahrscheinlichkeit und Auswirkung von Verzögerungen durch externe Services.
- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Enge Kopplung an externe Services ohne ordentliche Circuit Breaker oder Fallback-Mechanismen bedeutet, dass das System langsame Abhängigkeiten nicht elegant handhaben kann.

## Detection Methods ○

- **Verteiltes Tracing:** Nutzung von verteiltem Tracing, um einer Anfrage von der API zum externen Service zu folgen und zu identifizieren, wo die Zeit verbraucht wird.
- **Metriken und Alarmierung:** Überwachung der Latenz von Aufrufen an den externen Service. Einrichtung von Alarmen für den Fall, dass die Latenz einen bestimmten Schwellenwert überschreitet.
- **Statusseiten:** Überprüfung der Statusseite des externen Service, um zu sehen, ob Probleme gemeldet werden.
- **Service Level Agreements (SLAs):** Wenn ein SLA für den externen Service besteht, Überwachung der Service-Performance gegen das SLA.

## Examples
Eine E-Commerce-Anwendung nutzt einen Drittanbieter-Service zur Zahlungsabwicklung. Der Zahlungsservice ist langsam, was dazu führt, dass der Checkout-Prozess langsam ist. In einer Microservices-Architektur kann ein einzelner langsamer Service einen kaskadierenden Ausfall verursachen, der die gesamte Anwendung betrifft. Dies ist ein verbreitetes Problem in modernen Anwendungen, die oft durch die Komposition verschiedener Services gebaut werden. Während dieser Ansatz viele Vorteile hat, führt er auch neue Herausforderungen ein, wie die Notwendigkeit, mit Verzögerungen durch externe Services umzugehen.
