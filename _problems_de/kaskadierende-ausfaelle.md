---
title: Kaskadierende Ausfälle
description: Eine einzelne Änderung löst eine Kettenreaktion von Ausfällen über
  mehrere Systemkomponenten hinweg aus.
category:
- Architecture
- Code
- Performance
related_problems:
- slug: ripple-effect-of-changes
  similarity: 0.7
- slug: cascade-delays
  similarity: 0.65
- slug: change-management-chaos
  similarity: 0.65
- slug: configuration-chaos
  similarity: 0.65
- slug: system-outages
  similarity: 0.65
- slug: tight-coupling-issues
  similarity: 0.6
solutions:
- backpressure
- event-driven-architecture
- observability-and-monitoring
- asynchronous-processing
- bulkhead
- business-event-processing
- chaos-engineering
- circuit-breaker
- dead-letter-queue
- distributed-tracing
- failover-cluster
- failover-mechanisms
- fault-containment
- fault-tolerant-data-structures
- graceful-degradation
- high-availability-architectures
- idempotency-design
- idempotent-operations
- integration-tests
- isolation-of-faulty-components
- load-shedding
- nonstop-forwarding
- rate-limiting
- reactive-programming
- redundancy
- resilience
- retry
- security-incident-handling
- security-monitoring
- service-mesh
- site-reliability-engineering-sre
- status-monitoring
- stress-testing
- timeout-management
- transactions
- watchdog
- write-ahead-logging
- data-flow-control
- defense-lines
- error-handling
- exceptions
- incident-response-measures
- network-segmentation
- saga-pattern
layout: problem
lang: de
en_slug: cascade-failures
---

## Description

Kaskadierende Ausfälle entstehen, wenn eine einzelne Änderung, ein Fehler oder ein Ausfall in einer Komponente einen Dominoeffekt von Ausfällen durch miteinander verbundene Systemkomponenten hinweg verursacht. Diese Ausfälle breiten sich schnell durch das System aus, weil Komponenten eng gekoppelt sind oder kritische Ressourcen teilen, was es schwierig macht, Probleme auf ihre Quelle einzugrenzen. Kaskadierende Ausfälle sind besonders gefährlich, weil sie kleinere Probleme in systemweite Ausfälle verwandeln und die Wiederherstellung extrem erschweren können.

## Indicators ⟡
- Ausfälle einzelner Komponenten führen dazu, dass mehrere Systembereiche nicht mehr verfügbar sind
- Kleine Änderungen verursachen häufig weitreichende Testfehlschläge
- Systemausfälle betreffen scheinbar unzusammenhängende Funktionalität
- Die Wiederherstellung nach Ausfällen erfordert den Neustart mehrerer Komponenten oder des gesamten Systems
- Fehlermeldungen einer Komponente lösen Fehler in vielen anderen Komponenten aus

## Symptoms ▲

- [Erhöhte Fehlerraten](erhoehte-fehlerraten.md)
<br/>  Kaskadierende Ausfälle äußern sich als scheinbar zufällige Ausfälle über verschiedene Systemkomponenten hinweg, die schwer auf eine Grundursache zurückzuführen sind.
- [Unzufriedenheit der Stakeholder](unzufriedenheit-der-stakeholder.md)
<br/>  Systemweite Ausfälle durch kaskadierende Ausfälle beeinträchtigen die Nutzererfahrung und den Geschäftsbetrieb erheblich.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Die Diagnose und Behebung von Mustern kaskadierender Ausfälle erfordert umfangreiche Untersuchung über mehrere Komponenten hinweg, was die Kosten erhöht.

## Causes ▼

- [ABI-Kompatibilitätsprobleme](abi-kompatibilitaetsprobleme.md)
<br/>  In eng gekoppelten Systemen ohne Fehlerisolation kann eine Binärschnittstellen-Unstimmigkeit, die eine Komponente zum Absturz bringt, sich zu einer Kaskade von Ausfällen bei Abhängigen fortpflanzen.
- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Eng gekoppelte Komponenten pflanzen Ausfälle fort, weil sie nicht unabhängig arbeiten können, wenn Abhängigkeiten ausfallen.
- [Single Points of Failure](single-points-of-failure.md)
<br/>  Kritische gemeinsam genutzte Komponenten ohne Redundanz werden zu Ausfallquellen, die alle abhängigen Systeme betreffen.
- [Unzureichende Fehlerbehandlung](unzureichende-fehlerbehandlung.md)
<br/>  Unzureichende Fehlerbehandlung bedeutet, dass Komponenten abstürzen, statt sich elegant zu degradieren, wenn vorgelagerte Services ausfallen.
- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Fehlendes Testen von Ausfallszenarien bedeutet, dass Pfade kaskadierender Ausfälle erst entdeckt werden, wenn sie in der Produktion auftreten.
- [Fehler bei der Service Discovery](fehler-bei-der-service-discovery.md)
<br/>  Fehlgeschlagene Service Discovery führt dazu, dass Anfragen an nicht verfügbare Instanzen weitergeleitet werden, was kaskadierende Ausfälle auslöst.
- [API-Versionierungskonflikte](api-versionierungskonflikte.md)
<br/>  Eine inkompatible API-Version zwischen zwei Services kann dazu führen, dass ein Aufruf fehlschlägt, und in einer eng gekoppelten Aufrufkette pflanzt sich dieser einzelne Fehlschlag zu einer Kaskade über abhängige Services fort.

## Detection Methods ○
- **Abhängigkeits-Mapping:** Visualisierung von Komponentenabhängigkeiten zur Identifikation potenzieller Kaskadenpfade
- **Ausfallsimulation:** Chaos-Engineering-Ansätze, die absichtlich Komponenten ausfallen lassen, um Kaskadenverhalten zu testen
- **Monitoring-Korrelation:** Nachverfolgung, wie oft Ausfälle einer Komponente mit Ausfällen anderer zusammenfallen
- **Wiederherstellungszeit-Analyse:** Messung, wie lange die Wiederherstellung nach verschiedenen Arten von Ausfällen dauert
- **Fehlermuster-Analyse:** Identifikation von Mustern, bei denen einzelne Grundursachen mehrere Fehlertypen erzeugen

## Examples

Ein E-Commerce-System hat einen gemeinsam genutzten Nutzerauthentifizierungsdienst, von dem alle anderen Komponenten abhängen. Wenn ein Datenbankverbindungspool im Authentifizierungsdienst erschöpft ist, reagiert er nicht mehr auf Anfragen. Dies führt dazu, dass der Produktkatalogdienst ausfällt, weil er Nutzerberechtigungen nicht verifizieren kann, der Warenkorbdienst ausfällt, weil er Nutzer nicht identifizieren kann, der Zahlungsdienst beim Warten auf Nutzerverifikation ein Timeout erreicht und die Empfehlungs-Engine abstürzt, weil sie nicht auf Nutzerpräferenzen zugreifen kann. Was als einfaches Konfigurationsproblem eines Verbindungspools begann, hat die gesamte Plattform lahmgelegt. Die Wiederherstellung erfordert nicht nur die Behebung des Authentifizierungsdienstes, sondern auch den Neustart aller anderen Services, die beim Versuch, ihn zu erreichen, abgestürzt sind. Ein weiteres Beispiel betrifft eine Datenverarbeitungspipeline, bei der jede Stufe Ergebnisse synchron an die nächste Stufe weitergibt. Wenn die dritte Stufe auf einen korrupten Datensatz stößt und abstürzt, führt dies dazu, dass die zweite Stufe beim Warten auf eine Antwort ein Timeout erreicht, was dazu führt, dass die erste Stufe den Speicher mit angestauten Elementen erschöpft, sodass letztlich die gesamte Pipeline neu gestartet und alle Daten in Bearbeitung neu verarbeitet werden müssen.
