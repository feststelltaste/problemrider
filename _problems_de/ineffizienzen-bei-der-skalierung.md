---
title: Ineffizienzen bei der Skalierung
description: Eine Situation, in der es schwierig oder unmöglich ist, verschiedene
  Teile eines Systems unabhängig voneinander zu skalieren.
category:
- Architecture
- Performance
related_problems:
- slug: monolithic-architecture-constraints
  similarity: 0.6
- slug: technical-architecture-limitations
  similarity: 0.55
- slug: algorithmic-complexity-problems
  similarity: 0.55
- slug: architectural-mismatch
  similarity: 0.55
- slug: tight-coupling-issues
  similarity: 0.55
- slug: team-coordination-issues
  similarity: 0.55
solutions:
- capacity-planning
- elastic-scaling
- cloud-native-development
- cqrs
- cross-platform-frameworks
- data-partitioning
- data-replication
- data-stream-processing
- distributed-caching
- distributed-processing
- elastic-resource-utilization
- horizontal-scaling
- load-balancing
- load-testing
- microservices
- microservices-architecture
- monitoring-system-utilization
- nosql-databases
- parallelization
- performance-modeling
- pipelining
- proactive-capacity-management
- probabilistic-data-structures
- reactive-programming
- read-replicas
- serverless-computing
- specialized-hardware
- streaming
- stress-testing
- vertical-scaling
layout: problem
lang: de
en_slug: scaling-inefficiencies
---

## Description
Ineffizienzen bei der Skalierung treten auf, wenn es schwierig oder unmöglich ist, verschiedene Teile eines Systems unabhängig voneinander zu skalieren. Dies ist ein häufiges Problem in monolithischen Architekturen, wo alle Komponenten eng gekoppelt sind und als eine einzige Einheit deployt werden. Ineffizienzen bei der Skalierung können zu hoher Ressourcennutzung, langsamer Anwendungsperformance und schlechter Nutzererfahrung führen.

## Indicators ⟡
- Das gesamte System muss hoch- oder herunterskaliert werden, selbst wenn nur ein Teil des Systems hoher Last ausgesetzt ist.
- Es ist nicht möglich, verschiedene Teile des Systems unabhängig voneinander zu skalieren.
- Das System kann plötzliche Lastspitzen nicht bewältigen.
- Der Betrieb des Systems ist teuer, da es nicht effizient skaliert werden kann.

## Symptoms ▲

- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Wenn Systeme nicht unabhängig skaliert werden können, müssen Organisationen Ressourcen überprovisionieren, was zu unverhältnismäßig hohen Infrastruktur- und Wartungskosten führt.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Die Unfähigkeit, Engpasskomponenten unabhängig zu skalieren, bedeutet, dass das System Lastspitzen nicht bewältigen kann, was zu verschlechterter nutzerseitiger Performance führt.
- [Ressourcenkonkurrenz](ressourcenkonkurrenz.md)
<br/>  Wenn alle Komponenten dieselbe Skalierungseinheit teilen, konkurrieren Komponenten mit hoher Nachfrage mit solchen mit niedriger Nachfrage um begrenzte CPU-, Speicher- und I/O-Ressourcen.
- [Wettbewerbsnachteil](wettbewerbsnachteil.md)
<br/>  Die Unfähigkeit, effizient zu skalieren, führt zu langsameren Antwortzeiten und höheren Kosten, was die Organisation gegenüber Wettbewerbern mit skalierbareren Architekturen benachteiligt.
- [Systemausfälle](systemausfaelle.md)
<br/>  Die Unfähigkeit, unter Lastspitzen zu skalieren, kann zu Systemausfällen führen, wenn die Nachfrage die Kapazität der monolithischen Architektur übersteigt.

## Causes ▼

- [Einschränkungen durch monolithische Architektur](einschraenkungen-durch-monolithische-architektur.md)
<br/>  Monolithische Architekturen bündeln alle Komponenten in eine einzige deploybare Einheit, was es unmöglich macht, einzelne Teile unabhängig zu skalieren.
- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Eng gekoppelte Komponenten können nicht für unabhängige Skalierung getrennt werden, weil sie direkt von den Internas des jeweils anderen abhängen.
- [Gemeinsam genutzte Datenbank](gemeinsam-genutzte-datenbank.md)
<br/>  Eine gemeinsam genutzte Datenbank wird zum Skalierungsengpass, da alle Services ihren Datenbankzugriff zusammen skalieren müssen statt unabhängig.
- [Architektonische Fehlpassung](architektonische-fehlpassung.md)
<br/>  Eine Architektur, die für andere Skalierungsannahmen entworfen wurde, kann neue Lastanforderungen nicht effizient bewältigen.
- [Overhead durch atomare Operationen](overhead-durch-atomare-operationen.md)
<br/>  Konkurrenz um atomare Operationen verhindert, dass die Performance mit zusätzlichen CPU-Kernen skaliert.

## Detection Methods ○
- **Performance-Tests:** Nutzung von Performance-Testwerkzeugen zur Identifikation von Engpässen und Verbesserungsbereichen.
- **Ressourcen-Monitoring:** Überwachung der Ressourcennutzung des Systems zur Identifikation, welche Komponenten die meisten Ressourcen nutzen.
- **Architekturdiagramme:** Erstellung eines Diagramms der Systemarchitektur zur Identifikation, welche Komponenten unabhängig skaliert werden können.

## Examples
Ein Unternehmen hat eine große, monolithische E-Commerce-Anwendung. Die Anwendung besteht aus mehreren verschiedenen Komponenten, einschließlich eines Produktkatalogs, eines Warenkorbs und eines Zahlungsgateways. Der Produktkatalog ist lesehäufig, während der Warenkorb und das Zahlungsgateway schreibhäufig sind. Das Unternehmen kann den Produktkatalog nicht unabhängig vom Warenkorb und Zahlungsgateway skalieren. Infolgedessen muss das Unternehmen das gesamte System überprovisionieren, um die Spitzenlast des Produktkatalogs zu bewältigen. Dies ist teuer und ineffizient.
