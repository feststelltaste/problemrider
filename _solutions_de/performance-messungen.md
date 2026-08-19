---
title: Performance-Messungen
description: Kontinuierliche Messung und Speicherung von Performance-Metriken
  in Produktion.
category:
- Performance
- Operations
problems:
- gradual-performance-degradation
- monitoring-gaps
- slow-application-performance
- slow-incident-resolution
- quality-blind-spots
- capacity-mismatch
- alignment-and-padding-issues
- atomic-operation-overhead
- data-structure-cache-inefficiency
- dma-coherency-issues
- endianness-conversion-overhead
- false-sharing
- incorrect-index-type
- incorrect-max-connection-pool-size
- index-fragmentation
- inefficient-database-indexing
- interrupt-overhead
- lock-contention
- memory-barrier-inefficiency
- misconfigured-connection-pools
- poor-caching-strategy
- queries-that-prevent-index-usage
- unoptimized-file-access
- unused-indexes
- algorithmic-complexity-problems
- garbage-collection-pressure
- high-resource-utilization-on-client
- inefficient-code
- insufficient-worker-capacity
- long-running-database-transactions
- memory-fragmentation
- memory-swapping
- n-plus-one-query-problem
- virtual-memory-thrashing
- work-queue-buildup
- high-number-of-database-queries
- imperative-data-fetching-logic
- inefficient-frontend-code
- long-running-transactions
- rate-limiting-issues
- serialization-deserialization-bottlenecks
- task-queues-backing-up
layout: solution
lang: de
en_slug: performance-measurements
related_solutions:
- slug: continuous-performance-monitoring
  similarity: 0.9
- slug: transparent-performance-metrics
  similarity: 0.85
- slug: monitoring
  similarity: 0.8
- slug: performance-budgets
  similarity: 0.8
- slug: service-level-indicators
  similarity: 0.75
- slug: monitoring-system-utilization
  similarity: 0.75
---

## Description

Performance-Messung instrumentiert die Codepfade eines Systems, um kontinuierlich Zeit- und Ressourcennutzungsmetriken in Produktion zu erfassen und zu speichern, statt sich auf isolierte Benchmarks oder Nutzerbeschwerden zu verlassen, um zu offenbaren, wie sich das System tatsächlich verhält. Legacy-Systeme sammeln über Jahre inkrementeller Änderungen still Performance-Regressionen an, und ohne eine historische Aufzeichnung von Perzentil-Antwortzeiten, Ressourcennutzung und deren Korrelation zu spezifischen Deployments wird Verschlechterung erst sichtbar, wenn sie bereits Krisenniveau erreicht hat. Die Erfassung vollständiger Verteilungen — p50, p95, p99 — statt Durchschnittswerten legt Tail-Latenz-Probleme offen, die Durchschnitte vollständig verbergen, und die Korrelation dieser Daten mit Deployment-Ereignissen verwandelt „das System wurde irgendwann langsamer" in „diese spezifische Änderung hat es verursacht", was den Unterschied zwischen einer Untersuchung ausmacht, die Minuten dauert, und einer, die Wochen dauert.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Instrumentieren Sie zentrale Codepfade mit Zeitmetriken, beginnend mit den nutzersichtbarsten Operationen
- Setzen Sie ein Metriken-Erfassungssystem ein (z. B. Prometheus, Datadog, StatsD), das Zeitreihen-Performance-Daten speichert
- Erstellen Sie Dashboards, die Performance-Trends über die Zeit visualisieren und Verschlechterung sofort sichtbar machen
- Richten Sie Alarme für Performance-Schwellwertverletzungen ein, sodass Probleme erkannt werden, bevor Nutzer sie melden
- Erfassen Sie Perzentilverteilungen (p50, p95, p99) statt nur Durchschnittswerte, um das vollständige Performance-Bild zu verstehen
- Korrelieren Sie Performance-Metriken mit Deployment-Ereignissen, um durch spezifische Änderungen eingeführte Regressionen zu identifizieren
- Bewahren Sie historische Daten lange genug auf, um saisonale Muster und langfristige Trends zu beobachten

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Macht schrittweise Performance-Verschlechterung sichtbar, bevor sie Krisenniveau erreicht
- Liefert evidenzbasierte Daten zur Priorisierung von Performance-Verbesserungen
- Reduziert die durchschnittliche Zeit bis zur Lösung von Performance-Vorfällen durch schnellere Ursachenidentifikation
- Schafft Verantwortlichkeit, indem Performance-Änderungen mit spezifischen Deployments verknüpft werden

**Kosten und Risiken:**
- Instrumentierung fügt der Anfrageverarbeitung einen kleinen Overhead hinzu
- Legacy-Systeme ohne standardisierte Instrumentierungspunkte erfordern erheblichen anfänglichen Aufwand
- Metriken-Infrastruktur erfordert eigene Wartung, Speicherung und Überwachung
- Zu viele Metriken können Rauschen und Alarmmüdigkeit erzeugen
- Teams optimieren möglicherweise übermäßig auf messbare Kennzahlen, während sie von Nutzern wahrgenommene Probleme übersehen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Bankanwendung erlebte intermittierende Verlangsamungen, die von Kunden gemeldet, aber nie im Testing reproduziert werden konnten. Das Team fügte verteiltes Tracing und Antwortzeitmetriken zu allen API-Endpunkten hinzu und speicherte die Daten in Prometheus mit Grafana-Dashboards. Innerhalb von zwei Wochen offenbarten die Dashboards, dass die p99-Antwortzeit für Kontostandsabfragen jeden Tag zwischen 14 und 15 Uhr auf 15 Sekunden anstieg, korrelierend mit einem automatisierten Abgleichs-Batch-Job, der um Datenbankverbindungen konkurrierte. Diese Erkenntnis, unsichtbar ohne kontinuierliche Messung, führte zur Umplanung des Batch-Jobs auf verkehrsarme Stunden und zur Implementierung von Connection-Pool-Isolation.
