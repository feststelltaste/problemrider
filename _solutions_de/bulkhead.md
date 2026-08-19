---
title: Bulkhead
description: Aufteilung eines Systems in isolierte Bereiche, um Fehlerausbreitung
  zu begrenzen.
category:
- Architecture
problems:
- cascade-failures
- single-points-of-failure
- monolithic-architecture-constraints
- system-outages
- resource-contention
- thread-pool-exhaustion
- high-coupling-low-cohesion
- upstream-timeouts
layout: solution
lang: de
en_slug: bulkhead
related_solutions:
- slug: fault-containment
  similarity: 0.8
- slug: isolation-of-faulty-components
  similarity: 0.75
- slug: rate-limiting
  similarity: 0.7
- slug: circuit-breaker
  similarity: 0.7
- slug: resilience
  similarity: 0.65
- slug: backpressure
  similarity: 0.65
---

## Description

Das Bulkhead-Muster teilt die Ressourcenpools eines Systems — Threads, Datenbankverbindungen, Speicher — in separate, isolierte Partitionen auf, die unterschiedlichen Funktionen zugewiesen sind, sodass Erschöpfung oder Ausfall in einer Partition nicht die Kapazität verbrauchen kann, die eine andere Partition braucht, um weiter zu funktionieren. Der Mechanismus ist genau aus diesem Grund nach dem Schiffsdesign benannt: Ein flutendes Abteil sollte nicht das ganze Schiff versenken, und eine langsame oder ausfallende Abhängigkeit sollte nicht, durch den Verbrauch eines gemeinsamen Thread-Pools, nicht verwandte Funktionalität mit sich reißen, die zufällig denselben Prozess teilt. Legacy-Systeme neigen besonders zu dem Fehler, den dieses Muster verhindert, weil sie häufig als Monolithen gebaut wurden, in denen standardmäßig alle Funktionalität still einen Thread-Pool oder einen Verbindungspool teilt, ohne dass jemand bewusst entschieden hätte, dass Empfehlungs-Engine-Aufrufe und Checkout-Verarbeitungs-Aufrufe sich gegenseitig aushungern dürfen sollten. Bulkheads einzuführen bedeutet zu identifizieren, welche Funktionen kritisch sind und welche nicht, und jeder ihre eigene reservierte Kapazität zu geben — separate Thread-Pools, separate Verbindungspools, manchmal völlig separate Infrastruktur —, sodass eine langsame Drittanbieter-API, die von einer nicht kritischen Funktion aufgerufen wird, nur diese Funktion degradiert statt in einen websiteweiten Ausfall zu kaskadieren. Der Tradeoff ist, dass reservierte, aber ungenutzte Kapazität in einer unterausgelasteten Partition verschwendete Ressource ist, die ein vollständig gemeinsamer Pool genutzt hätte, sodass Bulkhead-Grenzen bewusst dimensioniert werden müssen statt einheitlich überall angewendet zu werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie kritische und nicht-kritische Systemfunktionen und trennen Sie ihre Ressourcenpools (Thread-Pools, Verbindungspools, Speicher)
- Isolieren Sie externe Serviceaufrufe in dedizierte Thread-Pools oder Prozessgrenzen, sodass eine langsame Abhängigkeit nicht das gesamte System aushungern kann
- Nutzen Sie separate Datenbankverbindungspools für verschiedene Module, um zu verhindern, dass die Abfragen eines Moduls gemeinsam genutzte Verbindungen erschöpfen
- Deployen Sie kritische Komponenten auf separater Infrastruktur, sodass ressourcenintensive Batch-Jobs Echtzeitoperationen nicht beeinträchtigen können
- Implementieren Sie Anfrageklassifizierung, um hochpriorisierten Traffic durch dedizierte Bulkhead-Partitionen zu leiten
- Fügen Sie Monitoring und Alerting für jede Bulkhead-Partition hinzu, um zu erkennen, wenn eine sich der Kapazitätsgrenze nähert

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Begrenzt Fehler auf eine einzelne Partition und verhindert kaskadierende Ausfälle im gesamten System
- Stellt sicher, dass kritische Funktionen verfügbar bleiben, selbst wenn nicht-kritische Komponenten ausfallen
- Bietet klarere Sichtbarkeit der Ressourcennutzung pro Systemfunktion
- Ermöglicht unabhängige Skalierung verschiedener Systempartitionen

**Kosten und Risiken:**
- Erhöht den Gesamtressourcenverbrauch, da jede Partition ihre eigene reservierte Kapazität benötigt
- Fügt Konfigurationskomplexität für die Verwaltung mehrerer Pools und Partitionsgrenzen hinzu
- Unterversorgte Partitionen könnten legitimen Traffic drosseln, während andere Partitionen untätig sind
- Erfordert sorgfältige Analyse, um Partitionsgrenzen an den richtigen Stellen zu ziehen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Online-Einzelhandelsplattform erlebte vollständige Ausfälle, wann immer ihre Empfehlungs-Engine aufgrund von Drittanbieter-API-Timeouts langsam wurde. Der Empfehlungsservice teilte einen Thread-Pool mit dem Checkout-Flow, sodass, wenn Empfehlungs-Threads blockierten, sich Checkout-Anfragen aufstauten und die gesamte Website nicht mehr reagierte. Das Team führte separate Thread-Pools für Checkout, Empfehlungen und Bestandsoperationen ein. Als sich die Empfehlungs-API verlangsamte, degradierten nur Empfehlungen, während der Checkout weiterhin normal Bestellungen verarbeitete. Diese einzelne Änderung eliminierte die häufigste Ursache ihrer websiteweiten Ausfälle.
