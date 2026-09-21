---
title: Isolation fehlerhafter Komponenten
description: Entwicklung von Mechanismen zur Isolation fehlerhafter Komponenten.
category:
- Architecture
problems:
- cascade-failures
- single-points-of-failure
- tight-coupling-issues
- monolithic-architecture-constraints
- system-outages
- unpredictable-system-behavior
layout: solution
lang: de
en_slug: isolation-of-faulty-components
related_solutions:
- slug: fault-containment
  similarity: 0.8
- slug: failover-mechanisms
  similarity: 0.75
- slug: bulkhead
  similarity: 0.75
- slug: resilience
  similarity: 0.75
- slug: circuit-breaker
  similarity: 0.7
- slug: rate-limiting
  similarity: 0.7
---

## Description

Isolation fehlerhafter Komponenten ist die Praxis, Eindämmungsgrenzen um Teile eines Systems zu errichten, sodass ein Ausfall in einem Teil sich nicht auf den Rest ausbreitet. Mechanisch stützt sich dies auf Techniken wie Circuit Breaker, die aufhören, eine ausfallende Abhängigkeit aufzurufen, Bulkheads, die Thread-Pools und Verbindungen pro Komponente partitionieren, Prozess- oder Container-Isolation, die verhindert, dass sich Ressourcenerschöpfung ausbreitet, und automatische Erkennungsauslöser basierend auf Health Checks und Fehlerraten, die entscheiden, wann eine Komponente abgeschnitten werden sollte. In Legacy-Systemen wurden Komponenten selten mit Fehlereindämmung im Blick entworfen — enge Kopplung, gemeinsamer Speicherbereich und gemeinsame Verbindungspools bedeuten, dass ein einzelnes überlastetes oder fehlfunktionierendes Modul Ressourcen erschöpfen oder Zustand für alles andere, was daneben läuft, beschädigen kann, was einen lokalen Defekt in einen kaskadierenden, systemweiten Ausfall verwandelt. Isolationsgrenzen in ein solches System nachzurüsten behebt nicht den zugrundeliegenden Fehler, aber es verändert den Ausfallmodus von vollständigem Zusammenbruch zu einem degradierten, teilweise verfügbaren System, was Zeit verschafft, um den tatsächlichen Defekt zu diagnostizieren und zu beheben. Dies ist besonders wertvoll während der Modernisierung, wo Legacy-Komponenten schrittweise ersetzt oder erdrosselt werden: Isolation erlaubt Teams, eine alte, fragile Komponente als unter Quarantäne stehende Einheit zu behandeln, deren Ausfälle erwartet und eingedämmt werden, statt als Landmine, die niemals auslösen darf.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Implementieren Sie Circuit-Breaker-Muster an Integrationspunkten, um das Aufrufen ausfallender Komponenten zu stoppen
- Nutzen Sie Prozessisolation oder Containerisierung, um zu verhindern, dass eine fehlerhafte Komponente gemeinsame Ressourcen verbraucht
- Führen Sie Bulkhead-Muster ein, um Thread-Pools und Verbindungspools pro Komponente zu trennen
- Entwerfen Sie automatische Erkennungs- und Isolationsauslöser basierend auf Fehlerraten, Antwortzeiten oder Health Checks
- Erstellen Sie Fallback-Antworten für den Fall, dass eine Komponente isoliert wird, damit abhängige Dienste weiterarbeiten können
- Protokollieren und alarmieren Sie bei Isolationsereignissen, um sicherzustellen, dass Betriebsteams die Grundursache umgehend untersuchen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verhindert, dass eine einzelne fehlerhafte Komponente das gesamte System lahmlegt
- Erlaubt gesunden Teilen des Systems, weiterhin Nutzer zu bedienen
- Bietet klare Signale darüber, welche Komponente ausfällt
- Ermöglicht unabhängige Wiederherstellung und Neustart isolierter Komponenten

**Kosten und Risiken:**
- Isolationsmechanismen fügen der Systemarchitektur Komplexität hinzu
- Aggressive Isolation kann bei vorübergehenden Netzwerkproblemen falsch-positive Ergebnisse verursachen
- Legacy-Monolithen könnten erhebliches Refactoring benötigen, um Komponentenisolation zu unterstützen
- Isolierte Komponenten können abhängige Workflows in einem unvollständigen Zustand hinterlassen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Reisebuchungsplattform erlebte vollständige Ausfälle, wann immer ihre Legacy-Preis-Engine während Flash-Sales überlastet wurde. Durch das Umhüllen von Aufrufen an die Preis-Engine mit einem Circuit Breaker und das Ausliefern zwischengespeicherter Preise, wenn der Circuit öffnete, isolierte das Team die fehlerhafte Komponente, während der Rest des Buchungsflusses betriebsfähig blieb. Nutzer konnten weiterhin zu den zuletzt bekannten Preisen browsen und buchen, und der Preis-Engine wurde Zeit gegeben, sich zu erholen, ohne den zusätzlichen Druck aufgestauter Anfragen.
