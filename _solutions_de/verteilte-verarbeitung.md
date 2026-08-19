---
title: Verteilte Verarbeitung
description: Aufteilung der Verarbeitung auf mehrere unabhängige Systeme.
category:
- Performance
- Architecture
problems:
- scaling-inefficiencies
- slow-application-performance
- single-points-of-failure
- capacity-mismatch
- monolithic-architecture-constraints
layout: solution
lang: de
en_slug: distributed-processing
related_solutions:
- slug: parallelization
  similarity: 0.85
- slug: pipelining
  similarity: 0.8
- slug: data-replication
  similarity: 0.8
- slug: load-balancing
  similarity: 0.8
- slug: distributed-caching
  similarity: 0.8
- slug: data-partitioning
  similarity: 0.8
---

## Description

Verteilte Verarbeitung zerlegt eine Rechen-Workload in unabhängige Arbeitseinheiten, die gleichzeitig über mehrere Maschinen statt sequenziell auf einer ausgeführt werden, mittels eines Arbeitsverteilungs-Frameworks wie MapReduce, Spark oder einer Task-Queue mit Worker-Pools, um die Ausführung zu koordinieren und Ergebnisse zu aggregieren. Dies ist speziell wichtig, wo die Gesamtgröße einer Workload das übertroffen hat, was eine einzelne Maschine in akzeptabler Zeit leisten kann — eine häufige Situation für Legacy-Batch-Jobs, Berichte und Simulationen, die vor Jahrzehnten für die Datenvolumina jener Ära entworfen wurden und nicht neu architektiert wurden, während diese Volumina wuchsen. Weil Legacy-Verarbeitungspipelines häufig als ein großer sequenzieller Durchlauf über die Daten geschrieben sind, erfordert ihre Verteilung zunächst zu identifizieren, welche Teile der Pipeline tatsächlich unabhängig sind, und die Pipeline entsprechend zu zerlegen, zusammen mit der Idempotenz einzelner Arbeitseinheiten, sodass eine fehlgeschlagene Aufgabe einfach auf einem anderen Knoten wiederholt werden kann. Der Gewinn ist, dass die Verarbeitungszeit ungefähr proportional zur Anzahl der auf das Problem angewandten Knoten sinkt, und die resultierende Architektur gewinnt auch Fehlertoleranz, da der Ausfall eines Knotens nicht mehr einen ganzen Lauf ungültig macht. Dies kommt auf Kosten von Komplexität verteilter Systeme — Netzwerkausfälle, partielle Ausfälle und Koordinationsoverhead —, mit denen ein Einzelmaschinen-Batch-Job nie umgehen musste, und nicht jede Legacy-Verarbeitungsaufgabe kann auf diese Weise zerlegt werden, da manche Berechnungen inhärent sequenziell sind.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie Workloads, die parallelisierbar sind: Datenverarbeitung, Berichtserstellung, Batch-Berechnungen
- Zerlegen Sie monolithische Verarbeitungspipelines in unabhängige Arbeitseinheiten, die auf separaten Knoten ausgeführt werden können
- Nutzen Sie ein für den Workload-Typ geeignetes Arbeitsverteilungs-Framework (MapReduce, Spark, Task-Queues mit Workern)
- Implementieren Sie idempotente Verarbeitung, sodass fehlgeschlagene Aufgaben sicher auf anderen Knoten wiederholt werden können
- Gestalten Sie für partiellen Ausfall: Einzelne Knotenausfälle sollten den gesamten Verarbeitungslauf nicht ungültig machen
- Beginnen Sie damit, die ressourcenintensivsten Verarbeitungsjobs zu verteilen, während einfachere zentralisiert bleiben
- Überwachen Sie die Verarbeitungsverteilung, um Hotspots zu erkennen, an denen Arbeit ungleich verteilt ist

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Ermöglicht die Verarbeitung von Workloads, die die Kapazität einer einzelnen Maschine übersteigen
- Bietet Fehlertoleranz durch Redundanz über mehrere Knoten
- Erlaubt lineare Skalierung durch Hinzufügen weiterer Verarbeitungsknoten
- Reduziert die Verarbeitungszeit für parallelisierbare Workloads proportional zur Anzahl der Knoten

**Kosten und Risiken:**
- Führt Komplexität verteilter Systeme ein: Netzwerkausfälle, partielle Ausfälle und Koordinationsoverhead
- Nicht alle Workloads sind parallelisierbar; manche erfordern sequenzielle Verarbeitung
- Datentransfer zwischen Knoten kann zu einem Engpass werden, wenn nicht verwaltet
- Das Debuggen verteilter Verarbeitungsfehler ist erheblich schwieriger als das Debuggen lokaler Verarbeitung

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Risikomodellierungssystem für ein Versicherungsunternehmen führte Monte-Carlo-Simulationen auf einem einzelnen Server aus und brauchte über 18 Stunden, um eine vollständige Portfolio-Risikobewertung abzuschließen. Das Geschäft brauchte täglich Ergebnisse, aber der Einzelserver-Ansatz war an seinen Grenzen. Das Team refaktorierte die Simulations-Engine, um unabhängige Simulationsläufe über einen Cluster von Worker-Knoten mittels einer Task-Queue zu verteilen. Jeder Worker verarbeitete einen Batch von Szenarien und meldete Ergebnisse an einen Aggregationsdienst zurück. Die vollständige Risikobewertung ist jetzt in unter zwei Stunden auf einem 12-Knoten-Cluster abgeschlossen, und das Unternehmen kann mehrere Bewertungen pro Tag mit unterschiedlichen Parametern ausführen.
