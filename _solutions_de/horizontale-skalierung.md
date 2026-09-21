---
title: Horizontale Skalierung
description: Steigerung der Performance durch Hinzufügen weiterer Komponenten.
category:
- Performance
- Operations
problems:
- scaling-inefficiencies
- capacity-mismatch
- single-points-of-failure
- slow-application-performance
- load-balancing-problems
- monolithic-architecture-constraints
layout: solution
lang: de
en_slug: horizontal-scaling
related_solutions:
- slug: vertical-scaling
  similarity: 0.85
- slug: elastic-resource-utilization
  similarity: 0.8
- slug: load-balancing
  similarity: 0.75
- slug: elastic-scaling
  similarity: 0.75
- slug: distributed-caching
  similarity: 0.75
- slug: data-replication
  similarity: 0.7
---

## Description

Horizontale Skalierung erhöht die Kapazität eines Systems, indem mehr Instanzen davon hinter einem Load Balancer laufen, statt eine einzelne Instanz leistungsfähiger zu machen, und sie erreicht nahezu lineares Kapazitätswachstum, solange jede Instanz jede eingehende Anfrage austauschbar bearbeiten kann. Diese letzte Bedingung ist der Kern der Relevanz des Musters für die Legacy-Modernisierung: Viele Legacy-Anwendungen halten Sitzungszustand, Datei-Caches oder geplante Task-Zustände im Speicher eines einzelnen Servers, was sie grundlegend unfähig macht, als mehrere austauschbare Instanzen zu laufen, egal wie viel Load-Balancing-Infrastruktur darum herum hinzugefügt wird. Horizontale Skalierung auf ein solches System anzuwenden beginnt daher typischerweise mit einem vorausgesetzten Refactoring-Schritt — Externalisierung der Sitzungsspeicherung in einen gemeinsamen Cache, Verschiebung hochgeladener Dateien in Objektspeicher, Entfernung instanzgebundener geplanter Jobs —, bevor die eigentliche Skalierungsinfrastruktur aus Load Balancern, Health Checks und Auto-Scaling-Richtlinien überhaupt Nutzen bieten kann. Sobald diese Zustandslosigkeit erreicht ist, bietet horizontale Skalierung zwei Vorteile gleichzeitig: Kapazität, die mit Commodity-Instanzen statt zunehmend teurer vertikaler Hardware-Upgrades wächst, und verbesserte Verfügbarkeit, da der Ausfall einer einzelnen Instanz nicht mehr den gesamten Dienst lahmlegt. Der Zielkonflikt ist, dass die Datenbank und andere gemeinsam genutzte Ressourcen, von denen alle Instanzen abhängen, zum neuen Engpass werden können, sobald die Anwendungsschicht selbst nicht mehr die Einschränkung ist, sodass horizontale Skalierung der Anwendungsschicht selten allein eine vollständige Antwort ist.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Machen Sie die Anwendung zustandslos, damit jede Instanz jede Anfrage bearbeiten kann: Externalisieren Sie Sitzungen, Caches und Dateispeicher
- Identifizieren und beseitigen Sie instanzspezifischen Zustand wie lokale Datei-Caches, In-Memory-Sitzungsspeicher und instanzgebundene geplante Tasks
- Setzen Sie einen Load Balancer ein, um Verkehr über mehrere Anwendungsinstanzen zu verteilen
- Implementieren Sie Health Checks, damit der Load Balancer ungesunde Instanzen erkennen und umgehen kann
- Nutzen Sie Auto-Scaling-Richtlinien basierend auf Metriken (CPU, Anfragenzahl, Queue-Tiefe), um Kapazität dynamisch hinzuzufügen
- Testen Sie die Anwendung unter Last mit mehreren Instanzen, um korrektes Verhalten ohne gemeinsam genutzten veränderlichen Zustand zu verifizieren
- Adressieren Sie Datenbankskalierung separat: Lese-Replikate, Connection Pooling oder Sharding nach Bedarf

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Bietet nahezu lineare Kapazitätssteigerung durch Hinzufügen weiterer Instanzen
- Verbessert die Verfügbarkeit: Einzelne Instanzausfälle legen nicht das gesamte System lahm
- Ermöglicht kosteneffiziente Skalierung, indem Kapazität nur hinzugefügt wird, wenn die Nachfrage es erfordert
- Nutzt Commodity-Hardware statt zunehmend teurer vertikaler Upgrades zu erfordern

**Kosten und Risiken:**
- Erfordert, dass Anwendungen zustandslos sind, was bei Legacy-Systemen oft nicht der Fall ist
- Datenbank und gemeinsam genutzte Ressourcen können zu Engpässen werden, die den Nutzen horizontaler Skalierung begrenzen
- Fügt Infrastrukturkomplexität hinzu: Load Balancer, Service Discovery, Instanzverwaltung
- Verteilte Koordinationsprobleme (Cache-Kohärenz, Leader-Wahl) nehmen mit der Instanzzahl zu

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-Webanwendung für ein Universitäts-Einschreibungssystem lief auf einem einzigen großen Server. Während Einschreibungsphasen wurde der Server von Verkehrsspitzen überwältigt, was zu Ausfällen zum ungünstigsten Zeitpunkt führte. Die Anwendung speicherte Sitzungsdaten im Server-Speicher, was das Laufen mehrerer Instanzen verhinderte. Das Team externalisierte die Sitzungsspeicherung nach Redis, verschob hochgeladene Dateien in Objektspeicher und deployte die Anwendung hinter einem Load Balancer mit drei Instanzen. Während der nächsten Einschreibungsphase fügte Auto-Scaling zwei weitere Instanzen hinzu, um die Spitze zu bewältigen, und das System blieb durchgehend reaktionsschnell. Nach der Spitze wurden Instanzen zurückskaliert, um Kosten zu senken.
