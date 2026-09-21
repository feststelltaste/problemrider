---
title: Vertikale Skalierung
description: Steigerung der Performance einzelner Komponenten.
category:
- Performance
- Operations
problems:
- slow-application-performance
- capacity-mismatch
- scaling-inefficiencies
- slow-database-queries
- high-database-resource-utilization
- gradual-performance-degradation
layout: solution
lang: de
en_slug: vertical-scaling
related_solutions:
- slug: horizontal-scaling
  similarity: 0.85
- slug: distributed-caching
  similarity: 0.75
- slug: data-replication
  similarity: 0.7
- slug: load-balancing
  similarity: 0.7
- slug: specialized-hardware
  similarity: 0.7
- slug: denormalization
  similarity: 0.7
---

## Description

Vertikale Skalierung erhöht die Kapazität einer einzelnen Komponente — mehr CPU-Kerne, mehr Speicher, schnellerer Speicherplatz —, statt Last über zusätzliche Instanzen zu verteilen, und erfordert keine Änderungen am Anwendungscode, was sie zum schnellsten verfügbaren Hebel macht, um einen Performance-Engpass zu lindern. Ihr Reiz in Legacy-Kontexten liegt genau darin, dass sie angewendet werden kann, ohne Software anzufassen, die niemand mehr vollständig versteht: Eine Legacy-Anwendung, deren Architektur ein Single-Instance-Deployment annimmt, oder die einfach nicht sicher für horizontale Skalierung refaktoriert werden kann in irgendeinem vernünftigen Zeitrahmen, kann oft immer noch bedeutsam mehr Spielraum bekommen, nur indem die Hardware oder Infrastruktur aufgerüstet wird, auf der sie bereits läuft. Der Mechanismus zahlt sich jedoch nur aus, wenn der tatsächliche Engpass zuerst korrekt diagnostiziert wird — CPU, Speicher, I/O oder Netzwerk —, da das Hinzufügen zusätzlicher Ressourcen zu einer Komponente, die nicht die echte Einschränkung ist, nichts erreicht. Da vertikale Skalierung eine harte Obergrenze hat, bestimmt durch verfügbare Hardware, und eine nichtlineare Kostenkurve, während Instanzen größer werden, funktioniert sie am besten als bewusste kurzfristige Maßnahme, die Zeit und Luft zum Atmen für ein Legacy-System unter akutem Performance-Druck kauft, statt als dauerhafter Ersatz für die Adressierung der architektonischen Engpässe, die die Kapazität jeder einzelnen Instanz begrenzen. So genutzt verwandelt sie eine dringende Kapazitätskrise in eine handhabbare, was dem Team den Raum gibt, einen strukturelleren Fix zu planen — Partitionierung, horizontale Skalierung oder eine Architekturänderung —, ohne dass diese Planung unter Notfallbedingungen geschieht.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Profilen Sie das System, um zu bestimmen, ob der Engpass CPU, Speicher, I/O oder Netzwerk ist, bevor Sie Hardware aufrüsten
- Erhöhen Sie Serverressourcen (CPU-Kerne, RAM, schnellerer Speicherplatz) für die als Einschränkung identifizierte Komponente
- Rüsten Sie Datenbankserver mit mehr Speicher auf, um Working Sets gecacht zu halten und Festplatten-I/O zu reduzieren
- Ersetzen Sie HDD durch SSD- oder NVMe-Speicher für I/O-gebundene Legacy-Anwendungen und -Datenbanken
- Stimmen Sie Anwendungs- und Datenbankserverkonfigurationen ab, um zusätzliche Ressourcen zu nutzen (Thread-Pools, Buffer-Pools, Heap-Größen)
- Nutzen Sie vertikale Skalierung als kurzfristige Maßnahme, um Zeit zu kaufen, während horizontale Skalierung oder architektonische Verbesserungen geplant werden
- Dokumentieren Sie die Skalierungsobergrenze für die aktuelle Architektur, sodass das Team weiß, wann vertikale Skalierung nicht mehr ausreicht

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Einfachster Skalierungsansatz, der keine Anwendungscodeänderungen erfordert
- Sofort wirksam für Legacy-Anwendungen, die nicht horizontal skaliert werden können
- Erhält das bestehende Single-Instance-Deployment-Modell und vermeidet verteilte Systemkomplexität
- Oft der schnellste Weg zur Lösung einer akuten Performance-Krise

**Kosten und Risiken:**
- Harte Obergrenze für vertikale Skalierung, bestimmt durch verfügbare Hardware
- Größere Instanzen sind unverhältnismäßig teuer (nichtlineare Kostenkurve)
- Adressiert keine architektonischen Engpässe, die die Single-Instance-Performance begrenzen
- Kann zugrunde liegende Probleme verschleiern und notwendiges Refactoring verzögern
- Schafft einen Single Point of Failure mit höherem Blast-Radius, während sich mehr Last auf einer Maschine konzentriert

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Buchhaltungssystem, das auf einem Server mit 16 GB RAM und rotierenden Festplatten lief, erlebte schwere Performance-Degradation, während die Transaktionsdatenbank über 500 GB wuchs. Die Analyse zeigte, dass der Datenbank-Buffer-Pool nur 20 Prozent des Working Sets cachen konnte, was konstantes Festplatten-I/O verursachte. Das Team rüstete den Server auf 128 GB RAM und NVMe-Speicher auf. Datenbankabfragezeiten verbesserten sich um das 10-Fache, und der Monatsabschlussprozess, der sich auf 14 Stunden ausgedehnt hatte, wurde in 90 Minuten abgeschlossen. Das Team nutzte den Performance-Spielraum, um eine Datenbankpartitionierungsstrategie für den Zeitpunkt zu planen, an dem der Datensatz sogar die Kapazität des aufgerüsteten Servers überschreiten würde.
