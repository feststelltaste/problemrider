---
title: Serverless Computing
description: Ausführung von Code ohne Verwaltung der zugrunde liegenden
  Infrastruktur.
category:
- Operations
- Architecture
problems:
- scaling-inefficiencies
- operational-overhead
- complex-deployment-process
- capacity-mismatch
- high-maintenance-costs
- poor-system-environment
layout: solution
lang: de
en_slug: serverless-computing
related_solutions:
- slug: cloud-native-development
  similarity: 0.75
- slug: containerization
  similarity: 0.75
- slug: distributed-processing
  similarity: 0.75
- slug: microservices-architecture
  similarity: 0.7
- slug: elastic-resource-utilization
  similarity: 0.7
- slug: load-balancing
  similarity: 0.7
---

## Description

Serverless Computing führt Code als Reaktion auf Events aus, ohne dass der Aufrufer irgendeinen zugrunde liegenden Server verwaltet, skaliert automatisch mit der Nachfrage und rechnet nur die tatsächliche Ausführungszeit ab, statt bereitgestellte Kapazität, die meistens ungenutzt bleibt. Für Legacy-Modernisierung liegt der praktische Wert üblicherweise nicht in der Migration des gesamten Systems, sondern in der Auslagerung spezifischer, gut abgegrenzter Workloads — Bildverarbeitung, Berichtsgenerierung, Webhook-Verarbeitung — aus einem Monolithen heraus und auf Serverless-Funktionen, selektiv über ein API-Gateway geroutet, während der Großteil der Legacy-Anwendung unverändert weiterläuft. Diese gezielte Extraktion entlastet genau die Art von Ressourcenkonkurrenz, die auftritt, wenn eine CPU-intensive, bedarfsgesteuerte Aufgabe innerhalb eines Monolithen die Performance für jeden anderen gleichzeitigen Nutzer verschlechtert, ohne dass der Rest des Legacy-Systems überhaupt angefasst werden muss. Die Kompromisse sind real, allerdings: Cold-Start-Latenz kann für latenzsensible Operationen schlecht passen, Zustand muss externalisiert werden, da Funktionen zustandslos sind, und der Ansatz führt eine neue Form von Vendor-Lock-in zu welcher Cloud-Plattform auch immer die Funktionen hostet ein.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie für Serverless geeignete Workloads: eventgesteuerte Aufgaben, periodische Batch-Jobs, Webhook-Handler, API-Endpunkte mit variablem Traffic
- Beginnen Sie mit der Auslagerung peripherer Funktionen (Bildverarbeitung, E-Mail-Versand, Berichtsgenerierung) vom Legacy-Monolithen zu Serverless-Funktionen
- Nutzen Sie API-Gateways, um spezifische Endpunkte zu Serverless-Funktionen zu routen, während der restliche Traffic weiter zur Legacy-Anwendung geht
- Refaktorieren Sie zustandsbehaftete Operationen, um Zustand zu verwalteten Diensten (Datenbanken, Caches, Warteschlangen) zu externalisieren, da Serverless-Funktionen zustandslos sind
- Implementieren Sie angemessene Timeout- und Retry-Strategien angesichts der Ausführungsgrenzen von Serverless-Plattformen
- Überwachen Sie Cold-Start-Latenz und optimieren Sie die Funktionsgröße, um Startverzögerungen zu minimieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Beseitigt Serververwaltung und Infrastrukturbereitstellung für migrierte Workloads
- Automatische Skalierung bewältigt Traffic-Spitzen ohne Kapazitätsplanung
- Pay-per-Use-Preisgestaltung reduziert Kosten für intermittierende oder variable Workloads
- Ermöglicht schnelles Deployment neuer Features ohne Infrastrukturänderungen

**Kosten und Risiken:**
- Cold-Start-Latenz kann für latenzsensible Operationen problematisch sein
- Vendor-Lock-in zur Serverless-Plattform eines bestimmten Cloud-Anbieters
- Ausführungszeitgrenzen und Speicherbeschränkungen könnten nicht für alle Workloads passen
- Debugging und Monitoring von Serverless-Funktionen erfordert andere Werkzeuge und Ansätze
- Zustandsverwaltungskomplexität steigt, wenn Funktionen zustandslos sein müssen
- Kosten können traditionelles Hosting bei konstant hohem Traffic übersteigen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Dokumentenmanagementsystem generierte PDF-Berichte auf Anfrage, eine CPU-intensive Operation, die Performance-Probleme für andere Nutzer verursachte, wenn mehrere Berichte gleichzeitig angefragt wurden. Das Team extrahierte die PDF-Generierungslogik in AWS-Lambda-Funktionen, ausgelöst durch eine SQS-Warteschlange. Die Legacy-Anwendung stellte einfach Berichtsanfragen in die Warteschlange und fragte den Abschluss ab. Dies beseitigte die Auswirkung auf die Performance der Hauptanwendung, skalierte automatisch während Monatsend-Berichtsspitzen und reduzierte die Infrastrukturkosten um 70 Prozent, da PDF-Generierung nur Ressourcen verbrauchte, während sie tatsächlich lief.
