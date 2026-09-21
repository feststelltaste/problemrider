---
title: Ereignisgetriebene Integration
description: Entkopplung von Produzenten und Konsumenten über asynchrone Message-Broker-Kommunikation.
category:
- Architecture
problems:
- tight-coupling-issues
- high-coupling-low-cohesion
- monolithic-architecture-constraints
- integration-difficulties
- microservice-communication-overhead
- cross-system-data-synchronization-problems
- deployment-coupling
layout: solution
lang: de
en_slug: event-driven-integration
related_solutions:
- slug: event-driven-architecture
  similarity: 0.8
- slug: business-event-processing
  similarity: 0.8
- slug: asynchronous-processing
  similarity: 0.75
- slug: protocol-abstraction
  similarity: 0.75
- slug: adapter
  similarity: 0.7
- slug: api-gateway
  similarity: 0.7
---

## Description

Ereignisgetriebene Integration ersetzt direkte, synchrone Aufrufe zwischen Systemen durch asynchrone Nachrichten, die an einen Broker veröffentlicht und von ihm konsumiert werden, sodass Produzenten unveränderliche Tatsachen darüber ausgeben, was geschehen ist, statt Befehle an Konsumenten zu erteilen, die jetzt sofort verfügbar und reaktionsfähig sein müssen. Dies entkoppelt die beiden Seiten sowohl zeitlich als auch räumlich: Ein Konsument, der ausgefallen, langsam oder noch nicht gebaut ist, blockiert den Producer nicht, und der Broker puffert Nachrichten, bis der Konsument aufholt. In Legacy-Systemen, die um lange Ketten synchroner Aufrufe zwischen Komponenten herum gebaut sind, ist diese Kopplung oft die direkte Ursache kaskadierender Fehler, bei denen ein langsamer oder nicht verfügbarer nachgelagerter Service die gesamte Anfrage verschlechtert oder bricht, und der Schwierigkeit, neue Konsumenten hinzuzufügen, ohne den Code des Producers anzufassen. Einen Broker wie Kafka oder RabbitMQ an den wertvollsten Integrationspunkten einzuführen, und dies schrittweise statt auf einmal zu tun, erlaubt einem Team, diese Kopplung graduell aufzubrechen, während der Legacy-Producer weitgehend intakt bleibt, obwohl es auch sofortige Konsistenz gegen Eventual Consistency eintauscht und neue operative Fläche einführt — Broker-Infrastruktur, Dead Letter Queues, Nachrichtenreihenfolge —, mit der sich synchrone Aufrufe nie befassen mussten.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie synchrone Integrationspunkte zwischen Legacy-Systemen, die Kopplungs- oder Zuverlässigkeitsprobleme verursachen
- Führen Sie einen Message Broker (z. B. Kafka, RabbitMQ) ein und lassen Sie Producer Domain Events ausgeben, statt direkte Aufrufe zu tätigen
- Gestalten Sie Events als unveränderliche Tatsachen darüber, was geschehen ist, nicht als Befehle für das, was geschehen sollte
- Fügen Sie Legacy-Systemen schrittweise Event-Veröffentlichung hinzu, beginnend mit den wertvollsten oder schmerzhaftesten Integrationspunkten
- Implementieren Sie idempotente Konsumenten, um erneute Nachrichtenzustellung graziös zu handhaben
- Nutzen Sie Event-Schemata mit einer Registry, um Kompatibilität aufrechtzuerhalten, während sich Events weiterentwickeln

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Entkoppelt Systeme zeitlich und räumlich: Producer und Konsumenten müssen nicht gleichzeitig verfügbar sein
- Ermöglicht das Hinzufügen neuer Konsumenten, ohne den Producer zu modifizieren, was schrittweise Modernisierung unterstützt
- Verbessert die Resilienz, indem Nachrichten während Konsumentenausfallzeiten gepuffert werden

**Kosten und Risiken:**
- Führt Eventual Consistency ein, was für Workflows herausfordernd sein kann, die sofortige Datenverfügbarkeit erwarten
- Fügt operative Komplexität durch Broker-Infrastruktur, Überwachung und Dead-Letter-Queue-Verwaltung hinzu
- Das Debuggen asynchroner Abläufe ist schwerer als das Verfolgen synchroner Anfrage-Antwort-Ketten
- Nachrichtenreihenfolge und Exactly-once-Zustellungsgarantien variieren je nach Broker und erfordern sorgfältiges Design

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Auftragsverwaltungssystem tätigte während der Auftragsverarbeitung synchrone HTTP-Aufrufe an fünf nachgelagerte Services. Wenn ein nachgelagerter Service langsam oder nicht verfügbar war, schlugen Bestellungen fehl. Das Team führte Kafka als Event-Broker ein, wobei das Bestellsystem OrderPlaced-Events veröffentlichte. Jeder nachgelagerte Service konsumierte Events unabhängig und in eigenem Tempo. Auftragsverarbeitungsfehler sanken von 5 Prozent auf unter 0,1 Prozent, und das Team konnte später einen neuen Analytics-Konsumenten hinzufügen, ohne das Bestellsystem überhaupt anzufassen.
