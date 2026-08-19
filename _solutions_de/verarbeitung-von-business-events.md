---
title: Verarbeitung von Business Events
description: Erkennung, Verarbeitung und Reaktion auf Geschäftsereignisse.
category:
- Architecture
- Business
problems:
- monolithic-architecture-constraints
- tight-coupling-issues
- legacy-business-logic-extraction-difficulty
- slow-application-performance
- cascade-failures
- complex-and-obscure-logic
layout: solution
lang: de
en_slug: business-event-processing
related_solutions:
- slug: event-driven-integration
  similarity: 0.8
- slug: event-driven-architecture
  similarity: 0.75
- slug: domain-driven-design
  similarity: 0.75
- slug: distributed-processing
  similarity: 0.75
- slug: asynchronous-processing
  similarity: 0.7
- slug: microservices-architecture
  similarity: 0.7
---

## Description

Verarbeitung von Business Events modelliert bedeutsame Vorkommnisse im Betrieb eines Geschäfts — eine Bestellung aufgegeben, eine Zahlung erhalten, eine Sendung versandt — als explizite, benannte Ereignisse, die an einen Event-Bus oder Message-Broker veröffentlicht werden, statt als implizite Schritte, die in einer größeren prozeduralen Transaktion vergraben sind. Der Mechanismus entkoppelt das System, das ein Ereignis erkennt, von jedem System, das darauf reagieren muss: Produzenten veröffentlichen, was geschah, ohne zu wissen oder sich darum zu kümmern, wer zuhört, und Konsumenten abonnieren die für sie relevanten Ereignisse und verarbeiten sie unabhängig, oft asynchron. Dies zielt direkt auf ein in Legacy-Systemen übliches Muster ab, bei dem eine einzelne prozedurale Transaktion mehrere Geschäftsbelange überspannt — Bestand, Abrechnung, Versand — innerhalb einer Alles-oder-nichts-Datenbanktransaktion, sodass ein Fehler irgendwo in der Kette die gesamte Operation zum Scheitern bringt und kein einzelnes Geschäftsereignis jemals allein sichtbar oder nachverfolgbar ist. Einen solchen Legacy-Workflow um explizite Ereignisse herum umzustrukturieren erlaubt es, jeden Belang unabhängig zu handhaben, zu skalieren und sogar zu ersetzen, und es macht Geschäftslogik durch einen Ereignisstrom nachverfolgbar, statt in prozeduralem Code vergraben zu sein, der sein Verhalten nur durch zeilenweises Lesen offenbart. Die Kosten sind, dass sofortige Konsistenz eventueller Konsistenz weicht, auf die sich das Design und die Nutzer eines Legacy-Systems möglicherweise implizit verlassen haben, und ereignisgetriebene Abläufe sind inhärent schwieriger nachzuverfolgen und zu debuggen als ein einzelner synchroner Call-Stack, sodass Kompensationstransaktionen und Monitoring notwendige Ergänzungen statt optionaler werden.

## How to Apply ◆

- Identifizieren Sie Schlüssel-Geschäftsereignisse im Legacy-System (Bestellung aufgegeben, Zahlung erhalten, Sendung versandt) und modellieren Sie sie explizit, statt sie in prozeduralen Abläufen einzubetten.
- Führen Sie einen Event-Bus oder Message-Broker (Kafka, RabbitMQ) ein, um Ereignisproduzenten von Konsumenten in der Legacy-Architektur zu entkoppeln.
- Refaktorieren Sie synchrone, eng gekoppelte Legacy-Workflows schrittweise zu ereignisgetriebenen Abläufen, beginnend mit den problematischsten Integrationspunkten.
- Definieren Sie Ereignisschemata und stellen Sie sicher, dass sie genug Kontext tragen, damit Konsumenten sie unabhängig verarbeiten können.
- Implementieren Sie Event Sourcing für kritische Geschäftsprozesse, wo Prüfpfad und Zustandsrekonstruktion benötigt werden.
- Fügen Sie Monitoring und Alerting für Ereignisverarbeitung hinzu, um Verzögerungen oder Fehler zu erkennen.

## Tradeoffs ⇄

**Vorteile:**
- Entkoppelt Geschäftsprozesse und ermöglicht unabhängige Skalierung und Weiterentwicklung von Produzenten und Konsumenten.
- Macht Geschäftslogik expliziter und durch Ereignisströme nachverfolgbar.
- Ermöglicht Echtzeitreaktionen auf Geschäftsereignisse, die Legacy-Batch-Verarbeitung nicht unterstützen kann.
- Erleichtert die schrittweise Zerlegung monolithischer Legacy-Systeme.

**Kosten:**
- Führt eventuelle Konsistenz ein, die Legacy-Systeme, die für sofortige Konsistenz designt wurden, möglicherweise nicht gut handhaben.
- Ereignisgetriebene Architekturen sind schwieriger zu debuggen und nachzuvollziehen als synchrone Aufrufketten.
- Erfordert Infrastruktur für verlässliche Ereigniszustellung und -verarbeitung.
- Die Nachrüstung von Ereignisverarbeitung in ein Legacy-System erfordert sorgfältige Identifikation impliziter Ereignisse.

## How It Could Be

Ein Legacy-Einzelhandelssystem verarbeitet Bestellungen durch eine monolithische Transaktion, die Bestand, Abrechnung und Versand in einer einzigen Datenbanktransaktion überspannt. Wenn irgendein Schritt fehlschlägt, scheitert die gesamte Bestellung. Das Team führt einen ereignisgetriebenen Ansatz ein: Die Bestellaufgabe sendet ein „OrderCreated"-Ereignis, und Bestand, Abrechnung und Versand abonnieren jeweils unabhängig. Jeder Service handhabt seinen Teil asynchron und sendet sein eigenes Abschlussereignis. Kompensationstransaktionen handhaben Fehler. Diese Entkopplung erlaubt es, das Versandmodul durch eine neue Implementierung zu ersetzen, ohne den Abrechnungscode zu berühren, und das System kann Spitzenlasten handhaben, indem es Ereignisse puffert statt Bestellungen abzulehnen.
