---
title: API-Gateway
description: Zentralisierung von Protokollübersetzung, Versionierung und Routing
  über einen einzigen Einstiegspunkt.
category:
- Architecture
- Operations
problems:
- legacy-api-versioning-nightmare
- api-versioning-conflicts
- microservice-communication-overhead
- poor-interfaces-between-applications
- single-entry-point-design
- high-api-latency
- rate-limiting-issues
- service-discovery-failures
- graphql-complexity-issues
layout: solution
lang: de
en_slug: api-gateway
related_solutions:
- slug: protocol-abstraction
  similarity: 0.8
- slug: adapter
  similarity: 0.8
- slug: api-deprecation-policy
  similarity: 0.75
- slug: containerization
  similarity: 0.7
- slug: event-driven-integration
  similarity: 0.7
- slug: api-first-development
  similarity: 0.7
---

## Description

Ein API-Gateway ist ein einziger Einstiegspunkt, der vor einem oder mehreren Backend-Services platziert wird und Belange wie Protokollübersetzung, Anfrage-Routing, Versionierung, Authentifizierung und Ratenbeschränkung zentralisiert, sodass Konsumenten mit einer konsistenten Schnittstelle interagieren, unabhängig davon, wie heterogen oder fragmentiert die dahinterliegenden Services tatsächlich sind. In Legacy-Umgebungen ist dies häufig der schnellste Weg, alte, schwer konsumierbare Schnittstellen — zum Beispiel eine Sammlung alternder SOAP-Services — für moderne Clients nutzbar zu machen, ohne die Legacy-Implementierungen überhaupt zu berühren, weil das Gateway Protokollübersetzung (wie SOAP zu REST oder XML zu JSON) durchführen und eine saubere, modern aussehende API nach außen präsentieren kann. Das Platzieren von Querschnittsbelangen wie Authentifizierung und Logging am Gateway beseitigt außerdem die Notwendigkeit, sie konsistent innerhalb jedes Legacy-Services neu zu implementieren, von denen viele über die Jahre ihre eigenen inkompatiblen Ad-hoc-Versionen dieser Belange entwickelt haben könnten. Weil das Gateway zur einzigen Naht zwischen Konsumenten und dem wird, was dahinter läuft, ermöglicht es inkrementelle Backend-Migration: Ein Service hinter dem Gateway kann ersetzt oder neu geschrieben werden, ohne irgendetwas am konsumentenseitigen Vertrag zu ändern, solange die Routing- und Transformationsregeln des Gateways entsprechend aktualisiert werden. Diese Konzentration von Verantwortung ist auch das Hauptrisiko des Gateways, da es zu einem Single Point of Failure wird, der für hohe Verfügbarkeit gebaut werden muss, und wenn unkontrolliert gelassen, kann es Geschäftslogik anhäufen, die eigentlich zu den Services selbst gehört statt zur Routing-Schicht.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Deployen Sie ein API-Gateway vor Legacy-Services, um einen einheitlichen Einstiegspunkt für alle Konsumenten bereitzustellen
- Nutzen Sie das Gateway zur Handhabung von Protokollübersetzung (z. B. SOAP zu REST), sodass Legacy-Backends unberührt bleiben
- Implementieren Sie API-Versionierung auf der Gateway-Ebene und leiten Sie Anfragen zur passenden Backend-Version weiter
- Fügen Sie Querschnittsbelange wie Authentifizierung, Ratenbeschränkung und Logging am Gateway statt in jedem Service hinzu
- Nutzen Sie das Gateway zur Aggregation von Antworten mehrerer Legacy-Services in eine einzige konsumentenfreundliche Antwort
- Beginnen Sie mit einer Pass-Through-Konfiguration und fügen Sie inkrementell Transformationsregeln hinzu

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Entkoppelt konsumentenseitiges API-Design von Legacy-Backend-Schnittstellen
- Zentralisiert Querschnittsbelange, was Duplizierung über Services hinweg verringert
- Ermöglicht inkrementelle Backend-Migration, ohne konsumentenseitige Verträge zu ändern
- Bietet einen einzigen Punkt für Monitoring und Traffic-Management

**Kosten und Risiken:**
- Das Gateway wird zu einem Single Point of Failure, wenn es nicht ordentlich für hohe Verfügbarkeit designt ist
- Kann Latenz durch zusätzliche Netzwerk-Hops und Transformations-Overhead einführen
- Komplexe Routing-Regeln können über die Zeit schwer zu verwalten und zu debuggen werden
- Risiko, dass das Gateway Geschäftslogik anhäuft, die eigentlich zu Services gehört

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Telekommunikationsunternehmen hatte Dutzende Legacy-SOAP-Services, mit denen sich mobile App-Teams schwertaten. Durch das Platzieren eines API-Gateways vor diese Services exponierte das Team saubere REST-Endpunkte, während die SOAP-Backends unverändert weiterliefen. Das Gateway handhabte XML-zu-JSON-Übersetzung, Anfrage-Routing basierend auf API-Versionsheadern und zentralisierte Authentifizierung. Dies erlaubte dem Mobile-Team, gegen moderne APIs zu bauen, während das Backend-Team inkrementelle Service-Ersätze über das folgende Jahr plante.
