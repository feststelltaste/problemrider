---
title: Optimierung von API-Aufrufen
description: Effiziente Gestaltung von API-Aufrufen.
category:
- Performance
- Architecture
problems:
- high-api-latency
- high-number-of-database-queries
- n-plus-one-query-problem
- slow-application-performance
- high-client-side-resource-consumption
- rest-api-design-issues
- network-latency
layout: solution
lang: de
en_slug: api-calls-optimization
related_solutions:
- slug: image-and-asset-optimization
  similarity: 0.75
- slug: lazy-loading
  similarity: 0.7
- slug: performance-optimization
  similarity: 0.7
- slug: pagination
  similarity: 0.7
- slug: api-first-design
  similarity: 0.7
- slug: connection-pooling
  similarity: 0.7
---

## Description

Optimierung von API-Aufrufen ist die Praxis, die Anzahl, Größe und Latenzkosten der Netzwerkanfragen zu verringern, die ein Client zur Erledigung einer Aufgabe stellen muss, typischerweise durch Konsolidierung geschwätziger Sequenzen feingranularer Aufrufe in weniger, grobkörnigere, das Hinzufügen von Paginierung und Feldauswahl, und das Bündeln verwandter Operationen in einen einzigen Roundtrip. Legacy-APIs zeigen häufig geschwätzige Designs, weil jeder Endpunkt unabhängig über die Zeit hinzugefügt wurde, um einen spezifischen Bildschirm- oder Integrationsbedarf zu erfüllen, ohne dass jemand zurücktrat, um zu überlegen, wie viele Roundtrips ein typischer Client-Workflow tatsächlich erfordert; das Ergebnis sind Seiten, die ein Dutzend oder mehr sequenzielle Aufrufe zum Rendern ausgeben, wobei jeder seine eigene Netzwerklatenz zu den anderen hinzufügt. Dieses Problem verstärkt sich auf Verbindungen mit hoher Latenz oder begrenzter Bandbreite, wie mobilen Netzwerken, wo jeder zusätzliche Roundtrip vom Endnutzer direkt als langsamere Seitenladezeiten und höhere Serverlast durch die Verarbeitung vieler kleiner statt weniger großer Anfragen gespürt wird. Die Optimierung dieser Aufrufe bedeutet, tatsächliche Nutzungsmuster zu analysieren, um die wirkungsvollsten Konsolidierungsmöglichkeiten zu finden, und dann die API-Oberfläche neu zu gestalten — ohne notwendigerweise die zugrunde liegende Legacy-Geschäftslogik zu berühren —, sodass ein Client abrufen kann, was er braucht, in ein oder zwei Aufrufen statt vielen. Der Ansatz verbessert direkt Antwortzeit, Durchsatz und Bandbreitenverbrauch, aber grobkörnigere Endpunkte sind inhärent weniger flexibel und können komplex in der Wartung werden, sodass das Redesign enge Koordination zwischen den Teams erfordert, die die API konsumieren und produzieren, um zu vermeiden, das Geschwätzigkeitsproblem einfach in einen übermäßig starren Vertrag zu verschieben.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie geschwätzige API-Muster, bei denen Clients mehrere Roundtrips für Daten machen, die in einem einzigen Aufruf abgerufen werden könnten
- Konsolidieren Sie verwandte Endpunkte in grobkörnigere Operationen, die alle benötigten Daten auf einmal zurückgeben
- Implementieren Sie Paginierung für Endpunkte, die große Sammlungen zurückgeben, um die Übertragung unnötiger Daten zu vermeiden
- Nutzen Sie Feldauswahl oder Sparse Fieldsets, sodass Clients nur die Daten anfragen, die sie benötigen
- Ersetzen Sie sequenzielle API-Aufrufe durch Batch-Endpunkte, die mehrere Operationen in einer einzigen Anfrage verarbeiten
- Fügen Sie Antwortkomprimierung hinzu und nutzen Sie ETags oder bedingte Anfragen zur Verringerung redundanter Datenübertragung
- Profilen Sie API-Nutzungsmuster zur Identifikation der wirkungsvollsten Optimierungsziele

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Verringert Netzwerk-Roundtrips, was Antwortzeiten und Durchsatz direkt verbessert
- Senkt Serverlast durch Konsolidierung mehrerer Operationen in weniger, effizientere Aufrufe
- Verringert Bandbreitenverbrauch, besonders wichtig für mobile Clients auf begrenzten Netzwerken
- Verbessert die Nutzererfahrung durch schnellere Seitenladezeiten und Interaktionen

**Kosten und Risiken:**
- Grobkörnigere APIs können übermäßig komplex und schwerer zu warten werden
- Batch-Endpunkte können die individuelle Anfrageverarbeitungszeit erhöhen, selbst wenn sie die Gesamt-Roundtrips verringern
- Übermäßige Optimierung kann die API-Flexibilität verringern, was es neuen Konsumenten erschwert, die API zu nutzen
- Erfordert Koordination zwischen Frontend- und Backend-Teams zur Vereinbarung optimaler API-Verträge

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Die Produktdetailseite einer Legacy-E-Commerce-Plattform erforderte 12 separate API-Aufrufe zum Laden: einen für Produktdaten, einen für Preise, einen für Bestand, einen für Bewertungen und mehrere weitere für Empfehlungen und verwandte Produkte. Jeder Aufruf fügte Netzwerklatenz hinzu, und auf mobilen Verbindungen brauchte die Seite über acht Sekunden zum Rendern. Das Team konsolidierte diese in zwei Aufrufe: einen primären Produktendpunkt, der Preise, Bestand und eine grundlegende Bewertungszusammenfassung enthielt, und einen sekundären Endpunkt für Empfehlungen, der asynchron lud. Die Seitenladezeit sank auf unter zwei Sekunden, und die Backend-Server-CPU-Nutzung verringerte sich um etwa 30 % aufgrund weniger zu verarbeitender Anfragen.
