---
title: Hexagonale Architektur
description: Isolierung der Geschäftslogik von der Infrastruktur durch Ports und
  Adapter.
category:
- Architecture
problems:
- tight-coupling-issues
- difficult-to-test-code
- legacy-business-logic-extraction-difficulty
- monolithic-architecture-constraints
- technology-lock-in
- vendor-dependency
- architectural-mismatch
- stagnant-architecture
- single-entry-point-design
layout: solution
lang: de
en_slug: hexagonal-architecture
related_solutions:
- slug: layered-architecture
  similarity: 0.75
- slug: abstraction-layers
  similarity: 0.75
- slug: adapter
  similarity: 0.75
- slug: database-abstraction
  similarity: 0.75
- slug: microservices-architecture
  similarity: 0.75
- slug: protocol-abstraction
  similarity: 0.7
---

## Description

Hexagonale Architektur, auch bekannt als Ports und Adapter, trennt die Kerngeschäftslogik eines Systems von der Infrastruktur, von der sie abhängt, indem Ports definiert werden — Schnittstellen, die ausdrücken, was die Domäne von der Außenwelt braucht und ihr anbietet — und Adapter, die diese Ports für spezifische Technologien wie eine Datenbank, eine Message Queue oder ein UI-Framework implementieren. Der Domänencode hängt nur von den Ports ab, die er definiert, nie von konkreter Infrastruktur, was die typische Abhängigkeitsrichtung umkehrt, die in Legacy-Systemen zu finden ist, wo Geschäftsregeln direkt mit JDBC-Aufrufen, SOAP-Clients oder UI-Event-Handlern verflochten sind. Diese Umkehrung ist es, was das Muster speziell für die Legacy-Modernisierung wertvoll macht: Weil Geschäftslogik keine Infrastrukturpakete mehr importiert, wird es möglich, diese Logik in Millisekunden mit In-Memory-Stubs zu testen statt eine echte Datenbank hochzufahren, und eine zugrundeliegende Technologie zu ersetzen — etwa von einem Datenbankanbieter zu einem anderen zu migrieren —, indem nur der Adapter neu geschrieben wird, während die Domäne unangetastet bleibt. Diese Grenze in ein bereits tief gekoppeltes Legacy-System nachzurüsten geschieht selten auf einmal; sie wird typischerweise inkrementell an den spezifischen Nahtstellen eingeführt, die den meisten Schmerz verursachen, indem eine Kategorie von Legacy-Infrastrukturaufrufen nach der anderen hinter eine Port-Schnittstelle gewickelt wird. Die Kosten sind zusätzliche Indirektion und mehr Dateien und Schnittstellen als ein eng gekoppeltes Äquivalent, was ein vernünftiger Tausch für ein Legacy-System in aktiver Modernisierung ist, aber Überkonstruktion für ein kleines, stabiles System sein kann, das seine Infrastruktur nie austauschen muss.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie die Kerngeschäftslogik im Legacy-System und trennen Sie sie von Infrastrukturbelangen wie Datenbanken, Messaging und UI
- Definieren Sie Ports (Schnittstellen), die ausdrücken, was die Geschäftslogik von der Außenwelt braucht und was sie anbietet
- Erstellen Sie Adapter, die Ports implementieren und die Lücke zwischen Domäne und Infrastrukturtechnologien überbrücken
- Beginnen Sie an den Grenzen, an denen Kopplung an Infrastruktur den meisten Schmerz verursacht, wie Datenbankzugriffsschichten
- Führen Sie das Muster inkrementell ein, indem Legacy-Infrastrukturaufrufe nacheinander pro Subsystem hinter Port-Schnittstellen gewickelt werden
- Nutzen Sie Dependency Injection, um Adapter mit Ports zu verdrahten, sodass Test Doubles echte Infrastruktur ersetzen können
- Stellen Sie sicher, dass kein Domänencode Infrastrukturpakete direkt importiert

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Geschäftslogik wird isoliert testbar, ohne Datenbanken, Netzwerke oder externe Dienste
- Technologiemigrationen werden machbar, weil nur Adapter ersetzt werden müssen, nicht die Kernlogik
- Erzwingt klare architektonische Grenzen, die architektonische Erosion über die Zeit verhindern
- Ermöglicht parallele Entwicklung: Teams können unabhängig an Adaptern und Domäne arbeiten

**Kosten und Risiken:**
- Führt zusätzliche Abstraktionen und Indirektion ein, die die Zahl der Dateien und Schnittstellen erhöhen
- Erfordert Disziplin, die Grenze zu pflegen, besonders unter Termindruck
- Das Nachrüsten in ein tief gekoppeltes Legacy-System kann eine große Vorabinvestition sein
- Risiko der Überkonstruktion, wenn auf einfache Systeme angewendet, die nicht von der Trennung profitieren

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Logistikunternehmen hatte ein monolithisches Auftragsverwaltungssystem, in dem Geschäftsregeln mit direkten JDBC-Aufrufen, SOAP-Client-Code und Swing-UI-Logik verflochten waren. Das Testen jeder Regel erforderte, die gesamte Anwendung mit einer echten Datenbank hochzufahren. Das Team begann, die Preis-Engine zu extrahieren, indem eine Port-Schnittstelle für Bestandsabfragen und eine weitere für Steuerberechnungen definiert wurde. Legacy-JDBC-Abfragen wurden in Adapter-Implementierungen gewickelt, während Tests In-Memory-Stubs nutzten. Innerhalb weniger Monate konnte die Preis-Engine in Millisekunden statt Minuten getestet werden, und als das Unternehmen später von Oracle zu PostgreSQL migrierte, mussten nur die Adapter-Implementierungen geändert werden.
