---
title: Datenformatkonvertierung
description: Bereitstellung von Mechanismen zur Konvertierung zwischen verschiedenen
  Datenformaten.
category:
- Architecture
- Database
problems:
- data-migration-complexities
- cross-system-data-synchronization-problems
- integration-difficulties
- legacy-business-logic-extraction-difficulty
- poor-interfaces-between-applications
layout: solution
lang: de
en_slug: data-format-conversion
related_solutions:
- slug: standardized-data-formats
  similarity: 0.85
- slug: data-formats
  similarity: 0.85
- slug: backward-compatible-data-formats
  similarity: 0.8
- slug: automated-migration-tools
  similarity: 0.75
- slug: data-strategy
  similarity: 0.75
- slug: platform-independent-data-storage
  similarity: 0.7
---

## Description

Datenformatkonvertierung stellt dedizierte Komponenten bereit, die Daten zwischen dem Format, das ein Legacy-System nativ erzeugt oder erwartet, und dem Format übersetzen, das von einem anderen System benötigt wird, mit dem Daten ausgetauscht werden müssen, typischerweise implementiert als eigenständiger Konvertierungsdienst oder eigenständige Bibliothek statt als in jeden Konsumenten eingebettete Logik. Wenn beide Formate gleichzeitig in Nutzung bleiben müssen — meist während einer gestuften Migration — arbeitet der Konverter bidirektional, übersetzt eingehende Daten für neue Konsumenten ins moderne Format, während er ausgehende Daten zurück ins Legacy-Format für Konsumenten übersetzt, die noch nicht migriert haben. Dieses Muster ist zentral für Legacy-Modernisierung, weil es selten machbar ist, jedes System, das ein gegebenes Format liest oder schreibt, gleichzeitig umzustellen: Ein Konverter lässt das Legacy-Format und das Zielformat so lange koexistieren, wie nötig, und entkoppelt das Tempo der Konsumentenmigration vom Zeitplan des Ersatzes des Quellsystems selbst. Die Konvertierungslogik an einem Ort zu zentralisieren, statt jeden Konsumenten seine eigene Übersetzung implementieren zu lassen, verhindert auch die subtile Drift, die entsteht, wenn mehrere Ad-hoc-Konverter dasselbe Legacy-Format leicht unterschiedlich interpretieren. Weil jede Übersetzung zwischen Formaten riskiert, an den Rändern Präzision zu verlieren oder Bedeutung zu verändern, müssen konvertierte Daten gegen das Zielschema validiert werden, und Konvertierungsfehler müssen protokolliert und überwacht werden, statt still verschluckt zu werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Bauen Sie dedizierte Konverter-Komponenten, die zwischen Legacy- und modernen Datenformaten übersetzen
- Implementieren Sie bidirektionale Konvertierung, wenn altes und neues System während der Migration koexistieren müssen
- Validieren Sie konvertierte Daten gegen das Zielschema, um Übersetzungsfehler früh zu erkennen
- Nutzen Sie eine Pipeline-Architektur für komplexe Konvertierungen, die mehrere Transformationsschritte verketten
- Protokollieren Sie Konvertierungsfehler und Anomalien für Überwachung und Fehlersuche
- Bieten Sie Konvertierungswerkzeuge als gemeinsame Bibliotheken oder Dienste an, um Duplizierung über Teams hinweg zu vermeiden

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Ermöglicht Kommunikation zwischen Systemen, die inkompatible Datenformate nutzen
- Unterstützt schrittweise Migration, indem alte und neue Formate koexistieren können
- Zentralisiert Formatübersetzungslogik, statt sie über Konsumenten zu verstreuen

**Kosten und Risiken:**
- Konvertierungslogik kann subtilen Datenverlust oder semantische Drift einführen, wenn sie nicht sorgfältig getestet wird
- Bidirektionale Konverter sind deutlich komplexer als unidirektionale
- Der Performance-Overhead der Konvertierung kann bei hochvolumigen Datenflüssen erheblich sein
- Konverter werden zur Pflegelast, wenn sich Quell- oder Zielformat häufig ändern

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Versorgungsunternehmen musste von einem proprietären Festbreiten-Datensatzformat, genutzt von einem 20 Jahre alten Abrechnungssystem, zu einem modernen JSON-basierten Format migrieren. Das Team baute einen Konverterdienst, der beide Richtungen handhabte: Eingehende Datensätze wurden für das neue System in JSON konvertiert, während ausgehende Daten für noch nicht migrierte nachgelagerte Systeme zurück ins Legacy-Format konvertiert wurden. Über 18 Monate wurden nachgelagerte Konsumenten einer nach dem anderen migriert, und der Rückkonverter wurde schließlich stillgelegt.
