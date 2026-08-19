---
title: CQRS
description: Trennung von Lese- und Schreibmodellen in unabhängig optimierte und
  skalierte Pfade.
category:
- Architecture
- Performance
problems:
- slow-database-queries
- scaling-inefficiencies
- database-query-performance-issues
- high-database-resource-utilization
- monolithic-architecture-constraints
- slow-response-times-for-lists
- imperative-data-fetching-logic
- entity-attribute-value-overuse
layout: solution
lang: de
en_slug: cqrs
related_solutions:
- slug: denormalization
  similarity: 0.7
- slug: read-replicas
  similarity: 0.7
- slug: data-replication
  similarity: 0.7
- slug: data-partitioning
  similarity: 0.7
- slug: business-event-processing
  similarity: 0.7
- slug: distributed-caching
  similarity: 0.7
---

## Description

CQRS (Command Query Responsibility Segregation) trennt das zum Schreiben von Daten genutzte Modell vom zum Lesen genutzten Modell, sodass jedes unabhängig optimiert und skaliert werden kann, statt beide durch ein einziges normalisiertes Schema zu zwingen, das primär um eines der beiden Anliegen herum gestaltet wurde. Legacy-Systeme haben häufig ein einzelnes relationales Schema entwickelt, das transaktionale Schreibvorgänge zum Zeitpunkt seiner Gestaltung gut bediente, aber während Berichts- und Analyseabfragebedürfnisse neben dem Transaktionsvolumen wuchsen, begannen beide Workloads, um dieselben Datenbankressourcen zu konkurrieren, und beeinträchtigten sich gegenseitig auf eine Weise, die keine Menge an Indizierung allein vollständig lösen konnte. Ein separates, denormalisiertes Lesemodell einzuführen — asynchron aus Domain-Events befüllt, die von der Schreibseite emittiert werden — erlaubt komplexen Berichtsabfragen, gegen ein speziell für sie gebautes Schema zu laufen, während die transaktionale Datenbank freigesetzt wird, sich rein auf Schreibdurchsatz und Konsistenz zu konzentrieren. Diese Trennung wird bewusst selektiv angewandt, auf die spezifischen Teile eines Systems, in denen Lese- und Schreibzugriffsmuster genug divergiert sind, um sie zu rechtfertigen, statt als systemweite architektonische Überarbeitung. Der Tradeoff ist die Einführung von Eventual Consistency zwischen den beiden Modellen und die zusätzliche operative Komplexität der Pflege von Synchronisation und Behandlung von Projektionsfehlern, beides muss gegen das Performance-Problem abgewogen werden, das CQRS lösen soll.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie Bereiche, in denen Lese- und Schreib-Workloads grundlegend unterschiedliche Performance-Eigenschaften oder Skalierungsbedürfnisse haben
- Trennen Sie das Lesemodell vom Schreibmodell, beginnend mit den performance-kritischsten Abfragen
- Erstellen Sie denormalisierte Leseprojektionen, optimiert für spezifische Abfragemuster
- Nutzen Sie Domain-Events, um Lesemodelle mit dem Schreibmodell synchron zu halten
- Beginnen Sie mit einer einfachen synchronen Projektion, bevor Sie bei Bedarf Eventual Consistency einführen
- Wenden Sie CQRS selektiv auf die Teile des Systems an, die am meisten profitieren, nicht als systemweites Muster
- Implementieren Sie Kompensationsmechanismen zur Behandlung von Eventual Consistency in der Nutzererfahrung

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Lese- und Schreibmodelle können unabhängig für ihre spezifischen Zugriffsmuster optimiert werden
- Ermöglicht unabhängige Skalierung von lese- und schreiblastigen Komponenten
- Lesemodelle können denormalisiert und vorberechnet werden für schnelle Abfrageantworten
- Vereinfacht komplexe Abfragen, indem Lesemodelle speziell für jeden Abfragebedarf gestaltet werden

**Kosten und Risiken:**
- Führt Eventual Consistency zwischen Lese- und Schreibmodellen ein, was die Nutzererfahrung verkompliziert
- Erhöht die Systemkomplexität mit separaten Modellen, Projektionen und Synchronisationslogik
- Erfordert sorgfältige Behandlung von Projektionsfehlern und Wiederaufbau-Szenarien
- Kann Overengineering für Systeme sein, in denen Lese- und Schreibmuster ähnlich sind

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Buchhaltungssystem nutzte dasselbe normalisierte relationale Schema sowohl für die Transaktionsaufzeichnung als auch für die Berichtserstellung. Während das Transaktionsvolumen wuchs, begannen komplexe Berichtsabfragen, mit der Transaktionsverarbeitung um Datenbankressourcen zu konkurrieren, was beide verlangsamte. Das Team führte CQRS ein, indem eine separate Lesedatenbank mit denormalisierten Views erstellt wurde, optimiert für die häufigsten Berichte. Domain-Events aus der transaktionalen Datenbank lösten Aktualisierungen der Leseprojektionen aus. Die Berichtserstellungszeiten sanken von Minuten auf Sekunden, und der Transaktionsverarbeitungsdurchsatz verdoppelte sich, weil die Schreibdatenbank keine teuren analytischen Abfragen mehr handhabte.
