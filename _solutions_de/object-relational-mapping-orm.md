---
title: Object-Relational Mapping (ORM)
description: Abstraktion von Datenbankinteraktionen durch Objekte.
category:
- Code
- Database
problems:
- technology-lock-in
- vendor-lock-in
- database-query-performance-issues
- database-schema-design-problems
- difficult-code-comprehension
- high-coupling-low-cohesion
- n-plus-one-query-problem
- imperative-data-fetching-logic
layout: solution
lang: de
en_slug: object-relational-mapping-orm
related_solutions:
- slug: database-abstraction
  similarity: 0.9
- slug: platform-independent-data-storage
  similarity: 0.8
- slug: abstraction-layers
  similarity: 0.75
- slug: data-replication
  similarity: 0.75
- slug: automated-migration-tools
  similarity: 0.75
- slug: adapter
  similarity: 0.75
---

## Description

Object-Relational Mapping übersetzt zwischen den relationalen Zeilen und Spalten einer Datenbank und den Objekten, die eine Anwendung im Code manipuliert, sodass Entwickler mit Domänenentitäten arbeiten statt mit handgeschriebenem SQL und Ergebnismengen. Diese Indirektion entkoppelt Anwendungslogik von einem spezifischen Datenbankdialekt, was genau die Kopplung ist, die Legacy-Systeme in die Falle lockt, wenn eine Datenbankmigration, Lizenzänderung oder ein Performance-Problem einen Anbieterwechsel erzwingt. In einer Legacy-Codebasis mit Tausenden verstreuter roher Abfragen erlaubt die Einführung eines ORM hinter einer Repository-Schicht Teams, inkrementell zu migrieren — stabile, einfache Entitäten zuerst abzubilden, während performancekritische oder unkonventionelle Abfragen zunächst als natives SQL belassen werden —, statt eine Neuschreibung der gesamten Datenzugriffsschicht auf einmal zu erfordern. Der Zielkonflikt ist, dass ORM-generierte Abfragen weniger effizient sein können als handabgestimmtes SQL, sodass die Abstraktion typischerweise eine Ausweichmöglichkeit für Fälle braucht, in denen sie im Weg steht.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Inventarisieren Sie alle rohen SQL-Abfragen und Datenbankzugriffsmuster in der Legacy-Codebasis, um den Migrationsumfang zu verstehen
- Wählen Sie ein für Sprache und Ökosystem passendes ORM-Framework (z. B. Hibernate, Entity Framework, SQLAlchemy)
- Beginnen Sie damit, die stabilsten Domänenentitäten auf ORM-Modelle abzubilden, während komplexe Abfragen zunächst als natives SQL erhalten bleiben
- Führen Sie eine Repository- oder Datenzugriffsschicht ein, die ORM-Nutzung hinter sauberen Schnittstellen kapselt
- Migrieren Sie rohes SQL inkrementell und ersetzen Sie handgeschriebene Abfragen Modul für Modul durch ORM-Äquivalente
- Konfigurieren Sie Lazy Loading, Eager Fetching und Abfrageoptimierungseinstellungen, um übliche Performance-Fallstricke zu vermeiden
- Behalten Sie die Option, natives SQL für performancekritische Abfragen zu nutzen, wo die ORM-Abstraktion unakzeptablen Overhead einführt

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Entkoppelt Anwendungscode von spezifischen Datenbankdialekten, was Datenbankmigration machbar macht
- Verringert Boilerplate-Datenzugriffscode und beseitigt viele Klassen von SQL-Injection-Schwachstellen
- Verbessert die Entwicklerproduktivität durch die Arbeit mit Domänenobjekten statt Ergebnismengen
- Vereinfacht Unit-Testing durch leichteres Mocken von Datenzugriffsschichten

**Kosten und Risiken:**
- ORM-generierte Abfragen können ineffizient sein, besonders für komplexe Joins oder Massenoperationen
- Das N+1-Abfrageproblem kann die Performance still verschlechtern, wenn Ladestrategien nicht sorgfältig konfiguriert sind
- Fügt eine Abstraktionsebene hinzu, die verbergen kann, was tatsächlich auf Datenbankebene passiert
- Legacy-Schemata mit unkonventionellen Strukturen lassen sich möglicherweise nicht sauber auf ORM-Modelle abbilden
- Lernkurve für Teams, die mit ORM-Konzepten und -Konfiguration nicht vertraut sind

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Fertigungsunternehmen hatte ein Legacy-Bestandssystem mit über 2.000 für Oracle geschriebenen rohen SQL-Abfragen. Als das Unternehmen entschied, zu PostgreSQL zu migrieren, um Lizenzkosten zu senken, erschien die Aussicht, jede Abfrage neu zu schreiben, entmutigend. Das Team führte SQLAlchemy als ORM-Schicht ein, beginnend mit den 50 meistgenutzten Entitätstypen. Über vier Monate migrierten sie 80 % der Abfragen zu ORM-verwalteten Operationen. Die verbleibenden 20 %, die komplexes Reporting und Massenoperationen betrafen, blieben als natives SQL, aber zentralisiert in einer Repository-Schicht mit dialektbewussten Query-Buildern. Die ursprünglich auf zwei Jahre geschätzte Migration zu PostgreSQL wurde in acht Monaten abgeschlossen.
