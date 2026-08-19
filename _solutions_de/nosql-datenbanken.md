---
title: NoSQL-Datenbanken
description: Speicherung von Daten in flexiblen, schemalosen Formaten.
category:
- Database
- Architecture
problems:
- database-schema-design-problems
- scaling-inefficiencies
- schema-evolution-paralysis
- slow-database-queries
- unbounded-data-growth
- high-database-resource-utilization
- data-migration-complexities
layout: solution
lang: de
en_slug: nosql-databases
related_solutions:
- slug: graph-databases
  similarity: 0.85
- slug: denormalization
  similarity: 0.75
- slug: data-replication
  similarity: 0.75
- slug: database-abstraction
  similarity: 0.75
- slug: materialized-views
  similarity: 0.75
- slug: containerized-databases
  similarity: 0.75
---

## Description

NoSQL-Datenbanken speichern Daten in flexiblen, oft schemalosen Formaten — Dokumente, Schlüssel-Wert-Paare, breite Spalten oder Graphen — statt der festen relationalen Tabellen, um die herum die meisten Legacy-Systeme gebaut wurden. Diese Flexibilität erlaubt Datenstrukturen, sich ohne die Migrationszeremonie weiterzuentwickeln, die relationale Schemaänderungen erfordern, und erlaubt, dass spezifische Zugriffsmuster (Hochvolumen-Schreibvorgänge, tief verschachtelte Dokumente, Graph-Traversierungen) von einer für sie entworfenen Speicher-Engine bedient werden, statt in eine relationale Form gezwungen zu werden. In Legacy-Kontexten, wo Jahre Ad-hoc-Schema-Erweiterungen oft ausufernde Entity-Attribute-Value-Tabellen oder brüchige Migrationsskripte produziert haben, ist NoSQL-Einführung meist am effektivsten als gezielte Auslagerung spezifischer Workloads statt einer vollständigen Ersetzung des relationalen Kerns, da sie transaktionale Garantien und ausgereifte Abfragefähigkeiten eintauscht, auf die Legacy-Geschäftslogik möglicherweise noch angewiesen ist.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie Anwendungsfälle, in denen das relationale Modell Reibung erzeugt: stark variable Schemata, dokumentorientierte Daten oder extreme Lese-/Schreibvolumina
- Wählen Sie die passende NoSQL-Kategorie (Dokument, Schlüssel-Wert, Column-Family, Graph) basierend auf den spezifischen Zugriffsmustern
- Beginnen Sie damit, spezifische Workloads auszulagern (z. B. Sitzungsspeicherung, Ereignisprotokolle, Produktkataloge), statt die gesamte relationale Datenbank zu ersetzen
- Implementieren Sie eine Datenzugriffsschicht, die das Speicher-Backend abstrahiert, damit der Rest der Anwendung nicht eng an die NoSQL-Technologie gekoppelt ist
- Planen Sie für eventuelle Konsistenz, falls Sie von einem stark konsistenten relationalen System wechseln, und stellen Sie sicher, dass die Geschäftslogik dies tolerieren kann
- Migrieren Sie Daten inkrementell, wobei beide Speicher während der Übergangsperiode parallel mit Abgleichsprüfungen laufen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Ermöglicht horizontale Skalierung für Workloads, die über relationale Einzelserver-Datenbanken hinauswachsen
- Beseitigt schmerzhafte Schema-Migrationen für Daten mit sich entwickelnden oder variablen Strukturen
- Kann Lese- und Schreibperformance für spezifische Zugriffsmuster dramatisch verbessern
- Verringert Impedance Mismatch zwischen Anwendungsobjekten und Speicherformat

**Kosten und Risiken:**
- Der Verlust von ACID-Transaktionen und starker Konsistenz erfordert sorgfältige Handhabung auf Anwendungsebene
- Teams, die nur mit relationalen Datenbanken erfahren sind, stehen vor einer erheblichen Lernkurve
- Der Betrieb mehrerer Datenbanktechnologien erhöht die betriebliche Komplexität
- Fehlende Schemadurchsetzung kann über die Zeit zu Datenqualitätsproblemen führen
- Abfragefähigkeiten sind oft eingeschränkter als SQL, was Komplexität in den Anwendungscode verschiebt

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Die Legacy-relationale Datenbank eines E-Commerce-Unternehmens hatte Schwierigkeiten mit einem Produktkatalog, bei dem jede Produktkategorie stark unterschiedliche Attribute hatte, was Hunderte spärlicher Spalten und eine Vermehrung von Entity-Attribute-Value-Tabellen zur Folge hatte. Das Team migrierte den Katalog zu MongoDB und speicherte jedes Produkt als Dokument mit kategoriespezifischen Feldern. Dies beseitigte die komplexen JOIN-Abfragen, die langsame Seitenladezeiten verursacht hatten, verringerte die Antwortzeit der Katalogabfrage von Sekunden auf Millisekunden und machte es für Merchandising-Teams trivial, neue Produktattribute hinzuzufügen, ohne Datenbankänderungsanfragen einreichen zu müssen.
