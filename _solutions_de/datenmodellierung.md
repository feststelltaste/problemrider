---
title: Datenmodellierung
description: Abbildung von Geschäftskonzepten und -beziehungen in einem konzeptuellen
  Datenmodell.
category:
- Database
- Architecture
problems:
- poor-domain-model
- database-schema-design-problems
- complex-domain-model
- data-migration-complexities
- legacy-business-logic-extraction-difficulty
- data-structure-cache-inefficiency
- incorrect-index-type
- inefficient-database-indexing
- queries-that-prevent-index-usage
- schema-evolution-paralysis
- unused-indexes
- entity-attribute-value-overuse
- master-data-ownership-gaps
layout: solution
lang: de
en_slug: data-modeling
related_solutions:
- slug: domain-modeling
  similarity: 0.8
- slug: business-process-modeling
  similarity: 0.75
- slug: canonical-data-model
  similarity: 0.7
- slug: data-strategy
  similarity: 0.65
- slug: story-mapping
  similarity: 0.65
- slug: domain-patterns
  similarity: 0.65
---

## Description

Datenmodellierung erzeugt eine konzeptuelle Repräsentation von Geschäftsentitäten und ihren Beziehungen, unabhängig vom physischen Schema irgendeines bestimmten Systems, typischerweise ausgedrückt als Entity-Relationship-Diagramme, die beschreiben, was das Geschäft als Kunde, Bestellung oder Produkt betrachtet, und wie diese Konzepte zueinander in Beziehung stehen. Dieses Modell für ein Legacy-System zu bauen bedeutet, das bestehende Schema per Reverse Engineering zu erschließen und es gegen Interviews mit den Menschen abzugleichen, die die Daten tatsächlich nutzen, was routinemäßig eine Lücke zwischen den beiden zutage fördert: physische Tabellen, die veraltete oder duplizierte Versionen desselben Konzepts repräsentieren, und Geschäftsregeln, die nur in verstreutem Anwendungscode durchgesetzt werden statt in irgendeinem dokumentierten oder durchgesetzten Teil des Datenmodells selbst. Dies ist für Legacy-Modernisierung wichtig, weil ein über Jahre durch Ad-hoc-, Feature-für-Feature-Erweiterung aufgebautes Schema niemandem von sich aus sagt, was das Geschäft tatsächlich repräsentieren muss — das konzeptuelle Modell muss bewusst rekonstruiert werden, und einmal vorhanden wird es zum Referenzpunkt, gegen den die Designprobleme des physischen Schemas (unnötige Komplexität, fehlende Beschränkungen, redundante Tabellen) sichtbar werden und für Konsolidierung bewertet werden können. In einem Migrations- oder Ersatzprojekt speziell fungiert das konzeptuelle Modell als der Bauplan, der bestimmt, welche physischen Tabellen auf welche Zielentitäten abgebildet werden, welche Beziehungen erstmals formalisiert werden müssen und welche impliziten Geschäftsregeln aus Anwendungscode extrahiert und in die explizite Domänenschicht des neuen Systems überführt werden müssen.

## How to Apply ◆

- Erstellen Sie ein konzeptuelles Datenmodell, das Geschäftsentitäten und ihre Beziehungen unabhängig vom physischen Schema des Legacy-Systems erfasst.
- Vergleichen Sie das konzeptuelle Modell mit dem Legacy-Datenbankschema, um Unstimmigkeiten, fehlende Konzepte und unnötige Komplexität zu identifizieren.
- Nutzen Sie Entity-Relationship-Diagramme, um das Legacy-Datenmodell zu dokumentieren und es Entwicklern und Geschäfts-Stakeholdern zu kommunizieren.
- Modellieren Sie Daten in Begriffen der Geschäftsdomäne statt technischer Bequemlichkeit, um Schema-Verbesserungen während der Modernisierung zu leiten.
- Identifizieren Sie Datenintegritätsbeschränkungen, die im Anwendungscode existieren, aber im Datenbankschema fehlen, und dokumentieren Sie sie im Datenmodell.
- Nutzen Sie das Datenmodell als Bauplan für Datenmigrationsplanung, wenn Legacy-Datenbanken ersetzt oder umstrukturiert werden.

## Tradeoffs ⇄

**Vorteile:**
- Schafft ein gemeinsames Verständnis der Geschäftsdatenlandschaft über technische und geschäftliche Teams hinweg.
- Identifiziert Schemadesignprobleme und Möglichkeiten zur Normalisierung oder Umstrukturierung.
- Bietet eine Grundlage für Datenmigrations- und Systemersatzplanung.
- Deckt Geschäftsregeln auf, die in Datenbankbeschränkungen oder gespeicherten Prozeduren eingebettet sind.

**Kosten:**
- Die Erstellung genauer Datenmodelle für Legacy-Systeme mit undokumentierten Schemata ist zeitintensiv.
- Datenmodelle können veralten, wenn sie nicht neben Schemaänderungen gepflegt werden.
- Kann unangenehme Wahrheiten über die Lücke zwischen dem idealen Modell und der Realität offenbaren.
- Übermäßig detaillierte Modelle können ebenso schwer zu verstehen sein wie die Schemata, die sie beschreiben.

## How It Could Be

Ein Legacy-Bestandsverwaltungssystem hat eine Datenbank mit über 400 Tabellen, viele mit kryptischen Namen und undokumentierten Beziehungen. Vor dem Versuch einer Migration zu einer modernen Plattform erstellt das Team ein konzeptuelles Datenmodell, indem das Schema per Reverse Engineering erschlossen und Lagerpersonal interviewt wird. Sie entdecken, dass dreißig Tabellen unterschiedliche, über Jahre durch Ad-hoc-Erweiterungen angehäufte Versionen desselben Konzepts repräsentieren, und dass kritische Geschäftsregeln (wie Mindestbestandsschwellen) nur im Anwendungscode durchgesetzt werden, nicht in Datenbankbeschränkungen. Das Datenmodell wird zum Migrationsbauplan und leitet, welche Tabellen konsolidiert, welche Beziehungen formalisiert und welche Geschäftsregeln in die Domänenschicht des neuen Systems extrahiert werden sollen.
