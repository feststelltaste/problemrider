---
title: Schlechtes Domänenmodell
description: Zentrale Geschäftskonzepte werden im System schlecht verstanden oder
  widergespiegelt, was zu brüchiger Logik und Missverständnissen führt.
category:
- Architecture
- Code
- Communication
related_problems:
- slug: complex-domain-model
  similarity: 0.75
- slug: legacy-business-logic-extraction-difficulty
  similarity: 0.65
- slug: architectural-mismatch
  similarity: 0.6
- slug: poorly-defined-responsibilities
  similarity: 0.6
- slug: complex-implementation-paths
  similarity: 0.6
- slug: integration-difficulties
  similarity: 0.6
solutions:
- modularization-and-bounded-contexts
- bounded-contexts
- business-process-automation
- business-process-modeling
- canonical-data-model
- data-ecosystems
- data-enrichment
- data-modeling
- data-strategy
- decision-tables
- rule-based-systems
- subject-matter-reviews
- ubiquitous-language
- domain-aligned-architecture
- domain-based-authorization-concept
- domain-driven-design
- domain-experts
- domain-modeling
- domain-patterns
- domain-specific-languages
- event-storming
layout: problem
lang: de
en_slug: poor-domain-model
---

## Description

Ein schlechtes Domänenmodell tritt auf, wenn das Softwaresystem die realen Geschäftskonzepte, Beziehungen und Regeln, die es unterstützen soll, nicht akkurat repräsentiert. Dies führt zu einer fundamentalen Trennung zwischen der Funktionsweise des Geschäfts und der Art, wie die Software diesen Betrieb modelliert. Das resultierende System wird brüchig, schwer zu modifizieren und fehleranfällig, weil die Codestruktur nicht mit der Geschäftsrealität übereinstimmt. Dieses Problem ist besonders kritisch bei der Legacy-Modernisierung, wo bestehende schlechte Modelle oft repliziert statt verbessert werden.

## Indicators ⟡

- Geschäfts-Stakeholder und Entwickler reden häufig aneinander vorbei, indem sie unterschiedliche Terminologie nutzen
- Datenbankschemas, die natürliche Geschäftsbeziehungen nicht widerspiegeln
- Geschäftsregeln, verstreut über die Codebasis statt zentralisiert in Domänenlogik
- Häufige Anfragen nach „einfachen" Änderungen, die viele nicht zusammenhängende Teile des Systems berühren müssen
- Fachexperten äußern Verwirrung darüber, wie das System ihre Arbeit repräsentiert
- Neue Teammitglieder kämpfen damit, die Verbindung zwischen Code und Geschäftsprozessen zu verstehen

## Symptoms ▲

- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Wenn das Domänenmodell nicht der Geschäftsrealität entspricht, erstellen Entwickler Workarounds, um die Fehlpassung zu kompensieren.
- [Wellenwirkung von Änderungen](wellenwirkung-von-aenderungen.md)
<br/>  Änderungen an Geschäftsregeln erfordern das Anfassen vieler nicht zusammenhängender Teile des Systems, weil Geschäftslogik verstreut statt zentralisiert ist.
- [Große Schätzungen für kleine Änderungen](grosse-schaetzungen-fuer-kleine-aenderungen.md)
<br/>  Einfache Geschäftsänderungen erfordern unverhältnismäßigen Aufwand, weil die Codestruktur nicht mit Geschäftskonzepten übereinstimmt.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Verstreute Geschäftslogik bedeutet, dass Änderungen in einem Bereich unbeabsichtigt Geschäftsregeln brechen, die anderswo durchgesetzt werden.
- [Probleme im Datenbankschema-Design](probleme-im-datenbankschema-design.md)
<br/>  Ein schlechtes Domänenmodell führt zu Datenbankschemas, die natürliche Geschäftsbeziehungen nicht widerspiegeln.

## Causes ▼

- [Kommunikationslücke zwischen Stakeholdern und Entwicklern](kommunikationsluecke-zwischen-stakeholdern-und-entwicklern.md)
<br/>  Wenn Entwickler und Geschäftsexperten nicht effektiv kommunizieren, weicht das Softwaremodell von der Geschäftsrealität ab.
- [Unzureichende Design-Fähigkeiten](unzureichende-design-faehigkeiten.md)
<br/>  Fehlende Expertise in Domänenmodellierung führt zu naiven Repräsentationen komplexer Geschäftskonzepte.
- [Unvollständiges Wissen](unvollstaendiges-wissen.md)
<br/>  Unvollständiges Verständnis von Geschäftsprozessen resultiert in einem Domänenmodell, das kritische Konzepte und Beziehungen verpasst.
- [Bequemlichkeitsgetriebene Entwicklung](bequemlichkeitsgetriebene-entwicklung.md)
<br/>  Entwickler modellieren die Domäne basierend darauf, was leicht zu implementieren ist, statt darauf, was das Geschäft akkurat repräsentiert.

## Detection Methods ○

- Durchführung von Domänenmodellierungs-Workshops mit Geschäftsexperten und Entwicklungsteams
- Überprüfung von Code auf Geschäftslogik, die über mehrere Schichten oder Module verstreut ist
- Analyse von Fehlermustern zur Identifikation von Bereichen, in denen Geschäftsregeln schlecht implementiert sind
- Kartierung von Geschäftsprozessen zu Codestrukturen zur Identifikation von Fehlpassungen
- Interview von Fachexperten dazu, wie gut das System ihre mentalen Modelle widerspiegelt
- Überprüfung von Datenbankschemas auf Tabellen und Beziehungen, die nicht auf Geschäftskonzepte abbilden
- Untersuchung von Integrationspunkten, an denen Domänenmodell-Fehlpassungen Übersetzungskomplexität verursachen

## Examples

Das Bestellverwaltungssystem eines E-Commerce-Unternehmens behandelt „Order" als einfache Datenstruktur mit Statusfeldern, statt die komplexe Geschäftsrealität zu modellieren, in der Bestellungen durch unterschiedliche Zustände gehen (aufgegeben, bestätigt, abgewickelt, versendet, geliefert) mit spezifischen Geschäftsregeln, die Übergänge regeln. Dies führt zu Szenarien, in denen Bestellungen als „versendet" markiert werden können, bevor sie „bestätigt" sind, oder als „geliefert" ohne „abgewickelt" zu sein. Geschäftsnutzer stoßen ständig auf Daten, die keinen Sinn ergeben, was manuelle Eingriffe erfordert. Als das Unternehmen versucht, neue Features wie Teillieferungen oder Bestelländerungen hinzuzufügen, entdecken sie, dass das schlechte Domänenmodell diese Änderungen extrem schwierig macht, was umfangreiches Refactoring über mehrere Systeme hinweg erfordert, statt einfacher Geschäftsregelergänzungen.
