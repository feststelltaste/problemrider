---
title: Domänenausgerichtete Architektur
description: Ausrichtung der Softwarestruktur an Domänenstrukturen und -prozessen.
category:
- Architecture
problems:
- architectural-mismatch
- organizational-structure-mismatch
- monolithic-architecture-constraints
- high-coupling-low-cohesion
- poor-domain-model
- complex-domain-model
- ripple-effect-of-changes
- shared-database
layout: solution
lang: de
en_slug: domain-aligned-architecture
related_solutions:
- slug: domain-driven-design
  similarity: 0.8
- slug: domain-modeling
  similarity: 0.8
- slug: bounded-contexts
  similarity: 0.75
- slug: team-boundaries-aligned-to-architecture
  similarity: 0.7
- slug: domain-patterns
  similarity: 0.7
- slug: modularization-and-bounded-contexts
  similarity: 0.7
---

## Description

Domänenausgerichtete Architektur strukturiert die Module eines Systems so um, dass ihre internen Grenzen die Grenzen der Geschäftsdomäne widerspiegeln, der sie dienen — Code wird um Konzepte wie „Bestellverwaltung" oder „Sendungsverfolgung" gruppiert —, statt ihn um technische Schichten wie Controller, Services und Repositories zu gruppieren, die jedes Geschäftskonzept gleichzeitig durchschneiden. Diese Unterscheidung ist in Legacy-Systemen enorm wichtig, wo technische-Schichten-Organisation das Standardergebnis dessen ist, wie Software traditionell gelehrt und strukturiert wurde, und sie erzeugt ein spezifisches, erkennbares Symptom: Eine einzelne Geschäftsänderung erfordert das gleichzeitige Anfassen mehrerer technischer Schichten und Koordination über welche Teams auch immer jede Schicht besitzen. Indem Code stattdessen entlang Domänenlinien umorganisiert wird, wird eine Änderung an einer Geschäftsfähigkeit auf das Modul lokalisiert, das sie besitzt, was die Wellenwirkung von Änderungen direkt reduziert, die Systeme plagt, in denen technische und geschäftliche Struktur auseinandergedriftet sind. Team-Eigentümerschaft an denselben Domänengrenzen auszurichten verstärkt den Nutzen, da ein Team, das eine Domäne Ende-zu-Ende besitzt, keine Releases mehr mit anderen Teams synchronisieren muss, um ein domänenspezifisches Feature auszuliefern. Die Umstrukturierung selbst ist notwendigerweise graduell und erfordert echtes Domänenverständnis, um die Grenzen korrekt zu ziehen, aber sie erzeugt auch natürliche Nahtstellen, entlang derer ein Monolith später in separat deploybare Services aufgeteilt werden kann, sollte das zum Ziel werden.

## How to Apply ◆

- Bilden Sie die aktuelle Modulstruktur des Legacy-Systems gegen die Geschäftsdomäne ab, um zu identifizieren, wo technische Zerlegung von Geschäftsgrenzen abweicht.
- Organisieren Sie Code entlang von Domänengrenzen statt technischer Schichten neu (z. B. gruppieren nach „Bestellverwaltung" statt „Controller, Services, Repositories").
- Richten Sie Team-Eigentümerschaft an diesen Domänengrenzen aus, sodass jedes Team eine kohärente Geschäftsfähigkeit Ende-zu-Ende besitzt.
- Nutzen Sie Domain Events, um Domänenmodule zu entkoppeln, die derzeit über gemeinsame Daten oder direkte Methodenaufrufe kommunizieren.
- Refaktorieren Sie gemeinsam genutzte Hilfsprogramme und Querschnittscode in explizite gemeinsame Bibliotheken, statt Domänenmodule voneinander abhängen zu lassen.
- Erstellen Sie explizite Schnittstellen an Domänengrenzen, die definieren, wie Domänen interagieren, und ersetzen Sie Ad-hoc-interne Kopplung.

## Tradeoffs ⇄

**Vorteile:**
- Änderungen an einer Geschäftsdomäne sind lokalisiert, was die Wellenwirkung über die Codebasis reduziert.
- Teams können unabhängig an ihrer Domäne arbeiten, ohne sich gegenseitig zu blockieren.
- Die Systemstruktur wird für Entwickler, die das Geschäft verstehen, intuitiver.
- Bietet natürliche Zerlegungsgrenzen für zukünftige Microservice-Extraktion.

**Kosten:**
- Die Umstrukturierung eines Legacy-Systems entlang von Domänengrenzen ist ein gradueller, mehrmonatiger Aufwand.
- Manche technischen Belange spannen sich echt über Domänen hinweg und müssen über gemeinsame Infrastruktur behandelt werden.
- Erfordert tiefes Verständnis der Geschäftsdomäne, um korrekte Grenzen zu ziehen.
- Kann mit bestehenden Teamstrukturen kollidieren und organisatorische Änderungen erfordern.

## How It Could Be

Ein Legacy-Logistiksystem ist nach technischer Schicht organisiert: aller Datenbankzugriff in einem Modul, alle Geschäftslogik in einem anderen, aller UI-Code in einem dritten. Eine Änderung am Sendungsverfolgungs-Feature erfordert Modifikationen über alle drei Schichten und Koordination zwischen drei Teams. Das Architekturteam strukturiert das System um, sodass Sendungsverfolgung, Lagerverwaltung und Spediteurintegration jeweils zu vertikalen Domänenmodulen werden, die ihren eigenen Datenzugriff, ihre Logik und UI-Komponenten enthalten. Teams werden diesen Domänen zugeordnet. Nach der Umstrukturierung kann das Sendungsverfolgungsteam Features unabhängig liefern, ohne teamübergreifende Koordination. Die durchschnittliche Zeit zur Lieferung eines domänenspezifischen Features sinkt von drei Wochen auf eine Woche, weil Änderungen keine synchronisierten Releases über mehrere Teams mehr erfordern.
