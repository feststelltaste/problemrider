---
title: Gemeinsam genutzte Datenbank
description: Eine Situation, in der sich mehrere Services oder Komponenten eine einzige
  Datenbank teilen.
category:
- Architecture
- Database
related_problems:
- slug: shared-dependencies
  similarity: 0.8
- slug: deployment-coupling
  similarity: 0.6
- slug: monolithic-architecture-constraints
  similarity: 0.55
- slug: database-schema-design-problems
  similarity: 0.55
- slug: schema-evolution-paralysis
  similarity: 0.55
- slug: high-number-of-database-queries
  similarity: 0.5
solutions:
- evolutionary-database-design
- modularization-and-bounded-contexts
- data-integration
- team-boundaries-aligned-to-architecture
- anti-corruption-layer
- domain-aligned-architecture
- change-impact-analysis
- bounded-contexts
- api-first-design
layout: problem
lang: de
en_slug: shared-database
---

## Description
Eine gemeinsam genutzte Datenbank ist eine Situation, in der sich mehrere Services oder Komponenten eine einzige Datenbank teilen. Dies ist ein häufiges Problem in monolithischen Architekturen, wo alle Komponenten eng gekoppelt und als eine einzige Einheit deployt sind. Eine gemeinsam genutzte Datenbank kann zu einer Reihe von Problemen führen, einschließlich Deployment-Kopplung, Ineffizienzen bei der Skalierung und Problemen durch enge Kopplung.

## Indicators ⟡
- Mehrere Services oder Komponenten lesen von und schreiben in dieselbe Datenbank.
- Es ist nicht möglich, das Datenbankschema zu ändern, ohne mehrere Services oder Komponenten zu beeinflussen.
- Es ist nicht möglich, die Datenbank für einen Service oder eine Komponente zu skalieren, ohne die anderen zu beeinflussen.
- Die Datenbank ist ein Single Point of Failure für das gesamte System.

## Symptoms ▲

- [Deployment-Kopplung](deployment-kopplung.md)
<br/>  Services, die sich eine Datenbank teilen, müssen Deployments koordinieren, um das Brechen gemeinsam genutzter Schema-Abhängigkeiten zu vermeiden, was Deployment-Kopplung schafft.
- [Ineffizienzen bei der Skalierung](ineffizienzen-bei-der-skalierung.md)
<br/>  Eine gemeinsam genutzte Datenbank kann nicht unabhängig für verschiedene Services skaliert werden, was die gesamte Datenbank zwingt, für die Spitzennachfrage jedes einzelnen Verbrauchers skaliert zu werden.
- [Lähmung der Schema-Evolution](laehmung-der-schema-evolution.md)
<br/>  Schemaänderungen werden riskant und schwierig, wenn mehrere Services von denselben Tabellen abhängen, was zu Lähmung der Schema-Evolution führt.
- [Probleme bei der Teamkoordination](probleme-bei-der-teamkoordination.md)
<br/>  Teams, die verschiedene Services besitzen, müssen Datenbankänderungen koordinieren, was Kommunikations-Overhead und teamübergreifende Abhängigkeiten schafft.
- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Services werden durch ihr gemeinsam genutztes Datenmodell eng gekoppelt, was es unmöglich macht, einen zu ändern, ohne alle anderen zu berücksichtigen.
- [Ressourcenkonkurrenz](ressourcenkonkurrenz.md)
<br/>  Eine gemeinsam genutzte Datenbank schafft Ressourcenkonkurrenz, da alle Services um dieselben Datenbank-CPU-, Speicher- und I/O-Ressourcen konkurrieren.

## Causes ▼

- [Einschränkungen durch monolithische Architektur](einschraenkungen-durch-monolithische-architektur.md)
<br/>  Monolithische Systeme nutzen naturgemäß eine einzige gemeinsam genutzte Datenbank, und dieses Muster besteht fort, selbst wenn Services aus dem Monolithen extrahiert werden.
- [Bequemlichkeitsgetriebene Entwicklung](bequemlichkeitsgetriebene-entwicklung.md)
<br/>  Das Teilen einer Datenbank ist der einfachste Weg für neue Services, auf bestehende Daten zuzugreifen, was Teams dazu bringt, Bequemlichkeit über ordentliche Entkopplung zu wählen.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Das Management priorisiert schnelle Feature-Lieferung über Datenbankentkopplung, was Muster gemeinsam genutzter Datenbanken fortbestehen lässt, die langfristige Probleme schaffen.

## Detection Methods ○
- **Architekturdiagramme:** Erstellung eines Diagramms der Systemarchitektur zur Identifikation, welche Services oder Komponenten sich eine einzige Datenbank teilen.
- **Datenbankschema-Analyse:** Analyse des Datenbankschemas zur Identifikation, welche Tabellen von mehreren Services oder Komponenten genutzt werden.
- **Entwicklerbefragungen:** Befragung von Entwicklern, ob sie das Gefühl haben, das Datenbankschema ändern zu können, ohne andere Services oder Komponenten zu beeinflussen.

## Examples
Ein Unternehmen hat eine große, monolithische E-Commerce-Anwendung. Die Anwendung besteht aus mehreren verschiedenen Services, einschließlich eines Produktkatalogs, eines Warenkorbs und eines Zahlungsgateways. Alle Services teilen sich eine einzige Datenbank. Wenn das Entwicklungsteam eine Änderung am Datenbankschema für den Produktkatalog vornehmen möchte, muss es darauf achten, den Warenkorb oder das Zahlungsgateway nicht zu brechen. Dies macht es schwierig, Änderungen an der Datenbank vorzunehmen, und führt oft zu Problemen.
