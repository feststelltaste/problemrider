---
title: Lähmung der Schema-Evolution
description: Das Datenbankschema kann nicht modifiziert werden, um neue Anforderungen
  zu unterstützen, aufgrund umfangreicher Abhängigkeiten und fehlendem Migrations-Tooling.
category:
- Code
- Database
related_problems:
- slug: database-schema-design-problems
  similarity: 0.65
- slug: stagnant-architecture
  similarity: 0.65
- slug: maintenance-paralysis
  similarity: 0.6
- slug: entity-attribute-value-overuse
  similarity: 0.6
- slug: system-stagnation
  similarity: 0.6
- slug: custom-report-sprawl
  similarity: 0.6
solutions:
- evolutionary-database-design
- backward-compatible-schema-migrations
- nosql-databases
- schema-registry
- domain-data-versioning
- parallel-run
- change-impact-analysis
- production-like-test-data
- contract-testing
- data-modeling
- typed-schema-extraction
- attribute-usage-analysis
- explicit-extension-points
layout: problem
lang: de
en_slug: schema-evolution-paralysis
---

## Description

Lähmung der Schema-Evolution tritt auf, wenn Datenbankschemas so tief mit Abhängigkeiten, Constraints und Legacy-Design-Entscheidungen verwurzelt sind, dass sie nicht sicher modifiziert werden können, um neue Geschäftsanforderungen oder technische Verbesserungen zu unterstützen. Dies schafft eine Situation, in der die Datenbankstruktur zu einem Engpass für die Systementwicklung wird, was Teams zwingt, um Schemabeschränkungen herumzuarbeiten, statt sie direkt anzugehen. Das Problem ist besonders akut in Legacy-Systemen, wo Jahre angehäufter Änderungen komplexe gegenseitige Abhängigkeiten geschaffen haben.

## Indicators ⟡

- Neue Feature-Anforderungen, die konsequent aufgrund von Datenbankschema-Beschränkungen abgelehnt werden
- Entwicklungsschätzungen, die aufblähen, wenn Datenbankänderungen betroffen sind
- Mehrere Anwendungsschichten, die Workarounds für Schemabeschränkungen implementieren
- Datenbankadministratoren, die hohe Angst vor jeder Schemamodifikationsanfrage äußern
- Fehlende automatisierte Datenbank-Migrationswerkzeuge oder -prozesse im Entwicklungsworkflow
- Schemadokumentation, die veraltet, unvollständig ist oder sich auf Warnungen konzentriert, was nicht geändert werden darf
- Feature-Anfragen, die Denormalisierung oder Datenduplizierung zur Implementierung erfordern

## Symptoms ▲

- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Wenn das Datenbankschema nicht geändert werden kann, erstellen Entwickler aufwendige Workarounds auf Anwendungsebene, um Schemabeschränkungen zu kompensieren.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Features, die Datenbankänderungen erfordern, brauchen viel länger zur Implementierung, wenn Schemamodifikationen vermieden werden, was das gesamte Liefertempo verlangsamt.
- [Architektonische Fehlpassung](architektonische-fehlpassung.md)
<br/>  Ein eingefrorenes Schema divergiert zunehmend von sich entwickelnden Geschäftsanforderungen, was eine wachsende Fehlpassung zwischen Systemfähigkeiten und Geschäftsbedürfnissen schafft.
- [Stagnierende Architektur](stagnierende-architektur.md)
<br/>  Wenn sich das Datenbankschema nicht weiterentwickeln kann, stagniert die Gesamtarchitektur, weil das Datenmodell grundlegend für das Systemdesign ist.
- [Frustration der Stakeholder](frustration-der-stakeholder.md)
<br/>  Geschäfts-Stakeholder werden frustriert, wenn scheinbar einfache Feature-Anfragen aufgrund von Datenbankschema-Beschränkungen Monate dauern.

## Causes ▼

- [Probleme im Datenbankschema-Design](probleme-im-datenbankschema-design.md)
<br/>  Schlechtes initiales Schemadesign schafft starre Strukturen mit komplexen gegenseitigen Abhängigkeiten, die über die Zeit zunehmend schwieriger zu modifizieren werden.
- [Gemeinsam genutzte Datenbank](gemeinsam-genutzte-datenbank.md)
<br/>  Mehrere Services, die dieselbe Datenbank teilen, vervielfachen die Abhängigkeiten von jedem Schemaelement, was Änderungen riskant und schwierig zu koordinieren macht.
- [Angst vor Breaking Changes](angst-vor-breaking-changes.md)
<br/>  Vergangene fehlgeschlagene Schema-Migrationen schaffen Angst vor jeder Datenbankänderung, was Vermeidungsverhalten verstärkt, das zu Lähmung führt.
- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Ohne automatisierte Tests können Teams nicht verifizieren, dass Schemaänderungen bestehende Funktionalität nicht brechen, was Modifikationen zu riskant zum Versuchen macht.

## Detection Methods ○

- Verfolgung der Häufigkeit und Erfolgsrate von Datenbankschema-Änderungsanfragen
- Überwachung der Auswirkung auf die Entwicklungsgeschwindigkeit, wenn Datenbankänderungen für Features erforderlich sind
- Analyse der Anhäufung technischer Schulden in Anwendungscode, der Schemabeschränkungen umgeht
- Befragung von Entwicklungsteams zu datenbankbezogenen Entwicklungsbeschränkungen und Frustrationen
- Überprüfung von Feature-Backlogs auf Elemente, die durch Datenbankschema-Beschränkungen blockiert sind
- Bewertung der Datenbank-Migrations- und Rollback-Fähigkeiten in aktuellen Entwicklungsprozessen
- Untersuchung von Datenbank-Performance-Problemen, die mit Schemaänderungen gelöst werden könnten
- Bewertung von Mustern der Machbarkeitsanalyse von Geschäftsanforderungen für datenbankabhängige Features

## Examples

Eine vor 10 Jahren aufgebaute E-Commerce-Plattform hat ein starres Schema, in dem Produktkategorien als einzelne Fremdschlüsselbeziehung implementiert sind, was die Hierarchie und Mehrfachkategorie-Zuweisung verhindert, die moderne Geschäftsanforderungen verlangen. Die Kundentabelle hat feste Spalten für Adressinformationen, die internationale Versandanforderungen oder mehrere Lieferadressen nicht berücksichtigen können. Wenn das Geschäft Produktbündel, personalisierte Empfehlungen oder Abonnementdienste implementieren möchte, erfordert jedes Feature umfangreiche Workarounds auf Anwendungsebene, weil das Schema nicht modifiziert werden kann. Die Datenbank hat keine Fremdschlüssel-Namenskonventionen, was Abhängigkeitsanalyse nahezu unmöglich macht, und frühere Versuche, das Schema zu modifizieren, führten zu 12-stündigen Ausfällen. Entwicklungsteams verbringen 40 % ihrer Zeit mit der Implementierung komplexer Anwendungslogik, um Schemabeschränkungen zu umgehen, während Geschäfts-Stakeholder frustriert sind, dass „einfache" Feature-Anfragen aufgrund von Datenbankbeschränkungen Monate dauern.
