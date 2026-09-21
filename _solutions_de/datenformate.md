---
title: Datenformate
description: Nutzung standardisierter und weit verbreiteter Datenformate für den
  Datenaustausch.
category:
- Architecture
problems:
- integration-difficulties
- cross-system-data-synchronization-problems
- technology-stack-fragmentation
- poor-interfaces-between-applications
- vendor-lock-in
- endianness-conversion-overhead
- alignment-and-padding-issues
layout: solution
lang: de
en_slug: data-formats
related_solutions:
- slug: standardized-data-formats
  similarity: 0.95
- slug: data-format-conversion
  similarity: 0.85
- slug: backward-compatible-data-formats
  similarity: 0.8
- slug: data-ecosystems
  similarity: 0.75
- slug: data-strategy
  similarity: 0.75
- slug: canonical-data-model
  similarity: 0.75
---

## Description

Diese Lösung ersetzt proprietäre, benutzerdefinierte oder undokumentierte Datenformate, die für den Austausch zwischen Systemen genutzt werden, durch weit verbreitete, gut spezifizierte Standards — JSON für APIs, CSV für Batch-Übertragungen, Parquet für Analyse-Workloads —, gewählt entsprechend dem Austausch-Anwendungsfall und begleitet von einem veröffentlichten Schema in einer Standard-Schemasprache. Der zugrunde liegende Mechanismus ist unkompliziert: Ein Standardformat kommt mit breiter Tooling-, Bibliotheks- und Dokumentationsunterstützung über Sprachen und Plattformen hinweg, sodass jedes neue System, das am Datenaustausch teilnehmen muss, dies mit Standardbibliotheken tun kann statt mit einem maßgeschneiderten Parser. Dies ist in Legacy-Kontexten überproportional wertvoll, weil vor Jahrzehnten definierte benutzerdefinierte Formate häufig nur von einer einzigen verbliebenen Person verstanden werden, wenn überhaupt, was jede neue Integration in eine mehrwöchige Reverse-Engineering-Übung statt in eine Routineaufgabe verwandelt. Die Migration weg von einem proprietären Format ist selten ein einzelner Umstieg; sie erfolgt typischerweise, indem das Legacy-System sowohl das alte als auch das neue Format für eine Übergangsperiode unterstützt, mit Formatvalidierung an der Grenze, um fehlerhafte Daten früh zu erkennen. Der Gewinn ist nicht nur schnellere Integration, sondern reduzierter Vendor Lock-in, da ein um offene, standardisierte Formate herum gebautes System nicht an welches Tooling oder welche Expertise auch immer gebunden ist, die zufällig noch seine ursprüngliche proprietäre Kodierung versteht.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Ersetzen Sie proprietäre oder benutzerdefinierte Datenformate durch weit verbreitete Standards (JSON, XML, CSV, Parquet) für den Datenaustausch
- Wählen Sie Formate basierend auf dem Anwendungsfall: JSON für APIs, CSV für Batch-Exporte, Parquet für Analyse-Workloads
- Definieren und veröffentlichen Sie Schemata für alle Austauschformate mittels Standard-Schemasprachen (JSON Schema, XSD)
- Migrieren Sie Legacy-Systeme schrittweise, indem Sie während des Übergangs sowohl proprietäre als auch standardisierte Formate unterstützen
- Nutzen Sie Formatvalidierung an Systemgrenzen, um fehlerhafte Daten früh abzulehnen
- Dokumentieren Sie Formatentscheidungen und ihre Begründung in Architekturentscheidungsaufzeichnungen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Breite Tooling- und Bibliotheksunterstützung über Sprachen und Plattformen hinweg reduziert Integrationsaufwand
- Senkt die Einstiegshürde für neue Systeme, am Datenaustausch teilzunehmen
- Reduziert Vendor Lock-in durch Vermeidung proprietärer Formate

**Kosten und Risiken:**
- Standardformate repräsentieren möglicherweise domänenspezifische Datenstrukturen nicht effizient
- Die Migration von proprietären Formaten erfordert Konvertierungsaufwand und Handhabung von Abwärtskompatibilität
- Generische Formate wie JSON fehlt eingebaute Schema-Durchsetzung, was zusätzliches Tooling erfordert
- Manche Legacy-Systeme haben möglicherweise keine Bibliotheken für moderne Standardformate verfügbar

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Regierungsbehörde tauschte Bürgerdatensätze zwischen Abteilungen mittels eines vor 15 Jahren definierten benutzerdefinierten Binärformats aus. Nur der ursprüngliche Entwickler verstand die Formatspezifikation, und die Integration neuer Abteilungen erforderte Wochen benutzerdefinierter Parser-Entwicklung. Durch die Migration zu JSON mit einem veröffentlichten JSON Schema sanken neue Abteilungsintegrationen von Wochen auf Tage, und drei Standard-Analytics-Werkzeuge konnten die Daten ohne benutzerdefinierten Code konsumieren.
