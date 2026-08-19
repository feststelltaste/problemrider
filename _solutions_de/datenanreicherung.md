---
title: Datenanreicherung
description: Ergänzung von Daten mit zusätzlichen Informationen aus externen Quellen.
category:
- Database
- Dependencies
problems:
- poor-domain-model
- feature-gaps
- data-migration-complexities
- cross-system-data-synchronization-problems
- silent-data-corruption
layout: solution
lang: de
en_slug: data-enrichment
related_solutions:
- slug: data-quality-checks
  similarity: 0.7
- slug: data-integration
  similarity: 0.65
- slug: data-integrity
  similarity: 0.65
- slug: data-deduplication
  similarity: 0.6
- slug: continuous-data-verification
  similarity: 0.6
- slug: data-strategy
  similarity: 0.6
---

## Description

Datenanreicherung ergänzt bestehende Datensätze mit zusätzlichen Attributen aus externen Quellen — Referenzdatenbanken, kommerzielle Datenanbieter oder andere interne Systeme —, statt zu erfordern, dass fehlende oder veraltete Informationen durch manuelle Neueingabe erfasst werden. Eine Anreicherungspipeline läuft typischerweise bei der Aufnahme oder nach Zeitplan, gleicht Legacy-Datensätze mit der externen Quelle mittels verfügbarer Identifikatoren ab und schreibt die resultierenden Felder entweder als Ergänzung zum ursprünglichen Datensatz oder, sicherer, in einen separaten Speicher, der bewahrt, welche Werte aus dem Legacy-System stammten und welche später hinzugefügt wurden. Diese Technik ist für Legacy-Systeme besonders relevant, weil ihre Daten häufig unter älteren, eingeschränkteren Geschäftsprozessen erfasst wurden und über Jahre des Betriebs verfallen oder veraltet sind, was Lücken schafft — fehlende Klassifikationen, veraltete Kontaktdaten, fehlende Geolokation —, die neue Features oder Analytics blockieren, die die Organisation nun auf alten Daten aufbauen möchte. Weil Anreicherung eine Abhängigkeit von der Verfügbarkeit, Qualität und Aktualisierungstaktung einer externen Quelle einführt, brauchen Pipelines Validierung gegen Geschäftsregeln und eine Fallback-Strategie für den Fall, dass diese Quelle nicht erreicht werden kann. Die Herkunft zwischen ursprünglichen und angereicherten Werten zu bewahren ist essenziell, da es die Anreicherung auditierbar und rückgängig machbar hält, falls sich die externe Quelle später als falsch erweist.

## How to Apply ◆

- Identifizieren Sie Lücken in Legacy-Daten, die die Systemeffektivität reduzieren (z. B. fehlende Geolokation, veraltete Kontaktinformationen, unvollständige Klassifikation).
- Integrieren Sie externe Datenquellen (APIs, Referenzdatenbanken, Drittanbieter-Services), um Legacy-Daten zu ergänzen.
- Bauen Sie Anreicherungspipelines, die bei der Aufnahme oder nach Zeitplan laufen und abgeleitete oder ergänzende Felder zu Legacy-Datensätzen hinzufügen.
- Validieren Sie angereicherte Daten gegen Geschäftsregeln, um zu verhindern, dass Fehler ins Legacy-System eingeführt werden.
- Speichern Sie Anreicherungsergebnisse separat von Originaldaten, um Datenherkunft zu bewahren und Rollback zu ermöglichen.
- Überwachen Sie die Anreicherungsqualität über die Zeit und etablieren Sie Fallback-Strategien für den Fall, dass externe Quellen nicht verfügbar sind.

## Tradeoffs ⇄

**Vorteile:**
- Verbessert die Qualität und Vollständigkeit von Legacy-Daten, ohne manuelle Dateneingabe zu erfordern.
- Ermöglicht neue Features und Analytics-Fähigkeiten, die die Legacy-Daten allein nicht unterstützen können.
- Kann Daten korrigieren oder ergänzen, die über Jahre des Legacy-Systembetriebs verfallen sind.

**Kosten:**
- Führt Abhängigkeiten von externen Datenquellen mit ihren eigenen Verfügbarkeits- und Qualitätsbedenken ein.
- Anreicherungsprozesse fügen der Datenpipeline Komplexität hinzu und erfordern laufende Pflege.
- Datenschutz- und Compliance-Überlegungen können einschränken, welche externen Daten integriert werden können.
- Falsche Anreicherung kann Fehler einführen, die schwer von Originaldaten zu unterscheiden sind.

## How It Could Be

Eine Legacy-Kundendatenbank enthält Millionen von Datensätzen, angehäuft über zwanzig Jahre, viele mit unvollständigen Adressen, fehlenden Branchenklassifikationen und veralteten Kontaktinformationen. Das Team baut eine Anreicherungspipeline, die Kundendatensätze mit einem kommerziellen Geschäftsdatenanbieter abgleicht, fehlende Felder ausfüllt und Datensätze markiert, bei denen gespeicherte Informationen im Konflikt mit externen Quellen stehen. Die Anreicherungsergebnisse werden in einer separaten, mit den Originaldatensätzen verknüpften Tabelle gespeichert, was die Fähigkeit bewahrt zu prüfen, was aus dem Legacy-System stammte gegenüber dem, was angereichert wurde. Vertriebsteams profitieren sofort von verbessertem Targeting, und die Datenqualitätsverbesserungen ermöglichen ein Kundensegmentierungs-Feature, das zuvor wegen unvollständiger Daten unmöglich war.
