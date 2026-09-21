---
title: API-Deprecation-Richtlinie
description: Außerbetriebnahme alter Schnittstellen mit Sunset-Headern, Zeitplänen
  und Migrationsanleitungen.
category:
- Architecture
- Process
problems:
- legacy-api-versioning-nightmare
- breaking-changes
- api-versioning-conflicts
- maintenance-overhead
- high-maintenance-costs
- technical-architecture-limitations
layout: solution
lang: de
en_slug: api-deprecation-policy
related_solutions:
- slug: backward-compatible-apis
  similarity: 0.8
- slug: deprecation-strategy
  similarity: 0.75
- slug: backward-compatibility
  similarity: 0.75
- slug: api-versioning-strategy
  similarity: 0.75
- slug: api-gateway
  similarity: 0.75
- slug: compatibility-measurement
  similarity: 0.7
---

## Description

Eine API-Deprecation-Richtlinie ist ein formales, veröffentlichtes Regelwerk, das bestimmt, wie eine alte Schnittstellenversion abgeschaltet wird, und die Phasen definiert, die ein veralteter Endpunkt durchläuft — Ankündigung, Ausgabe von Sunset-Headern, reduzierter Support und endgültige Entfernung —, zusammen mit dem minimalen Zeitfenster, das Konsumenten garantiert wird, bevor etwas bricht. Legacy-Systeme neigen dazu, API-Versionen unbegrenzt anzuhäufen, wenn eine solche Richtlinie fehlt, weil das Entfernen eines alten Endpunkts riskant erscheint, wenn niemand sicher ist, welche Konsumenten noch davon abhängen, sodass Teams jede Version standardmäßig am Leben erhalten und sich die Wartungslast mit jeder neu hinzugefügten Version verstärkt. Eine Deprecation-Richtlinie kehrt diesen Standard um: Statt dass ein Endpunkt für immer lebt, es sei denn, jemand entscheidet aktiv, ihn zu entfernen, wird er nach einem vorhersehbaren Zeitplan abgeschaltet, es sei denn, jemand entscheidet aktiv, ihn zu verlängern, was Nutzungsmonitoring zur Identifikation, welche Konsumenten noch nicht migriert haben, und Kommunikationskanäle — Changelogs, Entwicklerportale, direkte Kontaktaufnahme — erfordert, um den Zeitplan unmöglich zu verpassen zu machen. Die Übernahme der Richtlinie tauscht daher einen festen Governance- und Kommunikations-Overhead gegen eine begrenzte, schrumpfende Wartungsfläche, statt eines stetig wachsenden Satzes von Legacy-Schnittstellenversionen, die jeweils separaten Support benötigen. Dies zählt am meisten dort, wo Legacy-Plattformen zahlreiche parallele API-Generationen angehäuft haben, da die freigesetzte Entwicklungskapazität aus der Abschaltung alter Versionen ist, was den Bau ordentlich designter Ersatz-APIs überhaupt erst erschwinglich macht.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Definieren Sie eine Deprecation-Zeitplan-Richtlinie mit klaren Phasen: Ankündigung, Ausgabe von Sunset-Headern, reduzierter Support und Entfernung
- Fügen Sie HTTP-Sunset-Header und Deprecation-Warnungen zu Antworten von Legacy-API-Endpunkten hinzu
- Veröffentlichen Sie Migrationsanleitungen, die veraltete Endpunkte oder Felder auf ihre Ersatzformen abbilden
- Überwachen Sie die Nutzung veralteter Endpunkte zur Identifikation von Konsumenten, die noch nicht migriert haben
- Kommunizieren Sie Deprecation-Zeitpläne über Changelogs, Entwicklerportale und direkte Kontaktaufnahme mit bekannten Konsumenten
- Erzwingen Sie ein minimales Deprecation-Fenster (z. B. 6-12 Monate), um Konsumenten angemessene Übergangszeit zu geben

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Verhindert unbegrenzte Wartung von Legacy-API-Versionen, die Kosten anhäufen
- Gibt Konsumenten vorhersehbare Zeitpläne zur Planung ihrer Migrationen
- Verringert die Fläche unterstützter Schnittstellen über die Zeit, was das Bug-Risiko senkt

**Kosten und Risiken:**
- Erfordert organisatorische Disziplin zur Durchsetzung von Fristen und tatsächlichen Entfernung veralteter Endpunkte
- Konsumenten mit langsamen Release-Zyklen könnten Schwierigkeiten haben, mit Deprecation-Zeitplänen Schritt zu halten
- Verfrühte Deprecation kann Vertrauen schädigen und Konsumenten zu konkurrierenden Plattformen treiben
- Monitoring- und Kommunikationsinfrastruktur fügt operativen Overhead hinzu

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine E-Commerce-Plattform unterhielt fünf parallele API-Versionen, jede mit leicht unterschiedlichen Datenmodellen. Durch die Einführung einer formalen Deprecation-Richtlinie mit 12-monatigen Sunset-Fenstern und automatisierter Nutzungsverfolgung schaltete das Team über 18 Monate drei Versionen ab. Die verbleibende Wartungslast sank um etwa 40 %, und die freigesetzte Entwicklungskapazität wurde in den Bau der API der nächsten Generation mit ordentlicher Versionierungsunterstützung von Anfang an umgeleitet.
