---
title: Logging
description: Umsetzung umfassender Protokollierung und Überwachung des
  Systemverhaltens.
category:
- Operations
problems:
- debugging-difficulties
- monitoring-gaps
- slow-incident-resolution
- inadequate-error-handling
- unpredictable-system-behavior
- logging-configuration-issues
- silent-data-corruption
- log-spam
layout: solution
lang: de
en_slug: logging
related_solutions:
- slug: error-logging
  similarity: 0.85
- slug: monitoring
  similarity: 0.8
- slug: platform-independent-logging-frameworks
  similarity: 0.8
- slug: error-handling
  similarity: 0.8
- slug: error-logs
  similarity: 0.8
- slug: logging-and-monitoring
  similarity: 0.8
---

## Description

Logging ist die Praxis, ein System zu instrumentieren, um strukturierte, kontextbezogene Information über sein eigenes Verhalten — empfangene Anfragen, getroffene Entscheidungen, aufgetretene Fehler — aufzuzeichnen, während es läuft, sodass rekonstruiert werden kann, was tatsächlich passiert ist, statt nur erschlossen oder erraten zu werden. Effektives Logging kombiniert konsistente Schweregrade, strukturierte Felder wie Anfrage- und Korrelations-Identifikatoren, die erlauben, eine einzelne Transaktion über Komponenten hinweg zu verfolgen, und zentralisierte Aggregation, die die resultierenden Datensätze durchsuchbar macht statt über einzelne Maschinen verstreut. Legacy-Systeme sitzen häufig an einem von zwei wenig hilfreichen Extremen: Entweder protokollieren sie fast nichts über ein bloßes Prozess-lebt-Signal hinaus, mit abgefangenen und still verschluckten Fehlern, oder sie protokollieren so umfangreich und ohne Struktur, dass echt wichtige Ereignisse nicht von Routinerauschen zu unterscheiden sind — in beiden Fällen bleibt dem Team, wenn etwas schiefgeht, nur übrig, Verhalten aus Gedächtnis und Spekulation statt aus Evidenz zu rekonstruieren. Richtiges Logging in eine solche Codebasis nachzurüsten ist invasiv, da Instrumentierung an vielen bestehenden Einstiegspunkten, Fehlerbehandlern und Integrationsgrenzen hinzugefügt werden muss, die nie mit Observability im Blick entworfen wurden, aber es zielt direkt auf die Debugging-Schwierigkeiten und langsame Vorfallauflösung ab, die sonst chronisch in schlecht instrumentierten Legacy-Systemen sind. Weil Logs auch zu einer Belastung werden können, muss derselbe Aufwand sicherstellen, dass sensible Daten — Zugangsdaten, Tokens, persönliche Informationen — nie in sie geschrieben werden, was leicht übersehen wird, wenn Logging-Anweisungen ad hoc unter Zeitdruck hinzugefügt werden, statt einer bewussten Richtlinie zu folgen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Etablieren Sie konsistente Log-Level (DEBUG, INFO, WARN, ERROR) und definieren Sie Richtlinien dafür, wann jedes Level angemessen ist
- Fügen Sie strukturiertes Logging mit kontextbezogenen Feldern hinzu (Anfrage-ID, Nutzer-ID, Komponentenname) statt Freitextmeldungen
- Instrumentieren Sie zuerst kritische Pfade im Legacy-System: Einstiegspunkte, Fehlerbehandler und Integrationsgrenzen
- Zentralisieren Sie Logs mittels eines Log-Aggregationssystems, damit sie über alle Komponenten hinweg durchsuchbar sind
- Beziehen Sie Korrelations-IDs ein, um Anfragen über Dienstgrenzen hinweg zu verfolgen
- Überprüfen und verringern Sie übermäßiges Logging, das Rauschen erzeugt, während Sie Logging zu stillen Fehlerpfaden hinzufügen
- Stellen Sie sicher, dass sensible Daten nie protokolliert werden: Maskieren Sie personenbezogene Daten, Zugangsdaten und Sicherheitstokens

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verringert die Zeit zur Diagnose von Produktionsproblemen in Legacy-Systemen dramatisch
- Bietet Sichtbarkeit auf Systemverhalten, das sonst undurchsichtig sein könnte
- Ermöglicht proaktive Problemerkennung durch log-basierte Alarmierung
- Schafft eine Prüfspur für Compliance- und Sicherheitsuntersuchungen

**Kosten und Risiken:**
- Übermäßiges Logging verschlechtert die Performance und erhöht Speicherkosten
- Das Protokollieren sensibler Daten kann Sicherheits- und Compliance-Verstöße erzeugen
- Schlecht strukturierte Logs sind schwer abzufragen und können schlimmer sein als gar keine Logs
- Das Nachrüsten von Logging in eine Legacy-Codebasis erfordert das Anfassen vieler Dateien

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Telekommunikationsunternehmen hatte ein Legacy-Abrechnungssystem, das gelegentlich falsche Rechnungen produzierte, aber die Grundursache war unmöglich zu bestimmen, weil das System minimales Logging hatte. Die Fehlerbehandlung bestand darin, alle Exceptions abzufangen und still fortzufahren. Das Team fügte strukturiertes Logging an Schlüsselentscheidungspunkten in der Abrechnungspipeline hinzu, mit Korrelations-IDs, die jede Rechnung mit ihren Verarbeitungsschritten verknüpften. Innerhalb von zwei Wochen nach dem Deployment des erweiterten Loggings identifizierten sie eine Race Condition im Rabattberechnungsmodul, die monatelang still Abrechnungsdaten beschädigt hatte.
