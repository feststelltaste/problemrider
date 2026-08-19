---
title: Fehlerprotokollierung
description: Erfassung und Speicherung von Fehlern und Ausnahmen.
category:
- Operations
- Code
problems:
- monitoring-gaps
- debugging-difficulties
- slow-incident-resolution
- inadequate-error-handling
- excessive-logging
- logging-configuration-issues
- log-spam
layout: solution
lang: de
en_slug: error-logging
related_solutions:
- slug: logging
  similarity: 0.85
- slug: error-handling
  similarity: 0.85
- slug: error-reporting-and-analysis
  similarity: 0.85
- slug: error-logs
  similarity: 0.85
- slug: platform-independent-logging-frameworks
  similarity: 0.75
- slug: monitoring
  similarity: 0.75
---

## Description

Fehlerprotokollierung ist die Disziplin, jeden Fehler und jede Ausnahme, auf die das laufende System stößt, in einem standardisierten, strukturierten Datensatz zu erfassen — Zeitstempel, Schweregrad, Fehlertyp, Meldung, Stack Trace, Korrelations-ID und relevanter Geschäftskontext — und ihn an einen zentralen Ort zu leiten, wo er über Services hinweg durchsucht und korreliert werden kann. Legacy-Systeme kommen selten mit dieser Disziplin bereits etabliert an; stattdessen protokollieren sie über Module hinweg inkonsistent, wobei ein Teil der Codebasis eine Logging-Bibliothek nutzt, ein anderer in Flat Files schreibt und ein weiterer direkt auf Standard Error ausgibt, sodass die Rekonstruktion dessen, was während eines Vorfalls geschah, bedeutet, mehrere inkompatible Quellen manuell zusammenzufügen. Auf ein einziges strukturiertes Format zu standardisieren, typischerweise JSON, und die Sammlung mittels einer Log-Aggregationsplattform zu zentralisieren verwandelt Fehlerprotokollierung von verstreutem forensischen Ballast in einen durchsuchbaren Datensatz, der sowohl individuelle Vorfallreaktion als auch Mustererkennung über die Zeit unterstützt. Dieselbe Zentralisierung, die Diagnose schneller macht, erhöht auch die Einsätze dafür, was in einem Log-Eintrag landet, da beiläufig erfasste sensible Daten in dem Moment zu einer Datenschutz- und Sicherheitsbelastung werden, in dem Logs aggregiert und breit durchsuchbar gemacht werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Standardisieren Sie das Fehlerprotokollformat, um Zeitstempel, Schweregrad, Fehlertyp, Meldung, Stack Trace und Korrelations-ID einzubeziehen
- Implementieren Sie strukturiertes Logging (JSON-Format), um automatisiertes Parsen und Analyse zu ermöglichen
- Konfigurieren Sie angemessene Log-Level, damit Fehler sich vom informativen Rauschen abheben
- Zentralisieren Sie die Log-Sammlung mittels Werkzeugen wie ELK Stack, Splunk oder Datadog für serviceübergreifende Sichtbarkeit
- Fügen Sie Fehlerprotokollen kontextuelle Daten hinzu: Nutzer-ID, Anfrage-ID, betroffene Entität und relevante Parameter
- Implementieren Sie Log-Rotation und Aufbewahrungsrichtlinien, um Speicher zu verwalten, während notwendige Historie bewahrt wird
- Richten Sie Alarme für Fehlerprotokollmuster ein, um Probleme proaktiv statt reaktiv zu erkennen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Bietet die forensischen Daten, die nötig sind, um Produktionsprobleme zu diagnostizieren und zu beheben
- Ermöglicht Mustererkennung über Fehlerprotokolle hinweg, um systemische Probleme zu identifizieren
- Unterstützt Compliance- und Audit-Anforderungen durch umfassende Fehleraufzeichnungen
- Reduziert die mittlere Lösungszeit, indem Reagierenden der benötigte Kontext gegeben wird

**Kosten und Risiken:**
- Exzessives Logging kann die Anwendungsperformance beeinträchtigen und erheblichen Speicher verbrauchen
- Sensible Daten in Fehlerprotokollen schaffen Sicherheits- und Datenschutzrisiken, wenn nicht sorgfältig gehandhabt
- Schlecht strukturierte Logs sind schwer im großen Maßstab zu durchsuchen und zu analysieren
- Log-Infrastruktur erfordert eigene Überwachung und Pflege
- Teams könnten sich auf Logs als Ersatz für ordentliches Monitoring und Alarmierung verlassen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Zahlungssystem protokollierte Fehler inkonsistent: Manche Module nutzten log4j, andere schrieben in Flat Files, und manche gaben auf stderr aus. Als ein Transaktionsverarbeitungsfehler stille Fehlschläge verursachte, verbrachte das Team drei Tage damit, Informationen aus sechs unterschiedlichen Log-Quellen zu korrelieren, um das Problem zu diagnostizieren. Sie standardisierten auf SLF4J mit einem JSON-Appender, zentralisierten alle Logs in Elasticsearch und etablierten eine Logging-Richtlinie, die Korrelations-IDs und Transaktionskontext in jedem Fehlerprotokoll-Eintrag verlangte. Als das nächste Mal ein ähnliches Problem auftrat, identifizierte das Team die Ursache in 20 Minuten, indem es nach der betroffenen Transaktions-ID über alle Services in Kibana suchte.
