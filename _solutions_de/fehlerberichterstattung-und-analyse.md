---
title: Fehlerberichterstattung und -analyse
description: Systematische Erfassung, Analyse und Behebung von Fehlern und Problemen.
category:
- Process
- Operations
problems:
- increased-error-rates
- slow-incident-resolution
- monitoring-gaps
- debugging-difficulties
- constant-firefighting
- high-defect-rate-in-production
- delayed-bug-fixes
- delayed-issue-resolution
layout: solution
lang: de
en_slug: error-reporting-and-analysis
related_solutions:
- slug: error-logs
  similarity: 0.9
- slug: error-handling
  similarity: 0.85
- slug: error-logging
  similarity: 0.85
- slug: root-cause-analysis
  similarity: 0.8
- slug: exceptions
  similarity: 0.8
- slug: logging
  similarity: 0.8
---

## Description

Fehlerberichterstattung und -analyse führt dediziertes Tooling ein — Services wie Sentry, Rollbar oder Bugsnag —, das unbehandelte Exceptions und kritische Fehler automatisch mit vollem Kontext erfasst, Vorkommen desselben zugrunde liegenden Defekts dedupliziert und gruppiert, und sie durch einen definierten Workflow mit Schweregradklassifikationen, Eigentümerschaft und Lösungsverfolgung leitet. Dies geht über rohes Logging hinaus, indem einzelne Fehlervorkommen in verwaltete Issues verwandelt werden: Statt eines Stroms von Log-Zeilen, der manuell korreliert werden muss, sieht das Team eine nach Häufigkeit und Auswirkung sortierte Liste unterschiedlicher Fehlergruppen. Legacy-Systeme, die zuvor auf eine informelle Mischung aus Nutzerbeschwerden, Support-Tickets und Entwicklerbeobachtungen angewiesen waren, um von Produktionsproblemen zu erfahren, entdecken typischerweise, sobald solches Tooling eingeführt wird, dass eine kleine Anzahl von Fehlergruppen die überwältigende Mehrheit der Produktionsausfälle ausmacht — Defekte, die lange Zeit stückweise als Nutzerbeschwerden gemeldet wurden, ohne dass jemand sie mit einer einzigen Ursache verband. Weil dies erfordert, die Legacy-Anwendung zu instrumentieren, um umfassende Fehlerberichte zu erzeugen, und einen bezahlten oder selbst gehosteten Tracking-Service zu integrieren, sind die Hauptkosten der Integrationsaufwand selbst und die Notwendigkeit, die Schweregradklassifikation sorgfältig genug zu kalibrieren, um weder Alarmmüdigkeit noch verpasste kritische Probleme zu vermeiden, zusammen mit Aufmerksamkeit dafür, welche Nutzerdaten in den Berichten landen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Implementieren Sie einen Fehler-Tracking-Service (z. B. Sentry, Rollbar, Bugsnag), der Fehler automatisch erfasst, dedupliziert und gruppiert
- Instrumentieren Sie die Legacy-Anwendung, um unbehandelte Exceptions und kritische Fehler mit vollständigen Stack Traces und Kontext zu melden
- Definieren Sie Schweregradklassifikationen und Reaktionszeiterwartungen für jede Schweregradstufe
- Erstellen Sie Workflows, die Fehlerberichte basierend auf der betroffenen Komponente an das passende Team leiten
- Verfolgen Sie Fehlerlösungskennzahlen: Zeit bis zur Bestätigung, Zeit bis zur Lösung und Wiederauftrittsrate
- Führen Sie regelmäßige Fehlertrendüberprüfungen durch, um systemische Probleme hinter einzelnen Fehlerberichten zu identifizieren
- Integrieren Sie Fehlerberichterstattung mit dem Issue-Tracking-System, sodass Fehlermuster zu handlungsfähigen Arbeitsposten werden

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Verwandelt Fehlerbehandlung von reaktiver Vorfallreaktion in systematische Qualitätsverbesserung
- Automatische Deduplizierung und Gruppierung verhindert, dass derselbe Fehler mehrfach untersucht wird
- Bietet datengetriebene Priorisierung, welche Fehler die größte Auswirkung haben
- Schafft Verantwortlichkeit für Fehlerlösung durch Verfolgung und Kennzahlen
- Reduziert die Lösungszeit, indem vollständiger Fehlerkontext vorab bereitgestellt wird

**Kosten und Risiken:**
- Fehler-Tracking-Services fügen Kosten hinzu und erfordern Integrationsaufwand mit Legacy-Systemen
- Hohe Fehlervolumina können Teams überwältigen, wenn die Schweregradklassifikation nicht ordentlich kalibriert ist
- Übermäßige Berichterstattung kann Alarmmüdigkeit verursachen, was Teams dazu bringt, echt wichtige Fehler zu ignorieren
- Die Instrumentierung von Legacy-Code für Fehlerberichterstattung könnte das Anfassen vieler Dateien erfordern
- Datenschutzbedenken, wenn Fehlerberichte Nutzerdaten erfassen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-Gesundheitsanwendung hatte Fehler, gemeldet über eine Mischung aus E-Mail-Benachrichtigungen, Nutzer-Support-Tickets und Entwicklerbeobachtungen. Es gab keine einheitliche Sicht auf Fehlerhäufigkeit oder Auswirkung. Das Team integrierte Sentry in die Anwendung, was sofort enthüllte, dass die Top-10-Fehlergruppen 80 Prozent aller Produktionsfehler ausmachten. Drei davon waren Nullreferenz-Fehler im Patiententerminplanungsmodul, die als Nutzerbeschwerden gemeldet, aber nie mit Codedefekten verbunden worden waren. Durch die Behebung nur dieser drei Fehlergruppen über zwei Sprints reduzierte das Team die gesamte Produktionsfehlerrate um 60 Prozent und verringerte die Arbeitslast des Support-Teams erheblich.
