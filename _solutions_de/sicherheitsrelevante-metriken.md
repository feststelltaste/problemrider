---
title: Sicherheitsrelevante Metriken
description: Definition und Erfassung von Metriken zur Quantifizierung des
  Sicherheitsniveaus.
category:
- Security
- Management
problems:
- difficulty-quantifying-benefits
- invisible-nature-of-technical-debt
- monitoring-gaps
- quality-blind-spots
- poor-project-control
- modernization-roi-justification-failure
layout: solution
lang: de
en_slug: security-relevant-metrics
related_solutions:
- slug: security-metrics
  similarity: 0.95
- slug: security-frameworks
  similarity: 0.8
- slug: security-certification
  similarity: 0.8
- slug: threat-modeling
  similarity: 0.8
- slug: security-monitoring
  similarity: 0.8
- slug: security-training
  similarity: 0.75
---

## Description

Sicherheitsrelevante Metriken quantifizieren die Sicherheitslage eines Systems durch Indikatoren wie Schwachstellenalter, Patch-Compliance-Rate und Größe der Angriffsfläche und verwandeln Sicherheit von einem subjektiven Eindruck in etwas, das über ein Portfolio von Anwendungen hinweg verfolgt, getrended und verglichen werden kann. Dies ist besonders wertvoll für Organisationen, die viele Legacy-Systeme gleichzeitig betreuen, wo die Intuition darüber, „welche Anwendung am riskantesten ist", häufig falsch ist, bis die Metriken tatsächlich erfasst und Seite an Seite verglichen werden. Die Automatisierung der Erfassung aus Schwachstellenscannern, Codeanalysewerkzeugen und Vorfallsystemen, und die Verfolgung führender Indikatoren neben nachlaufenden, gibt Teams eine Evidenzbasis zur Priorisierung von Behebungsinvestitionen, statt sie gleichmäßig oder nach wer am lautesten klagt zu verteilen — obwohl sorglos an Anreize gekoppelte Metriken manipuliert werden können und manche echten Risiken sich vollständig gegen Quantifizierung sperren.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie Schlüsselsicherheitsindikatoren, relevant für das Legacy-System, wie Schwachstellenalter, Patch-Compliance und Größe der Angriffsfläche
- Automatisieren Sie die Erfassung von Metriken aus Schwachstellenscannern, Codeanalysewerkzeugen und Incident-Management-Systemen
- Verfolgen Sie führende Indikatoren (z. B. Abschluss von Sicherheitsschulungen, Patch-Rhythmus) neben nachlaufenden Indikatoren (z. B. Vorfallzahl, Auswirkung von Sicherheitsverletzungen)
- Benchmarken Sie Metriken gegen Branchenstandards und historische Basislinien, um Ergebnisse zu kontextualisieren
- Präsentieren Sie Metriken in Formaten, die für verschiedene Zielgruppen angemessen sind: technisches Detail für Teams, Trends für das Management
- Nutzen Sie Metriken, um messbare Sicherheitsverbesserungsziele zu setzen und Fortschritt vierteljährlich zu verfolgen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verwandelt Sicherheit von einer subjektiven Einschätzung in eine messbare Fähigkeit
- Unterstützt die Entwicklung von Business Cases für Sicherheitsinvestitionen in Legacy-Systeme
- Ermöglicht Trendanalyse, die offenbart, ob sich die Sicherheitslage verbessert oder verschlechtert
- Liefert Frühwarnsignale, bevor Sicherheitsprobleme zu Vorfällen werden

**Kosten und Risiken:**
- Die Erfassung bedeutsamer Metriken aus Legacy-Systemen mit begrenzter Instrumentierung kann herausfordernd sein
- Metriken können manipuliert werden, wenn sie ohne angemessenes Design an Anreize gekoppelt sind
- Übermäßiges Vertrauen auf Metriken kann blinde Flecken für schwer quantifizierbare Risiken erzeugen
- Metrikenprogramme erfordern laufende Pflege, um relevant zu bleiben, während sich die Bedrohungslandschaft weiterentwickelt

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Unternehmen, das ein Portfolio von Legacy-Anwendungen betreute, führte ein Sicherheitsmetriken-Programm ein, das Schwachstellendichte pro Anwendung, durchschnittliche Zeit bis zur Behebung kritischer Befunde und Prozentsatz der Anwendungen mit aktuellen Abhängigkeitsversionen verfolgte. Die Metriken offenbarten, dass zwei ihrer 12 Legacy-Anwendungen 73 % aller kritischen Schwachstellen ausmachten und Behebungszeiten hatten, die dreimal länger waren als der Portfolio-Durchschnitt. Diese Daten ermöglichten dem Sicherheitsteam, erfolgreich für priorisierte Modernisierung dieser zwei Anwendungen zu argumentieren, was innerhalb eines Quartals zu einer 50%igen Reduktion der portfolioweiten Anzahl kritischer Schwachstellen führte.
