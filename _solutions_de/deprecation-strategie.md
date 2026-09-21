---
title: Deprecation-Strategie
description: Systematische Kennzeichnung und schrittweise Entfernung veralteter Funktionen.
category:
- Process
- Architecture
problems:
- feature-bloat
- maintenance-overhead
- high-maintenance-costs
- uncontrolled-codebase-growth
- obsolete-technologies
- high-technical-debt
- legacy-api-versioning-nightmare
- rapid-system-changes
- abi-compatibility-issues
layout: solution
lang: de
en_slug: deprecation-strategy
related_solutions:
- slug: api-deprecation-policy
  similarity: 0.75
- slug: strategic-code-deletion
  similarity: 0.75
- slug: feature-usage-measurement
  similarity: 0.7
- slug: system-decommissioning
  similarity: 0.7
- slug: dependency-management-strategy
  similarity: 0.65
- slug: strangler-fig-pattern
  similarity: 0.65
---

## Description

Eine Deprecation-Strategie ist ein formeller, gestufter Prozess zur Stilllegung von Features, APIs oder Codepfaden, die nicht mehr gewollt sind, und ersetzt die informelle Gewohnheit, alte Funktionalität einfach auf unbestimmte Zeit bestehen zu lassen, weil ihre Entfernung riskant erscheint oder niemand die Entscheidung besitzt. Sie definiert explizite Lebenszyklusstufen — Ankündigung, eine Deprecation-Warnperiode, sanfte Entfernung, bei der das Feature versteckt wird, aber der Code als Sicherheitsnetz bestehen bleibt, und schließlich harte Entfernung, bei der der Code gelöscht wird — und verknüpft konkrete Daten und Kommunikationsverpflichtungen mit jeder Stufe. Dies verwandelt „das sollten wir irgendwann wirklich loswerden" in eine verfolgte Zusage mit einer Frist, was genau das ist, was Legacy-Systemen typischerweise fehlt: Ohne eine solche Struktur häuft sich veralteter Code jahrelang still an, weil kein einzelner Stakeholder sowohl die Autorität als auch die Motivation hat, seine Entfernung abzuschließen. In einem Legacy-Modernisierungskontext ist eine Deprecation-Strategie der Mechanismus, der das System tatsächlich verkleinert, statt bloß neue Funktionalität neben alter, ungenutzter Funktionalität hinzuzufügen, die weiterhin Wartungsaufwand, Sicherheitspatching und kognitive Last verbraucht. Ihre Wirkung kumuliert über die Zeit, da jeder erfolgreich abgeschlossene Deprecation-Zyklus die Fläche reduziert, die künftige Modernisierungsarbeit berücksichtigen muss.

## How to Apply ◆

> In Legacy-Systemen bleiben veraltete Features und APIs oft unbegrenzt bestehen, weil kein systematischer Prozess zu ihrer Entfernung existiert — eine Deprecation-Strategie bietet die Disziplin, die Entfernung tatsächlich abzuschließen.

- Etablieren Sie eine formelle Deprecation-Richtlinie, die die Lebenszyklusstufen definiert: Ankündigung, Deprecation-Warnperiode, sanfte Entfernung (Feature versteckt, aber Code erhalten) und harte Entfernung (Code gelöscht).
- Annotieren Sie veralteten Code mit maschinenlesbaren Markierungen (@Deprecated, @obsolete, Deprecation-Kommentaren), die das Deprecation-Datum, die empfohlene Alternative und das geplante Entfernungsdatum enthalten.
- Kommunizieren Sie Deprecation-Pläne an alle Konsumenten (interne Teams und externe API-Nutzer) mit ausreichendem Vorlauf und klarer Migrationsanleitung.
- Verfolgen Sie die Nutzung veralteter Features über Laufzeitüberwachung, um die tatsächliche Auswirkung zu verstehen und Konsumenten zu identifizieren, die noch nicht migriert haben.
- Setzen Sie feste Entfernungsdaten und halten Sie sie ein — eine Deprecation-Strategie, die nie tatsächlich etwas entfernt, bietet keinen Wert.
- Priorisieren Sie die Deprecation von Features, die die höchsten Wartungskosten oder Sicherheitsrisiken tragen, statt zu versuchen, alles auf einmal zu deprecaten.
- Beziehen Sie Deprecation-Bereinigung in reguläre Sprint-Arbeit ein, statt sie auf einen zukünftigen „Bereinigungs-Sprint" zu verschieben, der nie passiert.

## Tradeoffs ⇄

> Eine Deprecation-Strategie ermöglicht kontrollierte Entfernung von Legacy-Ballast, erfordert aber organisatorische Disziplin und klare Kommunikation.

**Vorteile:**

- Reduziert die Wartungslast, indem nicht mehr benötigter Code systematisch entfernt wird, was verhindert, dass die Codebasis unbegrenzt wächst.
- Bietet Konsumenten einen vorhersehbaren Zeitplan, um von veralteten Features wegzumigrieren, was Überraschung und Widerstand reduziert.
- Setzt Entwicklungskapazität frei, die derzeit für die Pflege veralteter Features aufgewendet wird, für Modernisierung und neue Fähigkeiten.
- Reduziert die Sicherheitsexposition, indem alte Codepfade entfernt werden, die ungepatchte Schwachstellen enthalten könnten.

**Kosten und Risiken:**

- Konsumenten veralteter Features könnten sich der Migration widersetzen, besonders wenn die Alternative erheblichen Aufwand auf ihrer Seite erfordert.
- Vorzeitige Deprecation von noch weit genutzten Features kann Nutzer stören und Vertrauen untergraben.
- Die Verfolgung der Nutzung veralteter Features erfordert Instrumentierung, die in Legacy-Systemen möglicherweise nicht existiert.
- Der Deprecation-Prozess selbst erfordert Koordinationsaufwand über Teams und organisatorische Grenzen hinweg.

## How It Could Be

> Das folgende Szenario zeigt, wie eine Deprecation-Strategie die Legacy-Wartungslast reduziert.

Ein B2B-Plattformanbieter erhielt Abwärtskompatibilität mit jeder über 12 Jahre veröffentlichten API-Version, was zu 8 aktiven API-Versionen führte. Die Unterstützung aller Versionen verbrauchte allein 40 Prozent der Backend-Team-Kapazität für Fehlerbehebungen und Sicherheitspatches. Das Team implementierte eine Deprecation-Strategie: jede API-Version würde drei Jahre unterstützt, mit 12 Monaten Vorankündigung vor der Entfernung. Nutzungsüberwachung zeigte, dass Versionen 1 bis 4 zusammen weniger als 50 aktive Clients hatten. Das Team kontaktierte diese Clients direkt, bot Migrationsanleitungen und dedizierten Support und entfernte die vier ältesten Versionen über sechs Monate. Dies eliminierte etwa 60.000 Zeilen Kompatibilitätscode und setzte zwei Entwickler frei, um Vollzeit an Modernisierung zu arbeiten. Die verbleibenden Clients erhielten eine vorhersehbare Lebenszykluszusage, die ihr Vertrauen in die Plattform tatsächlich verbesserte.
