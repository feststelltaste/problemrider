---
title: Code-Metriken
description: Sammlung und Analyse quantitativer Messgrößen zur Bewertung der Codequalität.
category:
- Code
- Process
problems:
- high-technical-debt
- invisible-nature-of-technical-debt
- lower-code-quality
- difficult-code-comprehension
- complex-and-obscure-logic
- monolithic-functions-and-classes
- bloated-class
- quality-degradation
- automated-tooling-ineffectiveness
- excessive-class-size
layout: solution
lang: de
en_slug: code-metrics
related_solutions:
- slug: static-analysis-and-linting
  similarity: 0.85
- slug: business-metrics
  similarity: 0.8
- slug: code-quality-gates
  similarity: 0.8
- slug: code-review-process-reform
  similarity: 0.8
- slug: technical-debt-backlog
  similarity: 0.8
- slug: code-coverage-analysis
  similarity: 0.8
---

## Description

Code-Metriken sind quantitative Messungen von Quellcode-Eigenschaften — zyklomatische Komplexität, Klassen- und Methodenlänge, Kopplung zwischen Komponenten, Duplizierungsprozentsatz —, automatisch durch statische Analysewerkzeuge gesammelt und über die Zeit verfolgt, um ein objektives, vergleichbares Bild der Codequalität zu geben. Statt sich auf das subjektive Gefühl der Entwickler zu verlassen, dass „dieser Teil der Codebasis schlecht ist", übersetzen Metriken diese Intuition in Zahlen, die über Module hinweg verglichen, als Trends verfolgt und an Menschen kommuniziert werden können, die den Code nicht selbst lesen. Diese Übersetzung ist genau das, was Legacy-Modernisierungsbemühungen brauchen, weil technische Schulden in einem Legacy-System sonst für die Stakeholder, die entscheiden, wohin Investition fließt, weitgehend unsichtbar sind — sie können die verworrene Logik oder brüchige Kopplung nicht sehen, die Entwickler bei jeder Codeberührung fühlen, aber sie können ein Dashboard verstehen, das zeigt, dass eine Handvoll Klassen einen unverhältnismäßigen Anteil an Komplexität und Defekten ausmacht. Die Kombination von Metriken mit Änderungshäufigkeitsdaten ist es, was sie umsetzbar statt nur beschreibend macht, da sie die spezifische Schnittmenge von „komplex" und „häufig geändert" identifiziert, wo Refaktorierungsinvestition den höchsten Ertrag bringt, statt Aufwand dünn und gleichmäßig über eine gesamte Legacy-Codebasis zu verteilen. Das Verfolgen von Metriktrends über den Verlauf einer Refaktorierungsinitiative gibt einem Team außerdem konkrete Beweise für Verbesserung, um sie Stakeholdern zu zeigen, und verwandelt ein qualitatives Argument über Codequalität in ein quantitatives. Das entsprechende Risiko ist, dass Metriken manipuliert werden können — Komplexitätszahlen können verringert werden, indem Methoden mechanisch aufgeteilt werden, ohne echten Gewinn an Verständlichkeit — und wichtige Qualitätsattribute wie Namensklarheit oder architektonische Passung werden von keiner automatisierten Metrik überhaupt erfasst.

## How to Apply ◆

> In Legacy-Systemen machen Code-Metriken das Unsichtbare sichtbar — sie quantifizieren die technischen Schulden und Komplexität, die Entwickler fühlen, aber Stakeholdern nicht leicht kommunizieren können.

- Integrieren Sie Code-Metriken-Werkzeuge (SonarQube, CodeClimate, NDepend oder sprachspezifische Alternativen) in die CI-Pipeline, um Metriken automatisch bei jedem Build zu verfolgen.
- Fokussieren Sie sich auf eine kleine Menge umsetzbarer Metriken: zyklomatische Komplexität, Klassen- und Methodenlänge, Kopplung zwischen Komponenten und Duplizierungsprozentsatz.
- Etablieren Sie Baselines für die aktuellen Metriken der Legacy-Codebasis und setzen Sie Verbesserungsziele, die Refaktorierungsprioritäten leiten.
- Nutzen Sie Metriken, um die schlimmsten Hotspots zu identifizieren — die Klassen und Methoden mit der höchsten Komplexität und den häufigsten Änderungen — als vorrangige Ziele für Refaktorierung.
- Präsentieren Sie Metriktrends an Stakeholder, um technische Schulden sichtbar zu machen und Investition in Codequalitätsverbesserung zu rechtfertigen.
- Kombinieren Sie Code-Metriken mit Änderungshäufigkeitsdaten, um Verbesserungsbemühungen auf Code zu fokussieren, der sowohl komplex als auch häufig geändert wird, was den Ertrag der Refaktorierungsinvestition maximiert.
- Setzen Sie Qualitäts-Gates, die verhindern, dass neuer Code Metrik-Regressionen einführt, um sicherzustellen, dass sich die Codebasis über die Zeit verbessert, statt weiter zu degradieren.

## Tradeoffs ⇄

> Code-Metriken bieten objektive Qualitätsindikatoren, können aber manipuliert werden und müssen im Kontext interpretiert werden.

**Vorteile:**

- Macht Codequalität sichtbar und messbar, was datengetriebene Entscheidungen darüber ermöglicht, wo Refaktorierungsaufwand investiert werden soll.
- Bietet objektive Evidenz zur Kommunikation technischer Schulden an nicht-technische Stakeholder, die das Problem sonst möglicherweise nicht wahrnehmen.
- Identifiziert die problematischsten Bereiche der Codebasis durch quantitative Analyse statt anekdotischer Evidenz.
- Verfolgt Qualitätsverbesserung über die Zeit und demonstriert die Auswirkung von Refaktorierungsinvestitionen.

**Kosten und Risiken:**

- Metriken können manipuliert werden — Entwickler können zyklomatische Komplexität verringern, indem sie Methoden aufteilen, ohne die tatsächliche Verständlichkeit zu verbessern.
- Übermäßige Betonung von Metriken kann zur Optimierung für Zahlen statt echter Codequalität führen.
- Zu aggressives Setzen von Metrikschwellen auf einer Legacy-Codebasis kann das Team mit Verstößen überwältigen, die nicht alle adressiert werden können.
- Manche wichtigen Qualitätsattribute (Namensqualität, Designangemessenheit, Geschäftsausrichtung) können von automatisierten Metriken nicht erfasst werden.

## How It Could Be

> Das folgende Szenario demonstriert, wie Code-Metriken Legacy-Modernisierungsprioritäten leiten.

Das Legacy-ERP-System eines Fertigungsunternehmens hatte 2 Millionen Codezeilen, und das Entwicklungsteam wusste, dass es Qualitätsprobleme hatte, konnte sich aber nicht einigen, wo Verbesserungsbemühungen fokussiert werden sollten. Nach der Integration von SonarQube entdeckte das Team, dass 15 Klassen (0,3 % der Codebasis) 40 % aller Komplexitätsverstöße und 35 % der Produktionsdefekte ausmachten. Diese „toxischen" Klassen — einschließlich eines 12.000-Zeilen-`OrderProcessor` und eines 8.000-Zeilen-`InventoryManager` — wurden zu den expliziten Zielen einer sechsmonatigen Refaktorierungsinitiative. Durch monatliches Verfolgen von Komplexitätsmetriken demonstrierte das Team dem Management, dass die durchschnittliche zyklomatische Komplexität geänderter Klassen im Verlauf der Initiative von 45 auf 12 sank, und die Defektrate in refaktorierten Bereichen sank proportional. Das Metriken-Dashboard wurde zu einem ständigen Tagesordnungspunkt in Management-Überprüfungen, was technische Schulden zu einem sichtbaren Geschäftsanliegen statt einer Entwicklerbeschwerde machte.
