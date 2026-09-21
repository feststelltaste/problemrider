---
title: A/B-Testing
description: Vergleich verschiedener Versionen, um das Nutzererlebnis zu optimieren.
category:
- Testing
- Requirements
problems:
- poor-user-experience-ux-design
- customer-dissatisfaction
- negative-user-feedback
- declining-business-metrics
- user-frustration
- difficulty-quantifying-benefits
- feature-bloat
layout: solution
lang: de
en_slug: a-b-testing
related_solutions:
- slug: user-centered-design
  similarity: 0.8
- slug: consistent-user-interface
  similarity: 0.75
- slug: adaptive-behavior
  similarity: 0.75
- slug: risk-analysis
  similarity: 0.75
- slug: cross-version-testing
  similarity: 0.75
- slug: usability-tests
  similarity: 0.75
---

## Description

A/B-Testing ist ein kontrolliertes Experiment, bei dem zwei oder mehr Varianten eines Features, Workflows oder einer Schnittstelle gleichzeitig verschiedenen Segmenten von Live-Nutzern gezeigt werden, wobei die resultierenden Verhaltens- oder Geschäftskennzahlen verglichen werden, um zu bestimmen, welche Variante besser abschneidet. Statt durch interne Überzeugung oder ästhetische Präferenz zu entscheiden, dass eine Änderung eine Verbesserung ist, behandelt die Methode jede vorgeschlagene Änderung am System als Hypothese, die gegen echte Nutzungsdaten validiert werden muss, bevor sie vollständig ausgerollt wird. In Legacy-Systemen, wo Jahre angehäufter Annahmen über Nutzerverhalten in die UI und den Workflow eingebacken, aber selten überprüft werden, bietet A/B-Testing einen Weg, diese Annahmen inkrementell und sicher infrage zu stellen, ohne sich auf ein riskantes Big-Bang-Redesign festzulegen. Es verwandelt außerdem vage Beschwerden über Nutzererfahrung in messbare Ergebnisse, was es möglich macht, Modernisierungsaufwand nach quantifizierter Auswirkung statt nach der Lautstärke der Beschwerde zu priorisieren. Da Legacy-Codebasen oft nicht mit Experimentieren im Sinn gebaut wurden, erfordert die Anwendung dieser Technik typischerweise zuerst das Hinzufügen von Feature-Flagging und Analytics-Instrumentierung als Voraussetzung, was selbst eine nützliche Zwangsfunktion zur Verbesserung der Beobachtbarkeit und Modularität des Systems ist.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie spezifische Nutzererfahrungshypothesen zum Testen, statt Änderungen allein auf Annahmen zu basieren
- Implementieren Sie Feature-Flagging-Infrastruktur, die kontrollierten Rollout von Änderungen an Untergruppen von Nutzern erlaubt
- Designen Sie Experimente mit klaren Erfolgsmetriken, die vor Testbeginn definiert werden
- Stellen Sie statistisch signifikante Stichprobengrößen und Testdauern sicher, um zuverlässige Ergebnisse zu produzieren
- Beginnen Sie mit risikoarmen UI-Änderungen im Legacy-System, bevor Sie fundamentalere Workflow-Modifikationen testen
- Instrumentieren Sie die Legacy-Anwendung mit Analytik zur Erfassung der für den Vergleich nötigen Nutzerverhaltensdaten
- Erstellen Sie einen Prozess zur Analyse von Ergebnissen und zum Treffen datengetriebener Entscheidungen darüber, welche Variante übernommen wird

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Ersetzt subjektive Design-Debatten durch datengetriebene Entscheidungen
- Verringert das Risiko, Änderungen auszurollen, die die Nutzererfahrung verschlechtern
- Liefert messbare Belege für Verbesserung zur Rechtfertigung von Modernisierungsinvestitionen
- Ermöglicht inkrementelle Verbesserung von Legacy-UIs ohne vollständiges Redesign

**Kosten und Risiken:**
- Legacy-Systemen fehlt oft die für ordentliches Experiment-Tracking nötige Instrumentierung
- Das Hinzufügen von Feature-Flagging zu Legacy-Code erhöht die Komplexität und erfordert sorgfältige Bereinigung
- Schlecht designte Experimente können irreführende Ergebnisse produzieren, die zu falschen Entscheidungen führen
- Das gleichzeitige Ausführen mehrerer Experimente kann Interaktionseffekte produzieren, die Ergebnisse verfälschen
- Manche Änderungen in Legacy-Systemen sind zu tief eingebettet, um leicht zwischen Varianten umgeschaltet zu werden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Versicherungsunternehmen wollte den Schadensmeldungs-Workflow in seinem Legacy-Webportal modernisieren, konnte sich aber intern nicht auf den besten Ansatz einigen. Das Team implementierte ein Feature-Flag-System und erstellte zwei alternative Workflows neben dem bestehenden, wobei 33 % der Nutzer zu jeder Version geleitet wurden. Nach vier Wochen zeigten die Daten, dass der vereinfachte dreistufige Workflow eine um 28 % höhere Abschlussrate und 40 % weniger Support-Tickets im Vergleich zum ursprünglichen siebenstufigen Prozess hatte. Dieser Beleg löste monatelange interne Debatten und lieferte konkrete Rechtfertigung für die Modernisierungsinvestition.
