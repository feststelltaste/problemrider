---
title: Pattern Language
description: Anwendung bewährter Lösungsmuster für wiederkehrende
  Designprobleme.
category:
- Architecture
- Code
problems:
- inconsistent-codebase
- suboptimal-solutions
- knowledge-gaps
- difficult-code-comprehension
- cargo-culting
- insufficient-design-skills
- misunderstanding-of-oop
layout: solution
lang: de
en_slug: pattern-language
related_solutions:
- slug: domain-patterns
  similarity: 0.85
- slug: facades
  similarity: 0.75
- slug: living-documentation
  similarity: 0.75
- slug: style-guide
  similarity: 0.75
- slug: data-strategy
  similarity: 0.75
- slug: adapter
  similarity: 0.75
---

## Description

Eine Pattern Language ist ein gemeinsames Vokabular bewährter, benannter Lösungen für wiederkehrende Designprobleme, aufgebaut, damit ein Team von „dem Adapter hier" oder „einer State-Maschine dort" sprechen kann, statt ein Design jedes Mal von Grund auf neu zu erklären. Legacy-Codebasen, die ohne dieses gemeinsame Vokabular gewachsen sind, neigen dazu, mehrere verschiedene, undokumentierte Lösungen für dasselbe zugrunde liegende Problem anzuhäufen, jede geschrieben von einem anderen Entwickler, der sich der anderen nicht bewusst war — oder sie nicht finden konnte. Bewusst zu katalogisieren, welche Muster für die Domäne und Technologie des Systems gelten, und sie konsistent in Designdiskussionen und Code-Reviews zu verwenden, ersetzt diese Inkonsistenz durch eine Codebasis, in der Entwickler vertraute strukturelle Idiome erkennen und sich schneller durch unbekannte Module navigieren können. Das Risiko besteht darin, Muster aus Gewohnheit statt aus Passung anzuwenden: ein Muster, das dort auferlegt wird, wo es nicht hingehört, fügt Zeremonie hinzu, ohne etwas zu lösen, sodass das Vokabular nur wertvoll ist, wenn es mit Urteilsvermögen darüber gepaart wird, wann ein Muster tatsächlich zum vorliegenden Problem passt.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Bauen Sie ein gemeinsames Vokabular von Designmustern auf, die für die Domäne und den Technologie-Stack des Legacy-Systems relevant sind
- Identifizieren Sie wiederkehrende Probleme in der Codebasis und ordnen Sie sie etablierten Mustern zu, statt Ad-hoc-Lösungen zu erfinden
- Dokumentieren Sie, welche Muster wo verwendet werden, damit künftige Entwickler die Absicht hinter dem Design verstehen
- Führen Sie musterorientierte Code-Reviews durch, bei denen Reviewer prüfen, ob bekannte Muster angemessen angewendet wurden
- Verwenden Sie Muster als Kommunikationswerkzeug während Architekturdiskussionen, um das Team auf die Designabsicht auszurichten
- Vermeiden Sie es, Muster zu erzwingen, wo sie nicht passen; ein in einem falschen Kontext angewendetes Muster richtet mehr Schaden als Nutzen an
- Organisieren Sie Study Groups oder Lunch-and-Learns, um die Vertrautheit des Teams mit relevanten Mustern aufzubauen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Bietet eine gemeinsame Sprache, die Missverständnisse in Designdiskussionen reduziert
- Erfasst bewährte Lösungen, sodass Teams das Rad nicht für häufige Probleme neu erfinden
- Macht Code vorhersagbarer und navigierbarer, wenn Entwickler vertraute Muster erkennen
- Beschleunigt das Onboarding, indem neuen Entwicklern ein Framework zum Verständnis der Codebasis gegeben wird

**Kosten und Risiken:**
- Übernutzung führt zu Muster-Sucht, bei der einfache Probleme in unnötige Komplexität eingewickelt werden
- Ohne Verständnis ihres Kontexts angewendete Muster können Code verschlechtern
- Kann ein falsches Gefühl von Vollständigkeit erzeugen: nicht jedes Designproblem hat ein passendes Muster
- Erfordert Investition in Teamschulung, um wirksam zu sein

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Logistikunternehmen hatte ein Legacy-System, in dem verschiedene Entwickler unabhängig voneinander mehrere Ansätze für dieselben Probleme entwickelt hatten: drei verschiedene Wege zur Handhabung von Zustandsübergängen, vier Varianten Observer-ähnlicher Benachrichtigungsmechanismen und zwei konkurrierende Strategien für Objektkonstruktion. Das Team katalogisierte diese Varianten und einigte sich auf ein Standardmuster für jedes Anliegen. Sie übernahmen das State-Muster für Bestellstatusübergänge und eine konsistente Observer-Implementierung für Benachrichtigungen. In den folgenden Monaten, während Code modifiziert wurde, ersetzten Entwickler Ad-hoc-Implementierungen durch die vereinbarten Muster. Die Codebasis wurde konsistenter, und Entwickler konnten unbekannte Module schneller verstehen, weil sie die gleichen strukturellen Idiome durchgängig erkannten.
