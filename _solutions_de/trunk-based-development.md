---
title: Trunk-Based Development
description: Kontinuierliche Integration kurzlebiger Branches in den
  Main-Branch für schnelle, sichere Änderungen.
category:
- Process
problems:
- long-lived-feature-branches
- merge-conflicts
- integration-difficulties
- large-pull-requests
- slow-development-velocity
- deployment-coupling
- large-risky-releases
- extended-cycle-times
- extended-review-cycles
- increased-time-to-market
- review-bottlenecks
- reduced-code-submission-frequency
layout: solution
lang: de
en_slug: trunk-based-development
related_solutions:
- slug: continuous-integration
  similarity: 0.8
- slug: continuous-integration-and-delivery
  similarity: 0.8
- slug: rollback-mechanisms
  similarity: 0.75
- slug: continuous-delivery
  similarity: 0.75
- slug: canary-releases
  similarity: 0.7
- slug: chaos-engineering
  similarity: 0.7
---

## Description

Trunk-Based Development ist ein Versionskontroll-Workflow, bei dem Entwickler kleine, kurzlebige Änderungen mindestens täglich in den Main-Branch integrieren, statt isoliert an langlebigen Feature-Branches zu arbeiten, die für Wochen oder Monate von Main abweichen, bevor sie gemergt werden. Unfertige Arbeit wird durch Verstecken hinter Feature-Flags sicher zum kontinuierlichen Mergen gemacht, und eine schnelle, umfassende CI-Pipeline validiert jede Integration, sodass Main im Wesentlichen jederzeit in einem release-fähigen Zustand bleibt. Die Praxis ist eine direkte Gegenmaßnahme zu einem Muster, das in Legacy-Codebasen mit verwurzelten Branching-Gewohnheiten häufig ist: Branches, die lange genug leben, um substanziell von Main abzudriften, was Merge-Konflikte produziert, die Tage an Nacharbeit verbrauchen, und Integrationsfehler, die erst zutage treten, wenn es teuer ist zu entwirren, welche der vielen angesammelten Änderungen sie verursacht hat. Indem das Intervall zwischen dem Schreiben von Code und seiner Integration zusammengezogen wird, verwandelt Trunk-Based Development Integration von einem seltenen, hochriskanten Ereignis in ein routinemäßiges, risikoarmes — genau die Verschiebung, die Legacy-Teams brauchen, wenn große, riskante, seltene Releases historisch jede Änderung gefährlich erscheinen ließen. Die Übernahme in einem Legacy-Kontext erfordert typischerweise Vorabinvestition in CI-Geschwindigkeit und Testzuverlässigkeit, da das gesamte Modell von schnellem, vertrauenswürdigem Feedback zu jedem kleinen Merge abhängt — ohne diese Investition bringt häufige Integration dieselben Probleme nur schneller zutage, statt sie tatsächlich zu lösen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Halten Sie Branches kurzlebig: Mergen Sie mindestens täglich in den Main-Branch, idealerweise mehrmals täglich
- Nutzen Sie Feature-Flags, um Deployment vom Feature-Release zu entkoppeln, sodass unfertige Arbeit sicher gemergt werden kann
- Investieren Sie in eine robuste CI-Pipeline, die schnelle, umfassende Tests bei jedem Merge zu Main ausführt
- Teilen Sie große Änderungen in kleine, inkrementelle Commits auf, die jeweils unabhängig gemergt werden können
- Beseitigen Sie langlebige Feature-Branches und ersetzen Sie sie durch Techniken wie Branch by Abstraction
- Stellen Sie sicher, dass der Main-Branch durch automatisierte Qualitätsgates immer in einem deploybaren Zustand ist
- Adressieren Sie flaky Tests aggressiv, da sie das Vertrauen in kontinuierliche Integration untergraben

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Reduziert Merge-Konflikte dramatisch durch häufige Integration von Änderungen
- Bietet schnelles Feedback zu Integrationsproblemen, statt sie zur Merge-Zeit zu entdecken
- Ermöglicht kontinuierliche Auslieferung, indem der Main-Branch immer release-fähig gehalten wird
- Reduziert die Code-Review-Last, weil Änderungen klein und fokussiert sind

**Kosten und Risiken:**
- Erfordert ausgereifte CI-Infrastruktur und schnelle Testsuiten, um häufige Merges zu unterstützen
- Feature-Flags fügen Komplexität hinzu und müssen bereinigt werden, um Flag-Schulden zu vermeiden
- Teams müssen Disziplin entwickeln, kleine, vollständige Inkremente zu committen, statt große Pakete
- Teilweise fertige Features auf Main erfordern sorgfältiges Management, um zu vermeiden, dass sie Nutzern ausgesetzt werden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Enterprise-Anwendungsteam hatte die Praxis, Feature-Branches über Wochen oder Monate zu pflegen. Der Merge-Tag wurde gefürchtet und verbrauchte oft einen ganzen Sprint. Während Merges entdeckte Integrationsfehler erforderten häufig Nacharbeit. Das Team wechselte zu Trunk-Based Development, beginnend damit, ihren aktuellen langlebigen Branch mithilfe von Feature-Flags in täglich mergebare Inkremente aufzuteilen. Sie investierten in die Beschleunigung der Testsuite von 45 Minuten auf 8 Minuten. Innerhalb von drei Monaten mergte das Team mehrmals täglich zu Main. Merge-Konflikte wurden selten, Integrationsfehler wurden sofort erfasst, und die Geschwindigkeit des Teams stieg messbar, weil sie weit weniger Zeit für merge-bezogene Nacharbeit aufwendeten.
