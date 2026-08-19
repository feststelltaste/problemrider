---
title: Code-Reviews
description: Durchführung regelmäßiger Reviews des Quellcodes durch Teammitglieder.
category:
- Code
- Process
problems:
- inadequate-code-reviews
- insufficient-code-review
- superficial-code-reviews
- lower-code-quality
- inconsistent-coding-standards
- knowledge-silos
- high-bug-introduction-rate
- difficult-code-comprehension
- clever-code
- improper-event-listener-management
- inconsistent-naming-conventions
- increased-technical-shortcuts
- mixed-coding-styles
- null-pointer-dereferences
- outdated-tests
- procedural-background
- queries-that-prevent-index-usage
- stack-overflow-errors
- unreleased-resources
- algorithmic-complexity-problems
- circular-references
- copy-paste-programming
- increased-bug-count
- inefficient-code
- log-spam
- n-plus-one-query-problem
- poor-naming-conventions
- database-connection-leaks
- defensive-coding-practices
- endianness-conversion-overhead
- excessive-logging
- incorrect-index-type
- increased-risk-of-bugs
- log-injection-vulnerabilities
- partial-bug-fixes
- undefined-code-style-guidelines
- customization-outside-version-control
layout: solution
lang: de
en_slug: code-reviews
related_solutions:
- slug: code-review-process-reform
  similarity: 0.85
- slug: static-analysis-and-linting
  similarity: 0.75
- slug: code-quality-gates
  similarity: 0.75
- slug: code-metrics
  similarity: 0.75
- slug: architecture-reviews
  similarity: 0.75
- slug: code-conventions
  similarity: 0.75
---

## Description

Code-Review ist die Praxis, einen oder mehrere andere Entwickler eine vorgeschlagene Codeänderung untersuchen zu lassen, bevor sie gemergt wird, wobei sie auf Korrektheit, Einhaltung von Konventionen und angemessene Testabdeckung geprüft wird, und das Review selbst als strukturierter Kontrollpunkt statt informeller Höflichkeit genutzt wird. In Legacy-Systemen trägt die Praxis zusätzliches Gewicht über das Abfangen von Bugs hinaus: Weil kritische Logik oft nur von einer kleinen Anzahl langjähriger Entwickler verstanden wird, ist die Rotation von Reviewern über das Team ein direkter Mechanismus, um dieses Wissen zu verbreiten und das Bus-Faktor-Risiko zu verringern, das sich um eine Handvoll Personen konzentriert. Reviews sind außerdem der Ort, wo undokumentierte Geschäftsregeln, tief in Legacy-Code eingebettet, abgefangen werden, bevor sie versehentlich gebrochen werden, da ein mit den Eigenheiten eines Moduls vertrauter Reviewer eine Änderung markieren kann, die isoliert korrekt aussieht, aber eine Einschränkung verletzt, die nur im institutionellen Gedächtnis existiert. Änderungen klein genug zu halten, um gründlich überprüft zu werden, ist in Legacy-Kontexten wichtiger als in Greenfield-Kontexten, da große, ausufernde Legacy-Refaktorierungsbemühungen sonst nahezu unmöglich sinnvoll in einem Durchgang zu überprüfen sind. Der Wert der Praxis hängt vollständig davon ab, dass das Review tatsächlich substanziell ist: Eine Review-Kultur, die zu blindem Abnicken degeneriert, bietet den Anschein eines Sicherheitsnetzes ohne jeglichen seiner tatsächlichen Schutz, während übermäßiges Nitpicking oder langsame Bearbeitungszeit den Prozess zu einem Engpass machen kann, den Teams lernen zu umgehen statt sich mit ihm zu befassen.

## How to Apply ◆

- Etablieren Sie Code-Review als verpflichtenden Schritt vor dem Mergen jeder Änderung in den Main-Branch der Legacy-Codebasis.
- Definieren Sie Review-Checklisten, die legacy-spezifische Belange beinhalten: ordentliche Handhabung bestehender Konventionen, Bewahrung undokumentierter Geschäftslogik, und angemessene Testabdeckung für geänderten Code.
- Halten Sie Pull Requests klein und fokussiert, um gründliche Reviews zu ermöglichen; brechen Sie große Legacy-Refaktorierungsbemühungen in überprüfbare Inkremente auf.
- Rotieren Sie Reviewer, um Wissen über die Legacy-Codebasis über das Team zu verbreiten und Wissenssilos zu verhindern.
- Nutzen Sie Code-Review als Lehrgelegenheit für Entwickler, die mit den Mustern und Einschränkungen des Legacy-Systems nicht vertraut sind.
- Setzen Sie Reaktionszeiterwartungen (z. B. Reviews abgeschlossen innerhalb eines Arbeitstages), um Review-Engpässe zu verhindern.

## Tradeoffs ⇄

**Vorteile:**
- Fängt Bugs und Logikfehler ab, bevor sie Produktion erreichen, besonders wichtig in Legacy-Systemen mit begrenzter Testabdeckung.
- Verteilt Wissen über die Legacy-Codebasis über Teammitglieder, was das Bus-Faktor-Risiko verringert.
- Setzt Konsistenz in Coding-Standards und architektonischen Mustern innerhalb des Legacy-Systems durch.
- Dient als Lernmechanismus für Entwickler, die neu zur Legacy-Codebasis sind.

**Kosten:**
- Fügt dem Entwicklungsworkflow Zeit hinzu, was unter Terminsdruck herausfordernd sein kann.
- Ineffektive Reviews (blindes Abnicken) bieten falsches Vertrauen, ohne Probleme abzufangen.
- Können Engpässe schaffen, wenn Reviewer-Verfügbarkeit begrenzt ist.
- Zwischenmenschliche Dynamiken (Nitpicking, widersprüchliche Meinungen) können Reviews kontraproduktiv machen.

## How It Could Be

Ein Legacy-Finanzsystem hat kritische Berechnungslogik, die nur zwei Senior-Entwickler vollständig verstehen. Das Team führt verpflichtende Code-Reviews mit einer Rotationsrichtlinie ein, die sicherstellt, dass jeder Entwickler über die Zeit Code über verschiedene Module hinweg überprüft. Innerhalb von sechs Monaten gewinnen drei zusätzliche Entwickler ausreichendes Verständnis der Berechnungs-Engine, um Änderungen zuversichtlich vorzunehmen. Reviews fangen außerdem mehrere Fälle ab, in denen neue Entwickler versehentlich undokumentierte Geschäftsregeln brechen, die im Legacy-Code eingebettet sind. Der Review-Prozess wird zum primären Mechanismus zur Übertragung institutionellen Wissens über die Eigenheiten und Konventionen des Legacy-Systems an neuere Teammitglieder.
