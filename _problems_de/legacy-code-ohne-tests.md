---
title: Legacy-Code ohne Tests
description: Bestehende Legacy-Systeme fehlt oft automatisierte Tests, was es herausfordernd
  macht, sie schrittweise hinzuzufügen und den Code sicher zu modifizieren.
category:
- Code
- Operations
- Testing
related_problems:
- slug: difficult-to-test-code
  similarity: 0.65
- slug: poor-test-coverage
  similarity: 0.65
- slug: outdated-tests
  similarity: 0.65
- slug: brittle-codebase
  similarity: 0.65
- slug: legacy-business-logic-extraction-difficulty
  similarity: 0.6
- slug: test-debt
  similarity: 0.6
solutions:
- test-coverage-strategy
- acceptance-tests
- automated-tests
- behavior-driven-development-bdd
- business-test-cases
- code-coverage-analysis
- functional-tests
- mutation-testing
- prepared-statements
- property-based-testing
- regression-tests
- security-tests
- smoke-testing
- specification-by-example
- test-driven-development-tdd
- dynamic-code-analysis
- fuzz-testing
- negative-testing
- static-code-analysis
- web-application-firewall
- characterization-tests
- dependency-breaking-techniques
- parallel-run
layout: problem
lang: de
en_slug: legacy-code-without-tests
---

## Description

Legacy-Code ohne Tests bezeichnet bestehende Produktionssysteme, die gebaut wurden, bevor umfassende Testpraktiken übernommen wurden, oder bei denen Testen während der Entwicklung depriorisiert wurde. Dieser Code ist besonders herausfordernd, weil er oft eng gekoppelt ist, versteckte Abhängigkeiten hat und die Designeigenschaften vermissen lässt, die Testen unkompliziert machen. Das Hinzufügen von Tests zu Legacy-Code erfordert erheblichen Aufwand und Expertise, was eine Barriere schafft, die Teams daran hindert, die Codequalität zu verbessern und technische Schulden zu verringern.

## Indicators ⟡
- Große Teile kritischen Produktionscodes haben keine zugehörigen automatisierten Tests
- Der Code wurde geschrieben, bevor das Team testgetriebene Entwicklung oder Test-Best-Practices übernahm
- Versuche, Tests zu bestehendem Code hinzuzufügen, erfordern umfangreiches Refactoring
- Entwickler vermeiden es, bestimmte Bereiche aufgrund fehlender Testabdeckung zu modifizieren
- Produktionssysteme laufen seit Jahren ohne umfassende Testsuiten

## Symptoms ▲

- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Wenn jede Änderung an ungetestetem Code übermäßige Vorsicht und manuelle Verifikation erfordert, kann das resultierende chronische Risiko und der Aufwand Entwickler über die Zeit zermürben.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Ohne Tests, die verifizieren, dass Änderungen sicher sind, werden Entwickler zurückhaltend, Code zu ändern, aus Angst, Regressionen einzuführen.
- [Wartungslähmung](wartungslaehmung.md)
<br/>  Teams vermeiden notwendige Verbesserungen, weil sie ohne Testabdeckung nicht verifizieren können, dass Änderungen bestehende Funktionalität nicht brechen.
- [Große Schätzungen für kleine Änderungen](grosse-schaetzungen-fuer-kleine-aenderungen.md)
<br/>  Ohne automatisierte Tests müssen Entwickler in ihren Schätzungen für jede Codeänderung umfangreiche manuelle Verifikation berücksichtigen.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Ohne automatisierte Tests, die Regressionen erfassen, brechen Änderungen häufig zuvor funktionierende Funktionalität.
- [Hohe Fehlerrate in Produktion](hohe-fehlerrate-in-produktion.md)
<br/>  Fehlende Testabdeckung bedeutet, dass Defekte während der Entwicklung unentdeckt bleiben und erst in Produktion auftauchen.
- [Erhöhter manueller Testaufwand](erhoehter-manueller-testaufwand.md)
<br/>  Ohne automatisierte Tests muss alle Verifikation manuell erfolgen, was den manuellen Testaufwand direkt erhöht.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Ohne ein Test-Sicherheitsnetz, das verifiziert, dass ein ordentlicher Fix sicher ist, greifen Entwickler auf Workarounds zurück, die es vermeiden, den ungetesteten Code direkt zu berühren.

## Causes ▼

- [Schwer testbarer Code](schwer-testbarer-code.md)
<br/>  Eng gekoppelter Legacy-Code mit versteckten Abhängigkeiten macht es strukturell schwierig, Tests ohne größeres Refactoring hinzuzufügen.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Das Management priorisierte historisch Feature-Lieferung über das Schreiben von Tests, was zu großen ungetesteten Codebasen führte.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Übermäßig gekoppelte Komponenten können nicht isoliert getestet werden, was es unpraktikabel macht, Legacy-Code Tests hinzuzufügen.
- [Rapid Prototyping wird zu Produktion](rapid-prototyping-wird-zu-produktion.md)
<br/>  Prototyp-Code, der nie dauerhaft sein sollte, gelangte ohne Tests in die Produktion und wurde nie nachträglich getestet.

## Detection Methods ○
- **Code-Abdeckungs-Analyse:** Messung der Testabdeckung für unterschiedliche Teile des Systems zur Identifikation ungetesteter Legacy-Bereiche
- **Code-Alters-Analyse:** Identifikation älterer Codeabschnitte, die geschrieben wurden, bevor Testpraktiken etabliert wurden
- **Abhängigkeitsanalyse:** Kartierung von Code-Abhängigkeiten zur Identifikation von Bereichen, die schwer zu testen wären
- **Änderungshäufigkeit vs. Testabdeckung:** Korrelation, wie oft Code modifiziert wird, mit seiner Testabdeckung
- **Entwicklerbefragungen:** Befragung von Teammitgliedern, welche Codebereiche sie am meisten fürchten zu modifizieren, aufgrund fehlender Tests

## Examples

Ein 10 Jahre altes Bestandsverwaltungssystem verarbeitet täglich Millionen von Dollar an Transaktionen, hat aber null automatisierte Tests. Die Kern-Bestandsverfolgungsalgorithmen, Preisberechnungen und Bestellabwicklungslogik sind alles ungetesteter Legacy-Code, geschrieben von Entwicklern, die das Unternehmen seitdem verlassen haben. Als das Geschäft Unterstützung für neue Produktkategorien hinzufügen muss, entdecken Entwickler, dass der bestehende Code globale Variablen nutzt, Datenbanken direkt innerhalb von Geschäftslogik-Methoden abfragt und zirkuläre Abhängigkeiten zwischen Klassen hat. Das Hinzufügen von Tests würde umfangreiches Refactoring erfordern, das bestehende Funktionalität brechen könnte, aber die Modifikation des Codes ohne Tests ist extrem riskant, angesichts der finanziellen Auswirkung von Fehlern. Das Team ist in einer Situation gefangen, in der es den Code nicht sicher verbessern kann, ohne Tests, aber keine Tests hinzufügen kann, ohne potenziell das bestehende System zu brechen. Ein weiteres Beispiel betrifft ein Kundenbeziehungsmanagementsystem, bei dem die Lead-Scoring-Algorithmen in einer 3.000-Zeilen-Klasse implementiert sind, die direkt auf externe APIs zugreift, Datenbankeinträge modifiziert und E-Mails sendet. Die Komplexität und enge Kopplung machen es praktisch unmöglich, Unit-Tests zu erstellen, während das Fehlen von Tests es gefährlich macht, den Code in besser testbare Komponenten zu refaktorieren.
