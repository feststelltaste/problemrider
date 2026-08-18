---
title: Unzureichendes Testen
description: Der Testprozess ist nicht umfassend genug, was zu einer hohen Fehlerrate
  in Produktion führt.
category:
- Code
- Process
related_problems:
- slug: high-defect-rate-in-production
  similarity: 0.8
- slug: poor-test-coverage
  similarity: 0.7
- slug: inadequate-test-data-management
  similarity: 0.7
- slug: inadequate-code-reviews
  similarity: 0.7
- slug: inadequate-test-infrastructure
  similarity: 0.65
- slug: insufficient-design-skills
  similarity: 0.65
solutions:
- definition-of-done
- test-coverage-strategy
- acceptance-tests
- automated-tests
- behavior-driven-development-bdd
- business-test-cases
- code-coverage-analysis
- compatibility-as-error
- compatibility-testing
- compatibility-testing-by-users
- cross-version-testing
- functional-tests
- mutation-testing
- platform-independent-test-frameworks
- prepared-statements
- property-based-testing
- red-teaming
- regression-tests
- requirements-traceability-matrix
- secure-software-development
- security-tests
- security-tests-by-external-parties
- smoke-testing
- specification-by-example
- user-acceptance-tests
- dynamic-code-analysis
- negative-testing
- penetration-tests
- vulnerability-scans
- parallel-run
- production-like-test-data
- production-readiness-criteria
- exploratory-testing
layout: problem
lang: de
en_slug: insufficient-testing
---

## Description
Unzureichendes Testen ist eine wesentliche Ursache schlechter Softwarequalität. Wenn ein Produkt nicht gründlich getestet wird, hat es wahrscheinlich eine hohe Anzahl an Fehlern, was zu einem schlechten Nutzererlebnis, einem Vertrauensverlust und einer erheblichen Menge an Nacharbeit führen kann. Eine umfassende Teststrategie sollte eine Mischung aus automatisiertem und manuellem Testen umfassen und von Anfang an in den Entwicklungsprozess integriert sein. Investitionen in Tests sind Investitionen in die Qualität und Stabilität des Produkts.

## Indicators ⟡
- Das Team hat keine automatisierten Tests.
- Das Team hat eine niedrige Testabdeckung.
- Das Team findet ständig Fehler in Produktion.
- Das Team hat Angst, Änderungen an der Codebasis vorzunehmen, aus Furcht, etwas zu brechen.

## Symptoms ▲

- [Hohe Fehlerrate in Produktion](hohe-fehlerrate-in-produktion.md)
<br/>  Ohne umfassendes Testen entkommen mehr Defekte in die Produktion, wo sie Nutzer betreffen.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Unzureichende Testabdeckung bedeutet, dass Änderungen häufig bestehende Funktionalität brechen, ohne dass dies vor dem Release erkannt wird.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Entwickler bekommen Angst, Code zu ändern, wenn es keine Tests gibt, die verifizieren, dass ihre Änderungen nichts brechen.
- [Häufige Hotfixes und Rollbacks](haeufige-hotfixes-und-rollbacks.md)
<br/>  Produktionsdefekte durch unzureichendes Testen erfordern Notfall-Fixes und Rollbacks.
- [Ständiges Feuerlöschen](staendiges-feuerloeschen.md)
<br/>  Teams verbringen erhebliche Zeit mit Produktionsproblemen, die durch Testen hätten erfasst werden sollen.
- [Kundenunzufriedenheit](kundenunzufriedenheit.md)
<br/>  Häufige Produktionsfehler durch unzureichendes Testen beeinträchtigen direkt Nutzererlebnis und Zufriedenheit.
- [Erhöhte Fehleranzahl](erhoehte-fehleranzahl.md)
<br/>  Unzureichendes Testen führt direkt dazu, dass mehr Fehler die Produktion erreichen, was ein natürlicheres Symptom ist als manche aktuellen.
- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Ohne Tests als Sicherheitsnetz steigt das Risiko, mit jeder Änderung Fehler einzuführen, erheblich.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Ohne ein Sicherheitsnetz aus Tests wird Refactoring riskant, sodass schuldenbelasteter Code unangetastet bleibt und sich weiter anhäuft.
- [Schwachstellen zur Umgehung der Authentifizierung](schwachstellen-zur-umgehung-der-authentifizierung.md)
<br/>  Ohne dedizierte Sicherheitstests wie Penetrationstests bleiben Authentifizierungslogikfehler, die eine Umgehung erlauben, vor dem Release unentdeckt.

## Causes ▼

- [Termindruck](termindruck.md)
<br/>  Unter Zeitdruck ist Testen oft die erste Aktivität, die gekürzt oder reduziert wird.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Die Priorisierung sofortiger Feature-Lieferung über langfristige Qualität führt zu Unterinvestition in Tests.
- [Unzureichende Design-Fähigkeiten](unzureichende-design-faehigkeiten.md)
<br/>  Schlecht gestalteter Code ist schwer zu testen, was umfassende Testbemühungen entmutigt.
- [Unzureichende Testinfrastruktur](unzureichende-testinfrastruktur.md)
<br/>  Fehlende ordentliche Testumgebungen und -Tooling machen umfassendes Testen unpraktikabel.

## Detection Methods ○

- **Fehlerverfolgungsmetriken:** Beobachtung der Anzahl der in Produktion vs. Vor-Produktions-Umgebungen gefundenen Fehler.
- **Code-Abdeckungs-Werkzeuge:** Nutzung von Werkzeugen zur Messung des Prozentsatzes von Code, der von Tests ausgeführt wird.
- **Testautomatisierungsberichte:** Analyse von Berichten aus automatisierten Testläufen zur Identifikation von Lücken oder Fehlschlägen.
- **Retrospektiven:** Diskussion der Testwirksamkeit und Identifikation von Verbesserungsbereichen in Team-Retrospektiven.
- **Manuelle Testfall-Überprüfung:** Überprüfung manueller Testfälle zur Identifikation von Bereichen, in denen Automatisierung eingeführt oder die Abdeckung verbessert werden könnte.

## Examples
Ein neues Feature wird veröffentlicht, und sofort melden Nutzer, dass ein kritischer Workflow defekt ist. Die Untersuchung zeigt, dass zwar einzelne Komponenten getestet wurden, der End-to-End-Fluss mit mehreren Diensten aber nie in einer integrierten Umgebung getestet wurde. In einem anderen Fall nimmt ein Entwickler eine kleine Änderung an einer Utility-Funktion vor. Ohne Unit-Tests für diese Funktion merkt er nicht, dass sie einen Nebeneffekt hat, der einen anderen, scheinbar unzusammenhängenden Teil der Anwendung bricht, was zu einem Regressionsfehler in Produktion führt. Dieses Problem entspringt oft einer Kultur, die Geschwindigkeit über Qualität priorisiert, oder mangelndem Verständnis der langfristigen Vorteile einer robusten Teststrategie. Es kann zu erheblichen technischen Schulden und einem ständigen Zustand des Feuerlöschens führen.
