---
title: Testkomplexität
description: Die Qualitätssicherung muss dieselbe Funktionalität an mehreren Stellen
  verifizieren, was den Testaufwand und das Risiko übersehener Bugs erhöht.
category:
- Code
- Testing
related_problems:
- slug: insufficient-testing
  similarity: 0.65
- slug: code-duplication
  similarity: 0.65
- slug: difficult-to-test-code
  similarity: 0.65
- slug: difficult-code-reuse
  similarity: 0.65
- slug: testing-environment-fragility
  similarity: 0.6
- slug: increased-manual-testing-effort
  similarity: 0.6
solutions:
- test-coverage-strategy
- platform-independent-test-frameworks
- characterization-tests
- dependency-breaking-techniques
- production-like-test-data
- isolated-test-environments
- containerization
- integration-tests
- contract-testing
- simulation-environments
- exploratory-testing
- explicit-extension-points
- variant-consolidation
- typed-schema-extraction
layout: problem
lang: de
en_slug: testing-complexity
---

## Description
Testkomplexität ist ein häufiges Problem in Softwaresystemen mit einem hohen Grad an Code-Duplizierung. Es tritt auf, wenn die Qualitätssicherung (QA) dieselbe Funktionalität an mehreren Stellen verifizieren muss. Dies erhöht den Testaufwand und das Risiko übersehener Bugs. Testkomplexität ist oft ein Zeichen eines schlecht designten Systems mit hohem Grad an Code-Duplizierung.

## Indicators ⟡
- Das QA-Team verbringt viel Zeit damit, dieselbe Funktionalität immer wieder zu testen.
- Das QA-Team kann mit dem Entwicklungstempo nicht Schritt halten.
- Das QA-Team übersieht viele Bugs.
- Das QA-Team ist mit der Qualität des Systems nicht zufrieden.

## Symptoms ▲

- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Der hohe Aufwand, der zum Testen duplizierter Funktionalität erforderlich ist, führt zu insgesamt unzureichender Testabdeckung.
- [Hohe Fehlerrate in Produktion](hohe-fehlerrate-in-produktion.md)
<br/>  Das erhöhte Risiko übersehener Bugs aufgrund duplizierter Testflächen bedeutet, dass mehr Probleme die Produktion erreichen.
- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Die Unfähigkeit des QA-Teams, mit der Entwicklung Schritt zu halten, aufgrund von Test-Overhead, verlangsamt die gesamte Lieferung.
- [Inkonsistente Qualität](inkonsistente-qualitaet.md)
<br/>  Manche Instanzen duplizierter Funktionalität werden gründlich getestet, während andere übersehen werden, was zu ungleichmäßiger Qualität führt.

## Causes ▼

- [Code-Duplizierung](code-duplizierung.md)
<br/>  Duplizierter Code bedeutet, dass dieselbe Funktionalität an mehreren Stellen verifiziert werden muss, was den Testaufwand direkt vervielfacht.
- [Komplexe und unklare Logik](komplexe-und-unklare-logik.md)
<br/>  Komplexe, schwer verständliche Logik erfordert aufwendigere Testszenarien und macht es schwieriger, angemessene Abdeckung zu erreichen.
- [Schwer testbarer Code](schwer-testbarer-code.md)
<br/>  Code, der aufgrund schlechten Designs inhärent schwer zu testen ist, trägt zur Gesamttestkomplexität bei.
- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Eng gekoppelte Komponenten können nicht isoliert getestet werden, was komplexe Integrationstestaufbauten erfordert.
- [Testschulden](testschulden.md)
<br/>  Wenn Testen zu komplex ist, nehmen Teams Abkürzungen und überspringen Tests, was über die Zeit Testschulden anhäuft.

## Detection Methods ○
- **Testfallanalyse:** Analyse Ihrer Testfälle zur Identifikation duplizierter Tests.
- **Codeabdeckungsanalyse:** Analyse Ihrer Codeabdeckung zur Identifikation von Systembereichen, die nicht getestet werden.
- **QA-Team-Feedback:** Anhörung von Feedback des QA-Teams zur Identifikation von Systembereichen, die schwer zu testen sind.
- **Bug-Triage:** Analyse Ihres Bug-Triage-Prozesses zur Identifikation von Bugs, die vom QA-Team übersehen werden.

## Examples
Eine E-Commerce-Website hat einen Checkout-Flow, der an zwei verschiedenen Stellen dupliziert ist. Das QA-Team muss den Checkout-Flow an beiden Stellen testen, um sicherzustellen, dass er korrekt funktioniert. Dies ist eine Verschwendung von Zeit und Aufwand und erhöht das Risiko, Bugs zu übersehen. Das Problem könnte gelöst werden, indem ein einziger, wiederverwendbarer Checkout-Flow erstellt wird, der an beiden Stellen genutzt wird.
