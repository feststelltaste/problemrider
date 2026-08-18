---
title: Veraltete Tests
description: Tests werden nicht aktualisiert, wenn sich der Code ändert, was zu falsch-positiven
  oder falsch-negativen Ergebnissen und verringertem Vertrauen führt.
category:
- Code
- Testing
related_problems:
- slug: flaky-tests
  similarity: 0.65
- slug: legacy-code-without-tests
  similarity: 0.65
- slug: testing-environment-fragility
  similarity: 0.6
- slug: test-debt
  similarity: 0.6
- slug: information-decay
  similarity: 0.6
- slug: poor-test-coverage
  similarity: 0.6
solutions:
- test-coverage-strategy
- mutation-testing
- characterization-tests
- code-quality-gates
- ci-cd-pipeline
- code-reviews
- definition-of-done
- regression-testing
layout: problem
lang: de
en_slug: outdated-tests
---

## Description

Veraltete Tests treten auf, wenn Test-Code nicht zusammen mit Produktionscode-Änderungen gepflegt wird, was zu Tests führt, die das beabsichtigte Verhalten nicht mehr akkurat verifizieren. Diese Tests können bestehen, wenn sie fehlschlagen sollten (falsch-positiv), oder fehlschlagen, wenn der Code tatsächlich korrekt ist (falsch-negativ). Veraltete Tests sind schlimmer als keine Tests, weil sie falsches Vertrauen in die Codequalität vermitteln, während sie Wartungsaufwand verbrauchen und die Entwicklung mit falschen Fehlschlägen verlangsamen.

## Indicators ⟡
- Tests bestehen, aber die Funktionalität, die sie verifizieren sollen, ist defekt
- Tests scheitern konsequent aus Gründen, die nichts mit tatsächlichen Codefehlern zu tun haben
- Testfehlschläge werden häufig ignoriert oder umgangen, weil bekannt ist, dass sie unzuverlässig sind
- Tests verifizieren veraltete Geschäftsregeln oder abgekündigte Funktionalität
- Erheblicher Aufwand wird für die Pflege und das Debugging von Tests aufgewendet statt für ihre Verbesserung

## Symptoms ▲

- [Flaky Tests](flaky-tests.md)
<br/>  Veraltete Tests, die sich auf geänderte Daten oder abgekündigte Funktionalität beziehen, schlagen intermittierend fehl, was sich als flakes Testverhalten äußert.
- [Qualitäts-blinde Flecken](qualitaets-blinde-flecken.md)
<br/>  Tests, die veraltetes Verhalten verifizieren, vermitteln falsches Vertrauen, während tatsächliche aktuelle Funktionalität unverifiziert bleibt, was Qualitäts-blinde Flecken schafft.
- [Hohe Fehlerrate in Produktion](hohe-fehlerrate-in-produktion.md)
<br/>  Falsches Vertrauen durch bestehende veraltete Tests bedeutet, dass echte Fehler unentdeckt in die Produktion gelangen.
- [Wartungs-Overhead](wartungs-overhead.md)
<br/>  Erhebliche Zeit wird mit dem Debuggen und Aktualisieren von Tests verbracht, die aus Gründen fehlschlagen, die nichts mit tatsächlichen Codefehlern zu tun haben.
- [Vermeidung des Review-Prozesses](vermeidung-des-review-prozesses.md)
<br/>  Wenn Tests unzuverlässig sind, beginnen Teams, Testergebnisse zu ignorieren oder zu umgehen, was den Qualitätssicherungsprozess untergräbt.

## Causes ▼

- [Feature-Creep ohne Refactoring](feature-creep-ohne-refactoring.md)
<br/>  Das kontinuierliche Hinzufügen von Features ohne Aktualisierung der entsprechenden Tests verursacht, dass Tests aus dem Gleichschritt mit aktuellem Verhalten geraten.
- [Zeitdruck](zeitdruck.md)
<br/>  Unter Termindruck aktualisieren Entwickler Produktionscode, überspringen aber die Aktualisierung der entsprechenden Tests, um Zeit zu sparen.
- [Fehlende Eigenverantwortung und Rechenschaftspflicht](fehlende-eigenverantwortung-und-rechenschaftspflicht.md)
<br/>  Ohne klare Verantwortung für die Testwartung verschlechtern sich Tests, da niemand die Verantwortung übernimmt, sie aktuell zu halten.
- [Unzureichendes Testdatenmanagement](unzureichendes-testdatenmanagement.md)
<br/>  Unrealistische oder veraltete Testdaten verursachen, dass Tests im Laufe der Zeit fehlangepasst zum tatsächlichen Systemverhalten werden.
- [Unzureichendes Code-Review](unzureichendes-code-review.md)
<br/>  Ohne gründliche Code-Reviews, die prüfen, ob Tests zusammen mit Produktionscode-Änderungen aktualisiert werden, häufen sich veraltete Tests an.

## Detection Methods ○
- **Test-Zuverlässigkeitsmetriken:** Nachverfolgung der Häufigkeit von Testfehlschlägen und ihrer Korrelation mit tatsächlichen Fehlern
- **Test-Wartungszeit:** Überwachung, wie viel Zeit für die Behebung von Tests versus die Verbesserung von Funktionalität aufgewendet wird
- **Falsch-Positiv-/Negativ-Analyse:** Identifikation von Tests, die falsche Ergebnisse über die Codequalität liefern
- **Testalters-Analyse:** Untersuchung, wie lange Tests relativ zu Codeänderungen ohne Aktualisierung geblieben sind
- **Entwickler-Feedback:** Befragung von Teammitgliedern zu ihrem Vertrauen in die Testzuverlässigkeit

## Examples

Ein Nutzerauthentifizierungssystem hat umfassende Tests, die Passwortkomplexitätsanforderungen verifizieren, einschließlich Regeln zu Sonderzeichen, Länge und Zeichenmischung. Die Geschäftsanforderungen änderten sich jedoch vor sechs Monaten, um einfachere Passwörter zur Verbesserung der Nutzererfahrung zu erlauben, und der Produktionscode wurde entsprechend aktualisiert. Die Tests verifizieren weiterhin die alten, strengeren Anforderungen und bestehen weiterhin, obwohl das System jetzt Passwörter akzeptiert, die die von den Tests geprüften Regeln verletzen. Entwickler und Stakeholder glauben, dass die Passwortvalidierung gründlich getestet ist, aber die tatsächliche Validierungslogik hat keine sinnvolle Testabdeckung. Ein weiteres Beispiel betrifft ein E-Commerce-Preissystem, bei dem Tests Rabattberechnungen mit hartcodierten Produkt-IDs und Preisen aus der ursprünglichen Test-Datenbank verifizieren. Über die Zeit wurde die Test-Datenbank modifiziert, Produkte wurden eingestellt, und Preisstrukturen haben sich geändert. Die Tests schlagen jetzt intermittierend fehl, abhängig vom Datenbankzustand, und Entwickler aktualisieren regelmäßig Testdaten, um Tests bestehen zu lassen, ohne zu verifizieren, dass die Rabattlogik tatsächlich korrekt funktioniert. Die Tests sind zu Wartungsoverhead geworden, der keine Zusicherung über die Genauigkeit der Preisberechnung bietet.
