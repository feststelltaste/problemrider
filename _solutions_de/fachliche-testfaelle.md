---
title: Fachliche Testfälle
description: Erstellung von Testfällen aus fachlicher Perspektive und Überprüfung
  durch Nutzer.
category:
- Testing
- Requirements
problems:
- insufficient-testing
- poor-test-coverage
- regression-bugs
- stakeholder-developer-communication-gap
- requirements-ambiguity
- legacy-code-without-tests
layout: solution
lang: de
en_slug: business-test-cases
related_solutions:
- slug: acceptance-tests
  similarity: 0.7
- slug: functional-tests
  similarity: 0.7
- slug: test-coverage-strategy
  similarity: 0.65
- slug: business-quality-scenarios
  similarity: 0.65
- slug: usability-tests
  similarity: 0.65
- slug: regression-testing
  similarity: 0.65
---

## Description

Fachliche Testfälle sind Testszenarien, die gemeinsam mit Fachanwendern verfasst und in Fachsprache formuliert werden, dann von denselben Nutzern überprüft und validiert, um zu bestätigen, dass der Test tatsächlich widerspiegelt, was korrektes Verhalten aus Sicht des Geschäfts bedeutet, statt aus der Annahme eines Entwicklers darüber. Der Mechanismus schließt eine spezifische Lücke: Entwickler, die Tests basierend auf ihrem eigenen Verständnis von Geschäftsregeln schreiben, fangen konstruktionsbedingt nur Abweichungen von genau diesem Verständnis ab und können nicht die Fälle abfangen, in denen das Verständnis des Entwicklers von der Regel von Anfang an falsch war. Dies ist am wichtigsten in Legacy-Systemen, deren Geschäftslogik — Gehaltsabrechnungsberechnungen, Leistungsregeln, Steuerbehandlung — über Jahre des Betriebs reale Randfälle angehäuft hat, die nirgendwo vollständig dokumentiert wurden außer im Gedächtnis der Spezialisten, die Ausnahmen täglich handhaben, und deren Abwesenheit aus jeder Test-Suite genau der Grund ist, warum subtile Berechnungsfehler jahrelang unentdeckt bestehen können. Fachspezialisten direkt in das Schreiben und Überprüfen von Testfällen einzubeziehen bringt genau diese Randfälle ans Licht, weil Domänenexperten Szenarien erleben und sich merken, die eine technische Lektüre des Codes nie zu suchen nahelegen würde. Die laufenden Kosten sind die wiederkehrende Beanspruchung der Zeit und Aufmerksamkeit von Fachanwendern, und das Risiko, dass selbst sie zu den üblichen Fällen tendieren, die sie täglich sehen, statt zu den selteneren Randfällen, wo sich Legacy-Defekte am wahrscheinlichsten verstecken.

## How to Apply ◆

- Arbeiten Sie mit Fachanwendern zusammen, um kritische Geschäfts-Workflows zu identifizieren, und übersetzen Sie sie in Testfälle, die in Fachsprache formuliert sind.
- Lassen Sie Fachanwender Testfälle überprüfen und validieren, um sicherzustellen, dass sie das erwartete Systemverhalten korrekt widerspiegeln.
- Decken Sie sowohl Standardfälle als auch wichtige Randfälle ab, die Fachanwender im täglichen Betrieb erleben.
- Nutzen Sie Testfälle als Akzeptanzkriterien für Entwicklungsarbeit und stellen Sie sicher, dass gelieferte Features den Geschäftserwartungen entsprechen.
- Automatisieren Sie fachliche Testfälle, wo möglich, um häufiges Regressionstesten von Legacy-Funktionalität zu ermöglichen.
- Pflegen Sie eine nachvollziehbare Verbindung zwischen Geschäftsanforderungen und ihren entsprechenden Testfällen.

## Tradeoffs ⇄

**Vorteile:**
- Stellt sicher, dass Tests tatsächliche Geschäftsbedürfnisse statt technischer Annahmen widerspiegeln.
- Bindet Fachanwender in die Qualitätssicherung ein, was das Vertrauen in Systemverhalten verbessert.
- Schafft Testdokumentation, die Geschäfts-Stakeholder verstehen und validieren können.
- Fängt Geschäftslogikfehler ab, die Entwickler möglicherweise nicht erkennen.

**Kosten:**
- Erfordert Zeit und Verfügbarkeit von Fachanwendern, die konkurrierende Prioritäten haben könnten.
- Fachanwender könnten sich auf übliche Szenarien fokussieren und Randfälle übersehen.
- Die Aktualisierung fachlicher Testfälle erfordert laufende Zusammenarbeit, während sich Anforderungen ändern.
- Die Übersetzung zwischen Fachsprache und automatisierten Tests kann Diskrepanzen einführen.

## How It Could Be

Ein Legacy-HR-System handhabt Gehaltsabrechnungsberechnungen mit komplexen Regeln für Überstunden, Leistungen und Steuerabzüge. Entwickler haben Unit-Tests basierend auf ihrem Verständnis der Regeln geschrieben, aber Gehaltsabrechnungsfehler bestehen weiter. Das Team bindet Gehaltsabrechnungsspezialisten ein, um fachliche Testfälle mit realen Szenarien zu erstellen, einschließlich Randfälle, die sie regelmäßig erleben: Mitarbeiter, die mitten in einer Lohnperiode Leistungspläne ändern, rückwirkende Gehaltsanpassungen und Steuersituationen über mehrere Bundesstaaten hinweg. Die Gehaltsabrechnungsspezialisten überprüfen monatlich automatisierte Testergebnisse, und mehrere ihrer Randfallszenarien offenbaren Berechnungsfehler, die seit Jahren falsche Gehaltsabrechnungen produziert haben. Diese fachlich validierten Testfälle werden zur maßgeblichen Verifikations-Suite für alle Änderungen an der Gehaltsabrechnungslogik.
