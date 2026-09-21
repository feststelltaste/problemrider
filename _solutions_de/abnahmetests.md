---
title: Abnahmetests
description: Verifikation der Erfüllung von Geschäftsanforderungen durch automatisierte
  Tests.
category:
- Testing
- Requirements
problems:
- insufficient-testing
- poor-test-coverage
- missing-end-to-end-tests
- regression-bugs
- legacy-code-without-tests
- fear-of-change
- inadequate-requirements-gathering
- increased-manual-testing-effort
- reduced-feature-quality
layout: solution
lang: de
en_slug: acceptance-tests
related_solutions:
- slug: automated-tests
  similarity: 0.8
- slug: functional-tests
  similarity: 0.75
- slug: test-coverage-strategy
  similarity: 0.75
- slug: behavior-driven-development-bdd
  similarity: 0.75
- slug: user-acceptance-tests
  similarity: 0.75
- slug: specification-by-example
  similarity: 0.75
---

## Description

Abnahmetests sind automatisierte Tests, die verifizieren, dass ein System seine Geschäftsanforderungen aus der Perspektive eines Endnutzers oder Geschäfts-Stakeholders erfüllt, statt Implementierungsdetails so zu prüfen, wie es Unit-Tests tun. Sie werden typischerweise gegen ein geschäftslesbares Spezifikationsformat geschrieben, mit Werkzeugen wie Cucumber, FitNesse oder Robot Framework, sodass dieselben Szenariodefinitionen von Nicht-Entwicklern verstanden, überprüft und sogar verfasst werden können. In Legacy-Systemen, die nie von automatisierten Tests abgedeckt waren, füllen Abnahmetests eine spezifische und dringende Lücke: Sie erfassen, was das System aktuell auf der für das Geschäft wichtigsten Ebene tun sollte, was dem Team ein Sicherheitsnetz gibt, bevor Code berührt wird, dessen internes Verhalten niemand mehr vollständig versteht. Dies macht Abnahmetests zu einer Voraussetzung für sichere Modernisierungsarbeit wie das Extrahieren von Modulen, das Ersetzen von Komponenten oder die Migration von Plattformen, weil eine bestehende Abnahmesuite direkter Beleg dafür ist, dass eine Änderung das extern sichtbare Geschäftsverhalten nicht verändert hat. Der Aufbau dieser Suite für ein bestehendes Legacy-System erfordert erhebliche Vorabinvestition, da die Tests rückwirkend für bereits bestehende Funktionalität geschrieben werden müssen, statt Test-First neben neuer Entwicklung, und es erfordert auch enge Zusammenarbeit mit Fachexperten, die bestätigen können, dass die Tests tatsächliche Geschäftsabsicht widerspiegeln statt angenommenes Verhalten. Über die Zeit dient die Suite zusätzlich als ausführbare Dokumentation des Systemverhaltens, oft die zuverlässigste Dokumentation, die das Legacy-System hat.

## How to Apply ◆

- Definieren Sie Abnahmekriterien für jede Geschäftsanforderung und übersetzen Sie sie vor oder neben der Implementierung in automatisierte Testfälle.
- Nutzen Sie Frameworks wie Cucumber, FitNesse oder Robot Framework, die es Geschäfts-Stakeholdern erlauben, Testszenarien zu lesen und zu validieren.
- Beginnen Sie mit den kritischsten Legacy-Workflows: Identifizieren Sie die wichtigsten Geschäftsprozesse und erstellen Sie Abnahmetests, die deren korrektes Verhalten verifizieren.
- Führen Sie Abnahmetests als Teil der CI/CD-Pipeline aus, um Regressionen vor dem Deployment zu erfassen.
- Nutzen Sie Abnahmetests als Sicherheitsnetz vor dem Refactoring von Legacy-Code, um sicherzustellen, dass bestehendes Verhalten erhalten bleibt.
- Beziehen Sie Fachexperten in die Überprüfung und das Verfassen von Testszenarien ein, um sicherzustellen, dass Tests tatsächliche Geschäftsabsicht widerspiegeln.

## Tradeoffs ⇄

**Vorteile:**
- Liefert Vertrauen, dass Geschäftsanforderungen nach Änderungen an Legacy-Code erfüllt werden.
- Schafft ausführbare Dokumentation des erwarteten Systemverhaltens.
- Überbrückt die Lücke zwischen Geschäfts-Stakeholdern und Entwicklern durch geteilte Testsprache.
- Ermöglicht sichereres Refactoring und Modernisierung durch Erkennung funktionaler Regressionen.

**Kosten:**
- Das Schreiben von Abnahmetests für bestehende Legacy-Funktionalität erfordert erhebliche Vorabinvestition.
- Tests können brüchig werden, wenn sie von UI-Elementen oder spezifischen Implementierungsdetails abhängen.
- Die Wartung einer großen Abnahmetest-Suite erfordert anhaltenden Aufwand, während sich Anforderungen weiterentwickeln.
- Langsame Ausführungszeiten für umfassende Abnahmetest-Suiten können Feedback verzögern.

## How It Could Be

Ein Einzelhandelsunternehmen erbt ein Legacy-Bestellverwaltungssystem ohne automatisierte Tests. Vor Beginn der Modernisierung arbeitet das Team mit Geschäftsanalysten zusammen, um die zwanzig kritischsten Bestell-Workflows zu identifizieren, und schreibt für jeden Abnahmetests mit Cucumber. Diese Tests verifizieren End-to-End-Verhalten einschließlich Bestellerstellung, Zahlungsverarbeitung, Bestandsaktualisierungen und Benachrichtigungslieferung. Als das Team später das Zahlungsmodul in einen separaten Service extrahiert, erfassen die Abnahmetests drei subtile Regressionen in der Rabattberechnungslogik, die Unit-Tests nicht entdeckt hätten. Die Testsuite wird zur definitiven Spezifikation korrekten Verhaltens, auf die sowohl Entwickler als auch Geschäfts-Stakeholder während Planungsdiskussionen verweisen.
