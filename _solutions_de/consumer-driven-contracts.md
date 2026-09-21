---
title: Consumer-Driven Contracts
description: Verträge, die die Erwartungen der Schnittstellennutzer definieren.
category:
- Testing
- Architecture
problems:
- breaking-changes
- integration-difficulties
- poor-interfaces-between-applications
- api-versioning-conflicts
- inadequate-integration-tests
- fear-of-breaking-changes
- microservice-communication-overhead
- communication-risk-outside-project
- poor-contract-design
- rapid-system-changes
layout: solution
lang: de
en_slug: consumer-driven-contracts
related_solutions:
- slug: contract-testing
  similarity: 0.85
- slug: api-first-development
  similarity: 0.75
- slug: backward-compatible-apis
  similarity: 0.7
- slug: design-by-contract
  similarity: 0.7
- slug: integration-tests
  similarity: 0.7
- slug: abstraction
  similarity: 0.65
---

## Description

Consumer-Driven Contracts kehren die übliche Richtung des Schnittstellentestens um: Statt dass ein Anbieter einseitig entscheidet, wie seine Schnittstelle aussieht, und hofft, dass Konsumenten mithalten, spezifiziert jeder Konsument genau, auf welche Felder, Endpunkte und Verhaltensweisen er tatsächlich angewiesen ist, und diese Spezifikation wird zu einem ausführbaren Vertrag, den der Anbieter bei jeder Änderung erfüllen muss. Werkzeuge wie Pact führen diese Verträge in der CI-Pipeline des Anbieters aus, sodass eine Änderung, die still einen Konsumenten brechen würde, den Build vor dem Mergen scheitern lässt, statt nachdem sie Produktion erreicht. Dies ist am wichtigsten in Legacy-Landschaften, die zu vielen Services mit impliziten, undokumentierten Abhängigkeiten zwischeneinander gewachsen sind, wo niemand auf der Anbieterseite die tatsächliche Nutzung einer Schnittstelle durch jeden Konsumenten aus dem Gedächtnis aufzählen kann, und wo Breaking Changes historisch nur als Produktionsvorfälle auftauchten. Weil Verträge die Schnittstellenform erfassen, auf die sich Konsumenten verlassen, statt vollständiges End-to-End-Verhalten, sind sie günstiger auszuführen und zu pflegen als breite Integrations-Test-Suiten, und sie erlauben es Teams, einige dieser brüchigen Integrationstests vollständig zu ersetzen. Der Ansatz erfordert jedoch, dass Konsumententeams ihre Verträge schreiben und aktuell halten, und er bietet nur Sicherheit für die Interaktionen, die tatsächlich unter Vertrag stehen, sodass die Praxis am wertvollsten ist, wenn sie zuerst auf die fragilsten oder geschäftskritischsten Integrationspunkte angewendet wird.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Lassen Sie jeden Konsumenten einen Vertrag definieren, der genau spezifiziert, auf welche Felder, Endpunkte und Verhaltensweisen er sich verlässt
- Nutzen Sie ein Vertragstest-Werkzeug (z. B. Pact), um Anbieteränderungen gegen alle registrierten Konsumentenverträge zu verifizieren
- Führen Sie Vertragstests in der CI-Pipeline des Anbieters aus, sodass Breaking Changes vor dem Mergen abgefangen werden
- Beginnen Sie damit, Verträge für die kritischsten oder fragilsten Integrationspunkte in der Legacy-Landschaft hinzuzufügen
- Speichern Sie Verträge in einem gemeinsamen Broker oder Repository, zugänglich für sowohl Konsumenten- als auch Anbieterteams
- Nutzen Sie Vertragstests, um brüchige End-to-End-Integrationstests zu ersetzen, wo möglich

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Anbieter wissen genau, auf welche Teile ihrer Schnittstelle Konsumenten angewiesen sind, was sichere Evolution ermöglicht
- Fängt Breaking Changes zur Build-Zeit statt in Produktion ab
- Ermöglicht unabhängiges Deployment von Services ohne koordinierte Release-Fenster

**Kosten und Risiken:**
- Erfordert, dass Konsumententeams ihre Verträge schreiben und pflegen, was teamübergreifende Koordination hinzufügt
- Vertragstest-Werkzeuge haben eine Lernkurve und Infrastrukturanforderungen
- Verträge testen nur die Schnittstellenform, nicht vollständiges Integrationsverhalten
- Veraltete Verträge können falsches Vertrauen geben, wenn Konsumententeams sie nicht aktualisieren

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Reisebuchungsplattform hatte 12 Microservices mit häufigen Integrationsfehlern, weil Backend-Änderungen unwissentlich Frontend-Erwartungen brachen. Das Team führte Pact-basierte Consumer-Driven Contracts für die fünf kritischsten Servicegrenzen ein. Innerhalb von drei Monaten fingen die Vertragstests 14 potenzielle Breaking Changes während des Code-Reviews ab, und integrationsbezogene Produktionsvorfälle sanken von einem wöchentlichen Vorkommnis auf ungefähr eines pro Quartal.
