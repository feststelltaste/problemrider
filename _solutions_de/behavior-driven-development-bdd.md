---
title: Behavior-Driven Development (BDD)
description: Entwicklung auf Basis erwarteten Systemverhaltens.
category:
- Testing
- Process
problems:
- requirements-ambiguity
- insufficient-testing
- legacy-code-without-tests
- stakeholder-developer-communication-gap
- misaligned-deliverables
- poor-test-coverage
- implementation-rework
- inadequate-requirements-gathering
- regression-bugs
layout: solution
lang: de
en_slug: behavior-driven-development-bdd
related_solutions:
- slug: specification-by-example
  similarity: 0.85
- slug: user-stories
  similarity: 0.75
- slug: acceptance-tests
  similarity: 0.75
- slug: evolutionary-requirements-development
  similarity: 0.75
- slug: feature-driven-development
  similarity: 0.75
- slug: strangler-fig-pattern
  similarity: 0.75
---

## Description

Behavior-Driven Development strukturiert sowohl Spezifikation als auch Testen um Given-When-Then-Szenarien, geschrieben in einer gemeinsamen, halbformalen Sprache, die Fachexperten, Tester und Entwickler alle lesen, überprüfen und als automatisierte Tests durch Frameworks wie Cucumber oder SpecFlow ausführen können. Der Mechanismus zwingt dazu, Geschäftsverhalten explizit und unmissverständlich zu formulieren, bevor die Implementierung beginnt, und hält diese Formulierung dann ausführbar, sodass sie nicht still aus der Synchronisation mit dem Code driften kann, wie es ein Prosa-Spezifikationsdokument typischerweise tut. Dies ist besonders wertvoll in der Legacy-Modernisierung, weil das tatsächliche Verhalten eines alten Systems häufig undokumentiert ist, nur in Codepfaden und dem Gedächtnis einer kleinen Anzahl langjähriger Entwickler kodiert, und jede Neuschreibung riskiert, still von Jahrzehnten angehäufter Geschäftsregeln und Randfälle abzuweichen, die nirgendwo sonst aufgeschrieben wurden. Das Ausführen von BDD-Szenarien, die gegen das aktuelle Verhalten des Legacy-Systems geschrieben wurden, gegen eine neue Implementierung gibt eine konkrete, geschäftslesbare Definition von „erfolgreich migriert" — ein Szenario besteht entweder oder es besteht nicht —, was ein vages, subjektives Gefühl ersetzt, dass eine Migration „richtig aussieht". Die Kosten sind, dass diese Technik von dauerhaftem Zugriff auf Fachexperten abhängt, um Szenarien zu verfassen und zu validieren, und Szenarien, die zu technisch oder zu oberflächlich geschrieben werden, verlieren den Kommunikationswert, der BDDs gesamter Existenzgrund gegenüber konventionellen automatisierten Tests ist.

## How to Apply ◆

> In der Legacy-Modernisierung schafft BDD eine gemeinsame Spezifikationssprache, die das Verhalten des Legacy-Systems in einem Format erfasst, das sowohl Fachexperten als auch Entwickler verifizieren können.

- Nutzen Sie Given-When-Then-Szenarien, um das aktuelle Verhalten des Legacy-Systems zu dokumentieren, bevor Sie es ändern, und schaffen Sie ausführbare Spezifikationen, die sowohl als Dokumentation als auch als Regressionstests dienen.
- Führen Sie „Three Amigos"-Sitzungen (Entwickler, Tester, Fachexperte) durch, um BDD-Szenarien für jedes zu migrierende Legacy-Feature zu schreiben und Randfälle zu erfassen, die nur Fachexperten kennen.
- Wählen Sie ein BDD-Framework, das zum Technologie-Stack des Legacy-Systems passt (Cucumber, SpecFlow, Behave), und integrieren Sie es in die Continuous-Integration-Pipeline.
- Schreiben Sie Szenarien auf der Ebene des Geschäftsverhaltens statt auf UI- oder technischer Ebene, sodass sie gültig bleiben, selbst wenn sich die zugrunde liegende Implementierung während der Modernisierung ändert.
- Nutzen Sie BDD-Szenarien als Akzeptanzkriterien für Migrations-Storys — ein Feature gilt als erfolgreich migriert, wenn alle seine BDD-Szenarien gegen die neue Implementierung bestehen.
- Bauen Sie eine nach Geschäftsfähigkeit organisierte Szenariobibliothek, um ein lebendiges Dokumentationssystem zu schaffen, das veraltete Spezifikationsdokumente ersetzt.

## Tradeoffs ⇄

> BDD schafft lebendige Dokumentation und richtet Teams um Verhalten aus, erfordert aber konsistente Beteiligung von Fachexperten.

**Vorteile:**

- Schafft ausführbare Spezifikationen, die sowohl als Tests als auch als Dokumentation dienen, und löst das Problem von Spezifikationen, die von der Implementierung abdriften.
- Überbrückt die Kommunikationslücke zwischen technischen und geschäftlichen Stakeholdern durch Nutzung einer gemeinsamen Sprache, die beide lesen und validieren können.
- Bietet ein klares Migrationsabschluss-Maß — den Prozentsatz der BDD-Szenarien, die gegen das neue System bestehen.
- Fängt Verhaltensregressionen während der Modernisierung ab, die Unit-Tests möglicherweise übersehen, weil sie Implementierung statt Verhalten testen.

**Kosten und Risiken:**

- BDD-Szenarien erfordern laufenden Zugriff auf Fachexperten, die möglicherweise nicht für das benötigte dauerhafte Engagement verfügbar sind.
- Schlecht geschriebene Szenarien, die zu detailliert oder zu technisch sind, verlieren ihren Wert als Kommunikationswerkzeug und werden zu nur einem weiteren Testformat.
- Die Step-Definition-Schicht zwischen Szenarien und Code kann zu einer Wartungslast werden, wenn sie nicht sauber und gut organisiert gehalten wird.
- Teams könnten sich darauf fokussieren, Szenarien für einfache Fälle zu schreiben, und die komplexen Randfälle vermeiden, wo BDD den größten Wert bietet.

## How It Could Be

> Das folgende Szenario zeigt, wie BDD die Legacy-Systemmigration unterstützt.

Ein Logistikunternehmen, das sein Sendungsverfolgungssystem migrierte, nutzte BDD, um die komplexen Geschäftsregeln rund um Lieferzeitfenster-Berechnungen zu erfassen. Das Legacy-System berechnete Lieferfenster unterschiedlich basierend auf Spediteur, Zielzone, Paketgewicht und Servicelevel — Regeln, die nur in prozeduralem Code und dem Gedächtnis zweier Senior-Entwickler existierten. Durch strukturierte BDD-Workshops schrieb das Team 180 Given-When-Then-Szenarien, die alle Kombinationen abdeckten. Als die erste Implementierung der neuen Berechnungs-Engine gegen diese Szenarien getestet wurde, scheiterten 23 — was Randfälle offenbarte, in denen die neue Implementierung vom Legacy-Verhalten abwich. Zwölf davon waren echte Bugs im neuen Code, und elf stellten sich als Bugs im Legacy-System heraus, die das Geschäft beschloss zu beheben statt zu replizieren. Die BDD-Szenarien wurden zur maßgeblichen Spezifikation für Lieferfenster-Berechnungen und überdauerten sowohl das Legacy-System als auch die Amtszeit der ursprünglichen Entwickler.
