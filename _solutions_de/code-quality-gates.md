---
title: Code-Quality-Gates
description: Sicherstellung der Codequalität durch standardisierte, automatisierte
  Prüfungen.
category:
- Process
- Code
problems:
- lower-code-quality
- high-technical-debt
- quality-degradation
- inconsistent-quality
- insufficient-code-review
- high-bug-introduction-rate
- regression-bugs
- quality-blind-spots
- automated-tooling-ineffectiveness
- feature-creep-without-refactoring
- inadequate-initial-reviews
- increased-technical-shortcuts
- mixed-coding-styles
- outdated-tests
- reduced-feature-quality
- review-process-avoidance
- rushed-approvals
- increased-bug-count
- style-arguments-in-code-reviews
- test-debt
- convenience-driven-development
- nitpicking-culture
- rapid-prototyping-becoming-production
- undefined-code-style-guidelines
layout: solution
lang: de
en_slug: code-quality-gates
related_solutions:
- slug: code-metrics
  similarity: 0.8
- slug: static-analysis-and-linting
  similarity: 0.8
- slug: code-review-process-reform
  similarity: 0.8
- slug: quality-ratchet
  similarity: 0.8
- slug: test-coverage-strategy
  similarity: 0.75
- slug: ci-cd-pipeline
  similarity: 0.75
---

## Description

Ein Code-Quality-Gate ist eine automatisierte, nicht verhandelbare Prüfung — Ergebnisse statischer Analyse, Testabdeckungsschwellen, Komplexitätsgrenzen, Abhängigkeits- und Sicherheitsscans —, die eine Codeänderung bestehen muss, bevor sie gemergt werden kann, mechanisch in der CI/CD-Pipeline durchgesetzt statt von der individuellen Sorgfalt oder Stimmung eines Reviewers abzuhängen. Weil die Prüfungen automatisch bei jedem Pull Request laufen, wenden sie denselben Standard einheitlich an, unabhängig davon, wer die Änderung einreicht oder wie eilig der Review-Zyklus ist. Dies adressiert ein besonders in Legacy-Systemen übliches Muster, bei dem neuer Code dazu tendiert, der Qualität des bereits umgebenden Codes zu entsprechen, einfach weil „das ist, wie es hier gemacht wird", was die Gesamtqualität mit jeder Ergänzung schrittweise erodiert, es sei denn, etwas stoppt diese Drift aktiv. Gates auf eine bereits große, inkonsistente Legacy-Codebasis einzuführen muss von Schwellen ausgehen, die auf ihre aktuelle, oft schlechte, Baseline kalibriert sind, statt auf ein idealisiertes Ziel, da am ersten Tag zu streng gesetzte Gates umgangen oder deaktiviert werden, statt Verbesserung anzutreiben; Schwellen werden dann schrittweise verschärft, während sich die Codebasis tatsächlich verbessert. Eine Coverage-Ratsche — die verlangt, dass neuer Code eine höhere Messlatte erfüllt als die Legacy-Baseline, neben der er hinzugefügt wird — ist ein üblicher Mechanismus dafür, der es der Gesamtqualität erlaubt, sich schrittweise zu verbessern, ohne einen sofortigen, unrealistischen Sprung zu verlangen. Quality Gates befreien Reviewer, ihre Aufmerksamkeit auf Design und Logik zu richten, statt Dinge mechanisch zu prüfen, die ein Werkzeug schneller und konsistenter prüfen kann, obwohl sie nur messen, was Tooling erkennen kann, und überhaupt kein Signal zu tieferen Design- oder Architekturangemessenheitsfragen liefern.

## How to Apply ◆

> In Legacy-Systemen verhindern Quality Gates, dass neuer Code die Dinge verschlimmert — sie sind die minimale Investition, die nötig ist, um die Blutung zu stoppen, während die Modernisierung voranschreitet.

- Definieren Sie eine Reihe automatisierter Qualitätsprüfungen, die alle Codeänderungen bestehen müssen, bevor sie gemergt werden: statische Analyse, Testabdeckungsschwellen, Komplexitätsgrenzen und Abhängigkeitsprüfungen.
- Integrieren Sie Quality Gates in die CI/CD-Pipeline, sodass sie automatisch bei jedem Pull Request laufen und sofortiges Feedback ohne manuellen Eingriff bieten.
- Beginnen Sie mit nachsichtigen Schwellen, angemessen für den aktuellen Zustand der Legacy-Codebasis, und verschärfen Sie sie schrittweise — ein am ersten Tag zu hoch gesetztes Gate wird umgangen oder deaktiviert.
- Implementieren Sie eine Coverage-Ratsche, die verlangt, dass neuer Code höhere Coverage-Standards erfüllt als die Legacy-Baseline, was Coverage-Regression verhindert.
- Beziehen Sie Sicherheitsscanning (SAST/DAST) in Quality Gates ein, um Schwachstellen abzufangen, bevor sie Produktion erreichen.
- Machen Sie Quality-Gate-Ergebnisse in Pull Requests sichtbar, sodass sich Reviewer auf Design und Logik statt mechanischer Qualitätsprüfungen fokussieren können.
- Überprüfen und passen Sie Gate-Kriterien vierteljährlich basierend auf der Erfahrung des Teams an — Gates, die zu viele False Positives produzieren, werden ignoriert.

## Tradeoffs ⇄

> Quality Gates verhindern Qualitätsverschlechterung automatisch, erfordern aber Kalibrierung, um weder zu nachsichtig noch zu restriktiv zu sein.

**Vorteile:**

- Verhindert das übliche Legacy-Systemmuster, dass neuer Code so schlecht ist wie bestehender Code, weil „das ist, wie es hier gemacht wird".
- Bietet objektive, konsistente Qualitätsdurchsetzung, die nicht von individueller Reviewer-Sorgfalt abhängt.
- Befreit Code-Reviewer, sich auf höherstufige Belange zu fokussieren, indem mechanische Qualitätsprüfungen automatisiert werden.
- Schafft einen messbaren Qualitätsboden, der sich über die Zeit verbessert, während Schwellen verschärft werden.
- Macht Qualitätserwartungen für alle Entwickler explizit und transparent.

**Kosten und Risiken:**

- Gates, die für eine Legacy-Codebasis zu streng sind, schaffen Reibung und könnten durch Workarounds oder Ausnahmen umgangen werden.
- False Positives von statischen Analysewerkzeugen können das Vertrauen in den Quality-Gate-Prozess erodieren.
- Quality Gates messen, was Werkzeuge erkennen können, verpassen aber Designqualität, Namensklarheit und architektonische Angemessenheit.
- Die Pflege der Quality-Gate-Infrastruktur und Werkzeugkonfigurationen erfordert laufenden Aufwand.

## How It Could Be

> Das folgende Szenario demonstriert, wie Quality Gates Qualitätsverschlechterung in einem Legacy-System stoppen.

Die Legacy-Plattform eines SaaS-Unternehmens hatte keine automatisierten Qualitätsprüfungen, und Code-Reviews waren inkonsistent — manche Reviewer prüften Qualität rigoros, während andere alles genehmigten, was kompilierte. Über fünf Jahre führte dies zu einer Codebasis, in der die Qualität zwischen Modulen stark variierte. Das Team führte Quality Gates ein, die verlangten: mindestens 70 % Zeilencoverage für geänderte Dateien, keine neuen kritischen oder schwerwiegenden statischen Analyseprobleme, keine TODO-Kommentare ohne verknüpfte Tickets, und alle Abhängigkeiten auf genehmigten Versionen. Der anfängliche Widerstand war hoch, mit 40 % der Pull Requests, die im ersten Monat an Gates scheiterten. Aber innerhalb von drei Monaten sanken die Fehlerraten auf 15 %, während Entwickler die Standards verinnerlichten. Nach einem Jahr verschärfte das Team die Coverage-Anforderungen auf 80 % für neuen Code und fügte Komplexitätsschwellen hinzu. Die Produktionsdefektrate in neuen Features sank um 45 % im Vergleich zur Baseline vor den Gates.
