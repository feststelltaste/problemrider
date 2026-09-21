---
title: Statische Codeanalyse
description: Automatische Prüfung von Quellcode auf Programmierfehler und
  Sicherheitslücken.
category:
- Security
- Code
- Testing
problems:
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- buffer-overflow-vulnerabilities
- lower-code-quality
- inconsistent-coding-standards
- high-bug-introduction-rate
- legacy-code-without-tests
- inadequate-code-reviews
- inefficient-code
- gradual-performance-degradation
- code-review-inefficiency
- difficult-code-comprehension
- queries-that-prevent-index-usage
- unused-indexes
- algorithmic-complexity-problems
- alignment-and-padding-issues
- n-plus-one-query-problem
- atomic-operation-overhead
- data-structure-cache-inefficiency
- dma-coherency-issues
- endianness-conversion-overhead
- false-sharing
- interrupt-overhead
- memory-barrier-inefficiency
layout: solution
lang: de
en_slug: static-code-analysis
related_solutions:
- slug: security-tests
  similarity: 0.85
- slug: static-analysis-and-linting
  similarity: 0.8
- slug: dynamic-code-analysis
  similarity: 0.8
- slug: secure-coding-guidelines
  similarity: 0.8
- slug: vulnerability-scans
  similarity: 0.8
- slug: secure-software
  similarity: 0.8
---

## Description

Statische Codeanalyse ist die automatisierte Inspektion von Quellcode ohne dessen Ausführung, unter Nutzung von Werkzeugen wie SonarQube, ESLint, PMD oder FindBugs, um Programmierfehler, Sicherheitslücken und Qualitäts- oder Performance-Antipatterns durch Musterabgleich gegen die Struktur des Codes zu erkennen. Anders als manuelles Code-Review skaliert sie zu Codebasen jeder Größe bei festen, wiederholbaren Kosten, was sie besonders wertvoll für Legacy-Systeme macht, wo die schiere Menge an Code — oft Hunderttausende von Zeilen, die über viele Jahre angesammelt wurden — erschöpfendes manuelles Sicherheits- oder Qualitätsreview unpraktikabel macht. Da statische Analysewerkzeuge bekannte Schwachstellenmuster (SQL-Injection, Buffer Overflows, Cross-Site Scripting) und Qualitäts-Antipatterns als Regeln kodieren, bringen sie Probleme zutage, die aktuellem sicherem-Codieren-Bewusstsein vorausgehen und für die niemand seit dem Schreiben des Codes die Zeit oder den Grund hatte, danach zu suchen. In Legacy-Kontexten liegt die praktische Herausforderung weniger im Ausführen des Werkzeugs als in der Triage: Ein anfänglicher Scan einer alten, ungeprüften Codebasis produziert routinemäßig Tausende von Befunden, von denen die meisten niedrigere Priorität haben oder falsch positiv sind, sodass der Wert des Werkzeugs davon abhängt, eine Baseline zu etablieren, neuen Code gegen Regression abzusichern, während der bestehende Rückstand inkrementell abgearbeitet wird, und das Regelwerk anzupassen, um die Alarmmüdigkeit zu vermeiden, die Entwickler dazu bringt, das Werkzeug ganz zu ignorieren. Statische Analyse kann jedoch keine reinen Laufzeitdefekte oder Geschäftslogikfehler erfassen, sodass sie Testing und menschliches Review ergänzt statt ersetzt.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Wählen Sie statische Analysewerkzeuge, die die in der Legacy-Codebasis genutzten Sprachen und Frameworks unterstützen (z. B. SonarQube, ESLint, PMD, FindBugs)
- Konfigurieren Sie Werkzeugregeln, um sich zunächst auf hochschwere Sicherheitsbefunde zu fokussieren, bevor zu Stil- und Qualitätsregeln erweitert wird
- Integrieren Sie statische Analyse in die CI/CD-Pipeline als erforderliche Prüfung für Pull Requests
- Etablieren Sie eine Baseline bestehender Befunde und erstellen Sie einen Plan, sie inkrementell zu reduzieren, statt alle auf einmal zu beheben
- Passen Sie Regeln an, um falsch positive Ergebnisse zu minimieren, die das Entwicklervertrauen in das Tooling untergraben
- Nutzen Sie inkrementelle Analyse, um nur geänderte Dateien zu prüfen, was die Scan-Zeit für große Legacy-Codebasen reduziert
- Schulen Sie Entwickler, statische Analysebefunde effektiv zu interpretieren und darauf zu reagieren
- Verfolgen Sie Befundtrends über die Zeit, um die Wirkung des statischen Analyseprogramms zu messen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Erfasst gängige Schwachstellenmuster und Performance-Antipatterns automatisch ohne manuellen Review-Aufwand
- Bietet konsistentes, objektives Codequalitäts-Feedback unabhängig von der Expertise des Reviewers
- Skaliert zu großen Legacy-Codebasen, wo manuelles Sicherheitsreview unpraktikabel ist
- Schafft eine kontinuierliche Feedback-Schleife, die Entwickler über sichere Codierungsmuster aufklärt

**Kosten und Risiken:**
- Legacy-Codebasen produzieren oft überwältigende Mengen anfänglicher Befunde, die Triage erfordern
- Falsch positive Ergebnisse können zu Alarmmüdigkeit führen und dazu, dass Entwickler echte Befunde ignorieren
- Statische Analyse kann keine Laufzeitschwachstellen, Geschäftslogikfehler oder datenabhängige Performance-Probleme erkennen
- Werkzeugkonfiguration und -pflege erfordert laufenden Aufwand und Expertise
- Manche Legacy-Sprachen oder -Frameworks könnten begrenzte Unterstützung durch statische Analysewerkzeuge haben

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Finanzdienstleistungsunternehmen setzte SonarQube mit sicherheitsfokussierten Regeln auf seiner 500.000-Zeilen-Legacy-Java-Codebasis ein. Der anfängliche Scan produzierte über 3.000 Befunde, die das Team in 180 echte Sicherheitsprobleme, 800 Qualitätsverbesserungen und den Rest als falsch positive oder niedrigpriore Elemente triagierte. Sie konfigurierten das Werkzeug, eine "Null neue Befunde"-Richtlinie auf allen neuen Code durchzusetzen, während sie einen vierteljährlichen Sprint zur Reduzierung des Legacy-Rückstands schufen. Nach einem Jahr war die Legacy-Befundanzahl um 65 % gesunken, und keine neuen kritischen Sicherheitsbefunde wurden in Code eingeführt, der das Gate der statischen Analyse passiert hatte.
