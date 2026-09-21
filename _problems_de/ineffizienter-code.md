---
title: Ineffizienter Code
description: Der Code, der für die Handhabung einer Anfrage zuständig ist, ist rechnerisch
  teuer oder enthält Performance-Engpässe.
category:
- Performance
related_problems:
- slug: slow-application-performance
  similarity: 0.75
- slug: inefficient-frontend-code
  similarity: 0.75
- slug: slow-database-queries
  similarity: 0.7
- slug: imperative-data-fetching-logic
  similarity: 0.7
- slug: high-api-latency
  similarity: 0.7
- slug: algorithmic-complexity-problems
  similarity: 0.7
solutions:
- efficient-algorithms
- profiling
- serialization-optimization
- memory-hierarchy
- static-code-analysis
- performance-measurements
- code-reviews
- performance-budgets
- load-testing
- continuous-performance-monitoring
layout: problem
lang: de
en_slug: inefficient-code
---

## Description
Ineffizienter Code ist ein breites Problem, das eine Vielzahl von Ursachen haben kann, von der Nutzung ineffizienter Algorithmen und Datenstrukturen bis zu fehlender ordentlicher Optimierung. Es zeichnet sich durch Code aus, der langsam ist, viele Ressourcen verbraucht oder generell schwer zu warten ist. Das Schreiben effizienten Codes erfordert ein tiefes Verständnis der Sprache und der Plattform sowie ein Engagement für Performance und Qualität. Es ist eine essenzielle Fähigkeit für jeden Softwareentwickler.

## Indicators ⟡
- Die Anwendung ist langsam, selbst auf einer leistungsstarken Maschine.
- Die Anwendung nutzt viel CPU oder Speicher.
- Die Anwendung ist nicht skalierbar.
- Der Code ist schwer verständlich und zu warten.

## Symptoms ▲

- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Rechnerisch teurer Code lässt die Anwendung direkt langsam auf Nutzeranfragen reagieren.
- [Aufstauende Task-Queues](aufstauende-task-queues.md)
<br/>  Task-Queues stauen sich auf, was kaskadierende Verzögerungen über das System hinweg verursacht.
- [Ineffizienter Frontend-Code](ineffizienter-frontend-code.md)
<br/>  Wenn sich allgemeine Code-Ineffizienz auf Frontend-Komponenten erstreckt, äußert sie sich als träge UI-Interaktionen und hoher Client-seitiger Ressourcenverbrauch.
- [Unoptimierter Dateizugriff](unoptimierter-dateizugriff.md)
<br/>  Dateizugriffsmuster sind nicht optimiert, was I/O-Engpässe verursacht.

## Causes ▼

- [Probleme mit algorithmischer Komplexität](probleme-mit-algorithmischer-komplexitaet.md)
<br/>  Die Nutzung ineffizienter Algorithmen und Datenstrukturen ist eine primäre Ursache rechnerisch teuren Codes.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwicklern, denen Wissen über Performance-Optimierung fehlt, schreiben unnötig ressourcenintensiven Code.
- [Unzureichendes Code-Review](unzureichendes-code-review.md)
<br/>  Ohne Code-Reviews bleiben Performance-Antipatterns unentdeckt und häufen sich in der Codebasis an.
- [Geringere Codequalität](geringere-codequalitaet.md)
<br/>  Minderwertiger Code enthält tendenziell redundante Operationen, unnötige Allokationen und andere Ineffizienzen.

## Detection Methods ○

- **Profiler:** Nutzung eines Profilers zur Analyse der CPU- und Speichernutzung der Anwendung und zur Identifikation der genauen Codezeilen, die den Engpass verursachen.
- **Code-Review:** Sorgfältige Überprüfung des Codes auf verbreitete Performance-Antipatterns.
- **Benchmarking:** Schreiben von Benchmarks zur Messung der Performance bestimmter Codeteile und Nachverfolgung der Performance über die Zeit.
- **Statische Analyse:** Nutzung statischer Analysewerkzeuge zur automatischen Erkennung potenzieller Performance-Probleme im Code.

## Examples
Eine Social-Media-Anwendung hat ein Feature, das die Timeline eines Nutzers anzeigt. Die Timeline wird von einer Funktion erzeugt, die über alle Freunde des Nutzers iteriert und dann über all deren Beiträge iteriert, um die neuesten zu finden. Diese verschachtelte Schleife ist sehr ineffizient und lässt die Timeline für Nutzer mit vielen Freunden langsam laden. In einem anderen Fall liest eine Datenverarbeitungsanwendung eine große Datei in den Speicher und verarbeitet sie dann zeilenweise. Dies ist ineffizient, weil es viel Speicher verbraucht. Ein besserer Ansatz wäre, die Datei zeilenweise zu verarbeiten, ohne die gesamte Datei zuerst in den Speicher zu lesen. Dies ist ein verbreitetes Problem in Anwendungen, die über eine lange Zeit von vielen unterschiedlichen Entwicklern entwickelt wurden. Über die Zeit kann die Codebasis komplex und schwer verständlich werden, was es leicht macht, Performance-Engpässe einzuführen.
