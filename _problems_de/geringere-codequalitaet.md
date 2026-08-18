---
title: Geringere Codequalität
description: Ausgebrannte oder gehetzte Entwickler machen mit höherer Wahrscheinlichkeit
  Fehler, was zu einer Zunahme von Defekten führt.
category:
- Code
- Process
- Team
related_problems:
- slug: developer-frustration-and-burnout
  similarity: 0.75
- slug: quality-degradation
  similarity: 0.7
- slug: inadequate-code-reviews
  similarity: 0.7
- slug: increased-stress-and-burnout
  similarity: 0.7
- slug: increased-cognitive-load
  similarity: 0.7
- slug: increased-cost-of-development
  similarity: 0.7
solutions:
- definition-of-done
- pair-and-mob-programming
- code-metrics
- code-reviews
- refactoring-katas
- secure-coding-guidelines
- static-code-analysis
- code-quality-gates
layout: problem
lang: de
en_slug: lower-code-quality
---

## Description

Geringere Codequalität tritt auf, wenn verschiedene Belastungen und Umstände dazu führen, dass Entwickler Code produzieren, der nicht den etablierten Standards für Wartbarkeit, Zuverlässigkeit oder Korrektheit entspricht. Diese Verschlechterung resultiert oft aus Burnout, Zeitdruck, mangelnder Motivation oder systemischen Problemen, die Entwickler daran hindern, ihre besten Praktiken anzuwenden. Anders als isolierte Qualitätsprobleme stellt dies einen systematischen Rückgang des Gesamtstandards des vom Team produzierten Codes dar.

## Indicators ⟡
- Code-Review-Kommentare fokussieren sich zunehmend auf grundlegende Qualitätsprobleme
- Fehlerraten steigen selbst bei erfahrenen Entwicklern
- Coding-Standards werden häufig ignoriert oder inkonsistent angewendet
- Technische Schulden häufen sich schneller an, als sie behoben werden
- Entwickler äußern Frustration darüber, keine Zeit zu haben, „es richtig zu machen"

## Symptoms ▲

- [Hohe Rate an neu eingeführten Fehlern](hohe-rate-an-neu-eingefuehrten-fehlern.md)
<br/>  Code geringerer Qualität enthält mehr Defekte, was direkt die Rate erhöht, mit der neue Fehler ins System eingeführt werden.
- [Wartungs-Overhead](wartungs-overhead.md)
<br/>  Schlecht geschriebener Code erfordert mehr Aufwand zum Verstehen, Modifizieren und Beheben, was die laufende Wartungslast erhöht.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Konsistent geringere Codequalität häuft technische Schulden an, während sich Abkürzungen und schlechte Implementierungen anhäufen.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Code, der ohne angemessene Sorgfalt, Tests oder Design geschrieben wurde, ist brüchiger und verursacht mit höherer Wahrscheinlichkeit Regressionen bei Modifikation.
- [Zunehmende Brüchigkeit](zunehmende-bruechigkeit.md)
<br/>  Code geringer Qualität mit fehlender Fehlerbehandlung, schwachen Abstraktionen und schlechter Struktur wird über die Zeit zunehmend brüchig.

## Causes ▼

- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Ausgebrannte Entwickler fehlt die Motivation und mentale Energie, hochwertigen Code zu schreiben, was zu Abkürzungen und Fehlern führt.
- [Zeitdruck](zeitdruck.md)
<br/>  Der Druck, schnell zu liefern, zwingt Entwickler, bei Codequalität, Tests und Design Abstriche zu machen.
- [Termindruck](termindruck.md)
<br/>  Aggressive Termine führen dazu, dass Entwickler bewährte Praktiken wie Code-Reviews, Tests und Refactoring überspringen.
- [Auswirkung von Team-Fluktuation](auswirkung-von-team-fluktuation.md)
<br/>  Der Verlust erfahrener Entwickler lässt weniger erfahrene Teammitglieder ohne angemessene Anleitung Code geringerer Qualität produzieren.
- [Oberflächliche Code-Reviews](oberflaechliche-code-reviews.md)
<br/>  Wenn Code-Reviews Qualitätsprobleme nicht erkennen, wird Code geringerer Qualität unwidersprochen gemerged, was schlechte Standards normalisiert.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Unerfahrene Entwickler ohne Kenntnis bewährter Praktiken produzieren natürlicherweise Code geringerer Qualität mit mehr Defekten und inkonsistenter Struktur.

## Detection Methods ○
- **Statische Codeanalyse:** Nutzung automatisierter Werkzeuge zur Messung von Codequalitätsmetriken über die Zeit
- **Code-Review-Metriken:** Nachverfolgung der Anzahl und Art der bei Code-Reviews gefundenen Probleme
- **Fehlerdichte-Analyse:** Überwachung von Defektraten und ihrer Korrelation mit Codekomplexität
- **Nachverfolgung technischer Schulden:** Messung der Anhäufung technischer Schulden über die Zeit
- **Entwickler-Feedback:** Befragung von Teammitgliedern zu ihrer Fähigkeit, Qualitätsstandards aufrechtzuerhalten

## Examples

Ein Softwareentwicklungsteam steht unter enormem Druck, ein wichtiges Feature zu liefern, bevor ein Wettbewerber seine Version launcht. Das Management betont wiederholt, dass das Verpassen der Deadline katastrophal für das Geschäft wäre. Unter diesem Druck beginnen Entwickler, Unit-Tests zu überspringen, Coding-Standards zu ignorieren und schnelle Fixes statt ordentlicher Lösungen zu implementieren. Code-Reviews werden oberflächlich, während alle darauf drängen, Änderungen zu genehmigen. Das Team liefert das Feature pünktlich, aber die Codebasis bleibt mit zahlreichen Qualitätsproblemen zurück: Funktionen ohne Fehlerbehandlung, duplizierte Logik, die abstrahiert werden sollte, und komplexe bedingte Anweisungen, die schwer zu verstehen sind. In den folgenden Monaten führen diese Qualitätskompromisse zu Produktionsfehlern, schwieriger Wartung und langsamerer Entwicklung nachfolgender Features. Ein weiteres Beispiel betrifft ein Team, in dem mehrere Senior-Entwickler aus Frustration über die Komplexität des Legacy-Systems gegangen sind. Die verbleibenden Entwickler sind überfordert und demoralisiert, was sie dazu bringt, Features mit minimalem Aufwand zu implementieren, nur um ihre zugewiesenen Aufgaben abzuschließen. Sie hören auf, umfassende Tests zu schreiben, überspringen Refactoring-Möglichkeiten und kopieren Code, statt wiederverwendbare Komponenten zu erstellen. Die Gesamtqualität neuer Code-Ergänzungen sinkt stetig, während das Team sowohl Kapazität als auch Motivation verliert, seine früheren Standards aufrechtzuerhalten.
