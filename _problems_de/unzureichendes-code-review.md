---
title: Unzureichendes Code-Review
description: Code-Review-Prozesse versäumen es, Designfehler, Fehler oder Qualitätsprobleme
  aufgrund unzureichender Tiefe, Zeit oder Expertise zu erfassen.
category:
- Code
- Process
- Team
related_problems:
- slug: inadequate-code-reviews
  similarity: 0.9
- slug: review-process-breakdown
  similarity: 0.8
- slug: inadequate-initial-reviews
  similarity: 0.8
- slug: code-review-inefficiency
  similarity: 0.8
- slug: superficial-code-reviews
  similarity: 0.8
- slug: reviewer-inexperience
  similarity: 0.75
solutions:
- code-review-process-reform
- code-reviews
- code-quality-gates
- fair-source
- code-review-guidelines
- small-change-batches
- work-in-progress-limits
- checklists
- pair-and-mob-programming
- team-working-agreements
layout: problem
lang: de
en_slug: insufficient-code-review
---

## Description

Unzureichendes Code-Review tritt auf, wenn der Code-Review-Prozess es versäumt, Designprobleme, potenzielle Fehler, Sicherheitslücken oder Wartbarkeitsprobleme wirksam zu identifizieren und zu beheben, bevor der Code die Produktion erreicht. Dies kann aus übereilten Reviews, fehlender Reviewer-Expertise, unzureichenden Review-Richtlinien oder kulturellen Problemen resultieren, die gründliches Feedback entmutigen. Schlechtes Code-Review erlaubt es problematischem Code, sich anzuhäufen, was die Gesamtsystemqualität verringert.

## Indicators ⟡

- Code-Reviews werden sehr schnell ohne substanzielles Feedback abgeschlossen
- Reviews konzentrieren sich primär auf Formatierung und Stil statt auf Logik und Design
- Komplexe Änderungen erhalten das gleiche Reviewniveau wie triviale Änderungen
- Reviewer genehmigen Code, den sie nicht vollständig verstehen
- Reviews werden als Formalität behandelt statt als Qualitätstor

## Symptoms ▲

- [Hohe Fehlerrate in Produktion](hohe-fehlerrate-in-produktion.md)
<br/>  Wenn Reviews es versäumen, Fehler und Designfehler zu erfassen, erreichen mehr Defekte die Produktion.
- [Hohe Rate an neu eingeführten Fehlern](hohe-rate-an-neu-eingefuehrten-fehlern.md)
<br/>  Ohne gründliche Reviews als Qualitätstor werden neue Fehler in höherer Rate eingeführt.
- [Geringere Codequalität](geringere-codequalitaet.md)
<br/>  Unzureichendes Review erlaubt es, dass sich schlechte Designmuster und Codequalitätsprobleme in der Codebasis anhäufen.
- [Inkonsistente Codebasis](inkonsistente-codebasis.md)
<br/>  Ohne wirksame Reviews, die Standards durchsetzen, driften Coding-Stile und -Muster über die Codebasis auseinander.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Designfehler und Abkürzungen, die durch unzureichende Reviews gelangen, häufen sich als technische Schulden an.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Reviews, die Nebeneffekte und Kopplungsprobleme übersehen, führen zu Regressionsfehlern, wenn sich Code ändert.
- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Unzureichendes Code-Review erhöht direkt das Fehlerrisiko, weil das Qualitätstor, das Defekte erfasst, geschwächt ist.

## Causes ▼

- [Termindruck](termindruck.md)
<br/>  Zeitdruck führt dazu, dass Reviewer Reviews überstürzen oder sie ganz überspringen.
- [Überlastete Teams](ueberlastete-teams.md)
<br/>  Überlasteten Teammitgliedern fehlt die Zeit und mentale Energie, um gründliche Code-Reviews durchzuführen.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Wenn Reviewern die Expertise fehlt, können sie Designfehler oder subtile Fehler im Code nicht identifizieren.
- [Teammitglieder nicht in den Review-Prozess eingebunden](teammitglieder-nicht-in-den-review-prozess-eingebunden.md)
<br/>  Nicht eingebundene Reviewer behandeln Reviews als Formalität statt als bedeutsames Qualitätstor.
- [Große Pull Requests](grosse-pull-requests.md)
<br/>  Sehr große Codeänderungen überfordern Reviewer und machen gründliches Review praktisch unmöglich.

## Detection Methods ○

- **Review-Tiefen-Analyse:** Messung der für Reviews aufgewendeten Zeit im Verhältnis zur Codekomplexität
- **Problem-Entdeckungsrate:** Nachverfolgung, wie viele Probleme in Produktion vs. während des Reviews gefunden werden
- **Review-Kommentar-Qualität:** Analyse von Arten und Tiefe des in Reviews gegebenen Feedbacks
- **Bewertung der Reviewer-Expertise:** Bewertung, ob Reviewer angemessenes Wissen für den zu reviewenden Code haben
- **Korrelation von Fehlern nach Review:** Vergleich von Fehlerraten für gründlich reviewten vs. leicht reviewten Code

## Examples

Ein Entwicklungsteam führt Code-Reviews durch, aber Reviewer verbringen typischerweise nur 5-10 Minuten damit, komplexe Änderungen mit Hunderten von Codezeilen zu reviewen. Reviews konzentrieren sich auf offensichtliche Syntaxfehler und Formatierungsprobleme, während sie architektonische Probleme, ineffiziente Algorithmen und potenzielle Sicherheitslücken übersehen. Ein komplexes Authentifizierungsmodul besteht das Review trotz eines subtilen Logikfehlers, der unbefugten Zugriff unter bestimmten Bedingungen erlaubt. Die Schwachstelle wird erst entdeckt, als Sicherheitstests das Problem Wochen später aufdecken, was Notfall-Fixes und Sicherheitspatches erfordert. Ein weiteres Beispiel betrifft ein Team, in dem Senior-Entwickler zu beschäftigt sind, um gründliche Reviews durchzuführen, sodass Junior-Entwickler den Code der anderen ohne ausreichende Expertise reviewen, um Designprobleme zu identifizieren. Ein performancekritisches Modul wird genehmigt, obwohl es ineffiziente Datenstrukturen und Algorithmen nutzt, die erhebliche Verlangsamungen in Produktion verursachen. Die Performance-Probleme werden erst entdeckt, als das System unter hoher Last steht, was umfangreiches Refactoring erfordert, das mit erfahrenerem Review hätte vermieden werden können.
