---
title: Teilweise Fehlerbehebungen
description: Probleme scheinen gelöst zu sein, tauchen aber in anderen Kontexten
  wieder auf, weil die Korrektur nicht auf alle Instanzen des duplizierten Codes
  angewendet wurde.
category:
- Code
related_problems:
- slug: inconsistent-behavior
  similarity: 0.7
- slug: code-duplication
  similarity: 0.65
- slug: difficult-code-reuse
  similarity: 0.65
- slug: incomplete-projects
  similarity: 0.65
- slug: synchronization-problems
  similarity: 0.65
- slug: incomplete-knowledge
  similarity: 0.65
solutions:
- definition-of-done
- regression-tests
- root-cause-analysis
- characterization-tests
- improvement-budget
- workaround-registry
- defect-triage-process
- code-reviews
- exploratory-testing
- change-impact-analysis
- duplication-detection
layout: problem
lang: de
en_slug: partial-bug-fixes
---

## Description
Teilweise Fehlerbehebungen sind ein häufiges Problem in Softwaresystemen mit hohem Grad an Code-Duplizierung. Sie treten auf, wenn ein Fehler in einer Instanz des duplizierten Codes behoben wird, aber nicht in allen. Dies kann zu einer Reihe von Problemen führen, einschließlich Regressionsfehlern, einem Vertrauensverlust in das System und erheblicher Frustration für das Entwicklungsteam. Teilweise Fehlerbehebungen sind oft ein Zeichen für ein schlecht designtes System mit hohem Grad an Code-Duplizierung.

## Indicators ⟡
- Derselbe Fehler wird immer wieder gemeldet.
- Das Team behebt ständig Regressionsfehler.
- Das Team ist sich nicht sicher, ob ein Fehler behoben wurde.
- Das Team ist nicht in der Lage, von Nutzern gemeldete Fehler zu reproduzieren.

## Symptoms ▲

- [Regressionsfehler](regressionsfehler.md)
<br/>  Fehler, die angeblich behoben wurden, tauchen in anderen Kontexten wieder auf, weil die Korrektur nur auf manche Instanzen des duplizierten Codes angewendet wurde.
- [Nutzerfrustration](nutzerfrustration.md)
<br/>  Nutzer erleben denselben Fehler wiederholt, nachdem ihnen gesagt wurde, er sei behoben, was Frustration und Vertrauensverlust verursacht.
- [Inkonsistentes Verhalten](inkonsistentes-verhalten.md)
<br/>  Derselbe Geschäftsprozess funktioniert in einem Kontext korrekt, scheitert aber in einem anderen, weil die Korrektur nicht einheitlich über duplizierten Code hinweg angewendet wurde.
- [Erhöhte Fehleranzahl](erhoehte-fehleranzahl.md)
<br/>  Jede teilweise Korrektur behebt nur eine Instanz des Fehlers, während andere offen bleiben, was die Defektanzahl hoch hält.

## Causes ▼

- [Code-Duplizierung](code-duplizierung.md)
<br/>  Duplizierter Code ist der primäre Ermöglicher teilweiser Fehlerbehebungen, da dieselbe Logik an mehreren Stellen existiert, die alle aktualisiert werden müssen.
- [Unvollständiges Wissen](unvollstaendiges-wissen.md)
<br/>  Entwickler sind sich nicht aller Orte bewusst, an denen dieselbe Logik existiert, sodass sie den Fehler dort beheben, wo sie ihn kennen, aber andere Instanzen verpassen.
- [Schlechte Testabdeckung](schlechte-testabdeckung.md)
<br/>  Ohne umfassende Tests, die alle Instanzen duplizierter Logik abdecken, bleiben teilweise Korrekturen unentdeckt, bis Nutzer auf die unbehobenen Instanzen treffen.
- [Zeitdruck](zeitdruck.md)
<br/>  Unter dem Druck, Fehler schnell zu beheben, korrigieren Entwickler die gemeldete Instanz, ohne nach allen Vorkommnissen zu suchen und sie zu beheben.

## Detection Methods ○
- **Code-Duplizierungsanalyse:** Nutzung statischer Analysewerkzeuge zur Identifikation duplizierten Codes.
- **Regressionstests:** Nutzung von Regressionstests zur Verifikation, dass zuvor behobene Fehler nicht wieder aufgetreten sind.
- **Code-Reviews:** Code-Reviews sind eine großartige Methode zur Identifikation teilweiser Fehlerbehebungen.
- **Fehler-Tracking-System:** Nutzung eines zentralisierten Fehler-Tracking-Systems zur Nachverfolgung des Status von Fehlern.

## Examples
Eine E-Commerce-Website hat einen Fehler in ihrem Checkout-Flow. Der Fehler wird im Checkout-Flow für reguläre Kunden behoben, aber nicht im Checkout-Flow für Gastkunden. Infolgedessen ist der Fehler weiterhin im System vorhanden und beeinträchtigt weiterhin Nutzer. Das Problem hätte vermieden werden können, wenn der Entwickler, der den Fehler behoben hat, sich des duplizierten Codes bewusst gewesen wäre und den Fehler an beiden Stellen behoben hätte.
