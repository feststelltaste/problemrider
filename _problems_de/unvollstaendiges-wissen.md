---
title: Unvollständiges Wissen
description: Entwickler sind sich nicht aller Orte bewusst, an denen ähnliche Logik
  existiert, was zu Synchronisationsproblemen und anderen Problemen führen kann.
category:
- Communication
- Team
related_problems:
- slug: knowledge-gaps
  similarity: 0.7
- slug: team-silos
  similarity: 0.7
- slug: incomplete-projects
  similarity: 0.65
- slug: inconsistent-behavior
  similarity: 0.65
- slug: insufficient-design-skills
  similarity: 0.65
- slug: inexperienced-developers
  similarity: 0.65
solutions:
- architecture-decision-records
- documentation-as-code
- knowledge-sharing-practices
- domain-quiz
- code-reading-sessions
- knowledge-rotation
- knowledge-base
- living-documentation
- pair-and-mob-programming
- internal-technical-coaching
- domain-immersion
layout: problem
lang: de
en_slug: incomplete-knowledge
---

## Description
Unvollständiges Wissen ist ein verbreitetes Problem in der Softwareentwicklung. Es tritt auf, wenn Entwickler sich nicht aller Orte bewusst sind, an denen ähnliche Logik existiert. Dies kann zu einer Reihe von Problemen führen, einschließlich Synchronisationsproblemen, Code-Duplizierung und erheblicher Frustration für das Entwicklungsteam. Unvollständiges Wissen ist oft ein Zeichen für ein schlecht dokumentiertes System mit einem hohen Grad an Code-Duplizierung.

## Indicators ⟡
- Das Team erfindet ständig das Rad neu.
- Das Team ist sich nicht aller Features im System bewusst.
- Das Team ist sich nicht sicher, wie sich das System verhalten soll.
- Das Team kann Fragen zum System nicht beantworten.

## Symptoms ▲

- [Synchronisationsprobleme](synchronisationsprobleme.md)
<br/>  Wenn Entwickler nicht alle Orte kennen, an denen ähnliche Logik existiert, verfehlen Aktualisierungen einer Kopie andere, was abweichendes Verhalten verursacht.
- [Inkonsistentes Verhalten](inkonsistentes-verhalten.md)
<br/>  Entwickler, die eine Instanz von Geschäftslogik modifizieren, sind sich anderer Instanzen nicht bewusst, was zu inkonsistentem Systemverhalten führt.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Änderungen, die ohne Bewusstsein aller betroffenen Orte vorgenommen werden, brechen unbeabsichtigt Funktionalität in unbekannten Teilen des Systems.
- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Entwickler, die mit unvollständigem Verständnis des Systems arbeiten, führen mit höherer Wahrscheinlichkeit Defekte ein.

## Causes ▼

- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Veraltete oder fehlende Dokumentation hindert Entwickler daran, alle relevanten Teile des Systems kennenzulernen.
- [Wissenssilos](wissenssilos.md)
<br/>  Wenn Wissen bei einzelnen Teammitgliedern eingeschlossen ist, können andere nichts über Systembereiche lernen, an denen sie nicht persönlich gearbeitet haben.
- [Hohe Fluktuation](hohe-fluktuation.md)
<br/>  Häufige Abgänge von Teammitgliedern führen zum Verlust institutionellen Wissens über Systemstruktur und Logikorte.
- [Code-Duplizierung](code-duplizierung.md)
<br/>  Umfangreiche Code-Duplizierung über die Codebasis macht es für jeden Entwickler inhärent schwierig, alle Orte zu kennen, an denen ähnliche Logik existiert.

## Detection Methods ○
- **Entwicklerbefragungen:** Befragung von Entwicklern zu ihrem Wissen über das System.
- **Code-Reviews:** Code-Reviews sind ein guter Weg, um Wissenslücken zu identifizieren.
- **Pair Programming:** Pair Programming ist ein guter Weg, um Wissen zwischen Entwicklern zu teilen.
- **Wissens-Mapping:** Erstellung einer Wissenslandkarte des Systems, um Bereiche mit Wissenslücken zu identifizieren.

## Examples
Ein Unternehmen hat ein großes, komplexes System. Das System ist nicht gut dokumentiert, und es gibt eine hohe Fluktuationsrate im Team. Infolgedessen hat das Team ein sehr unvollständiges Wissen über das System. Das Team erfindet ständig das Rad neu, und es kann keine Fragen zum System beantworten. Das Unternehmen muss schließlich ein Beraterteam engagieren, um das System zu dokumentieren und das Team zu schulen.
