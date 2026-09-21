---
title: Bequemlichkeitsgetriebene Entwicklung
description: Eine Entwicklungspraxis, bei der Entwickler die einfachste und bequemste
  Lösung wählen, statt der besten Lösung.
category:
- Code
- Process
related_problems:
- slug: cv-driven-development
  similarity: 0.65
- slug: increased-technical-shortcuts
  similarity: 0.6
- slug: assumption-based-development
  similarity: 0.55
- slug: copy-paste-programming
  similarity: 0.55
- slug: feature-creep-without-refactoring
  similarity: 0.55
- slug: defensive-coding-practices
  similarity: 0.5
solutions:
- architecture-reviews
- clean-code
- separation-of-concerns
- solid-principles
- architecture-governance
- architecture-review-board
- lightweight-design-review
- code-review-guidelines
- code-quality-gates
- preparatory-refactoring
- communities-of-practice
- debt-accrual-analysis
- quality-ratchet
layout: problem
lang: de
en_slug: convenience-driven-development
---

## Description
Bequemlichkeitsgetriebene Entwicklung ist eine Entwicklungspraxis, bei der Entwickler die einfachste und bequemste Lösung wählen, statt der besten Lösung. Dies führt oft zu einer schrittweisen Verschlechterung der Codebasis, da Entwickler Abkürzungen nehmen und Design-Entscheidungen treffen, die nicht im besten langfristigen Interesse des Projekts sind. Bequemlichkeitsgetriebene Entwicklung ist oft ein Zeichen für mangelnde Erfahrung oder mangelnde Disziplin seitens des Entwicklungsteams.

## Indicators ⟡
- Die Codebasis ist voller Hacks und Workarounds.
- Das Design der Codebasis ist inkonsistent.
- Es gibt viel duplizierten Code.
- Die Codebasis ist schwer zu verstehen und zu warten.

## Symptoms ▲

- [Code-Duplizierung](code-duplizierung.md)
<br/>  Den bequemen Weg zu gehen bedeutet oft, bestehenden Code zu kopieren, statt Zeit in die Erstellung wiederverwendbarer Abstraktionen zu investieren.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Durchgängig die einfachste Lösung statt der besten zu wählen, häuft Design-Abkürzungen an, die zu technischen Schulden werden.
- [Inkonsistente Codebasis](inkonsistente-codebasis.md)
<br/>  Wenn jeder Entwickler seine eigenen bequemen Abkürzungen nimmt, entwickelt die Codebasis inkonsistente Muster und Design-Ansätze.
- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Bequeme Abkürzungen wie schlechte Benennung, fehlende Abstraktionen und Ad-hoc-Lösungen machen die Codebasis im Laufe der Zeit schwerer verständlich.
- [Zunehmende Brüchigkeit](zunehmende-bruechigkeit.md)
<br/>  Abkürzungen und Schnelllösungen machen die Codebasis zunehmend brüchig, da sie ordentliche Design-Prinzipien umgehen.

## Causes ▼

- [Zeitdruck](zeitdruck.md)
<br/>  Der Druck, schnell zu liefern, drängt Entwickler zur schnellsten Lösung statt zur am besten entworfenen.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler ohne Erfahrung erkennen möglicherweise nicht, dass die bequeme Lösung langfristige Probleme schafft, und greifen standardmäßig auf das zurück, was sie kennen.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Wenn das Management sofortige Lieferung über langfristige Codegesundheit stellt, werden Entwickler dazu angereizt, bequeme Abkürzungen zu nehmen.
- [Termindruck](termindruck.md)
<br/>  Intensiver Termindruck lässt Entwicklern keine Zeit, ordentliche Lösungen zu verfolgen, was Bequemlichkeit zur einzigen praktikablen Option macht.

## Detection Methods ○
- **Code-Reviews:** Suche nach Code, der schlecht entworfen und schwer verständlich ist.
- **Statische Analysewerkzeuge:** Nutzung von Werkzeugen zur Identifikation von Code Smells wie duplizierter Code und großen Klassen.
- **Entwickler-Umfragen:** Befragung von Entwicklern, ob sie das Gefühl haben, hochwertigen Code schreiben zu können.

## Examples
Ein Entwickler muss ein neues Feature zu einem Legacy-System hinzufügen. Der Entwickler steht unter Druck, das Feature so schnell wie möglich zu liefern. Der Entwickler beschließt, einen großen Codeblock aus einem anderen Teil des Systems zu kopieren, statt sich die Zeit zu nehmen, den Code zu refaktorieren und eine neue, wiederverwendbare Komponente zu erstellen. Dies spart dem Entwickler kurzfristig ein paar Stunden Arbeit, erhöht aber die technischen Schulden des Systems und erschwert dessen langfristige Wartung.
