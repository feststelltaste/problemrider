---
title: Zeitdruck
description: Teams werden gezwungen, Abkürzungen zu nehmen, um unmittelbare Termine
  einzuhalten, wobei ordentliche Lösungen aufgeschoben und wichtige Aufgaben wie
  Code-Reviews überstürzt werden.
category:
- Code
- Process
related_problems:
- slug: deadline-pressure
  similarity: 0.85
- slug: increased-technical-shortcuts
  similarity: 0.75
- slug: unrealistic-deadlines
  similarity: 0.75
- slug: high-technical-debt
  similarity: 0.7
- slug: increased-stress-and-burnout
  similarity: 0.7
- slug: slow-development-velocity
  similarity: 0.65
solutions:
- iterative-development
- short-iteration-cycles
- capacity-based-planning
- improvement-budget
- explicit-prioritization-framework
- work-in-progress-limits
- definition-of-ready
- team-retrospectives
- sustainable-pace-practices
layout: problem
lang: de
en_slug: time-pressure
---

## Description
Zeitdruck ist ein durchdringendes Problem in der Softwareentwicklung, bei dem die Betonung von Geschwindigkeit und dem Einhalten von Terminen zu Kompromissen bei der Qualität führt. Wenn Teams konstant unter Druck stehen, ist es wahrscheinlicher, dass sie Abkürzungen nehmen, wichtige Schritte wie Testen und gründliche Code-Reviews überspringen und suboptimale Designentscheidungen treffen. Dies kann zu einer Anhäufung technischer Schulden, einem Rückgang der Codequalität und einer Zunahme der Anzahl von Bugs führen.

## Indicators ⟡
- Das Team arbeitet konsequent Überstunden, um Termine einzuhalten.
- Features werden häufig am Ende eines Release-Zyklus reduziert oder überstürzt.
- Es gibt ein allgemeines Gefühl, sich in einem konstanten Zustand des „Feuerlöschens" zu befinden.

## Symptoms ▲

- [Zunehmende technische Abkürzungen](zunehmende-technische-abkuerzungen.md)
<br/>  Unter Zeitdruck nehmen Teams Schnelllösungen und Workarounds statt ordentliche Lösungen zu implementieren.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Abkürzungen, die unter Zeitdruck genommen werden, häufen sich als technische Schulden an, die zunehmend teuer werden, anzugehen.
- [Qualitätskompromisse](qualitaetskompromisse.md)
<br/>  Zeitdruck zwingt Teams, Qualitätsstandards bewusst zu senken, um Termine einzuhalten.
- [Erhöhter Stress und Burnout](erhoehter-stress-und-burnout.md)
<br/>  Anhaltender Zeitdruck führt zu Überarbeitung, Stress und schließlich Burnout bei Teammitgliedern.
- [Geringere Codequalität](geringere-codequalitaet.md)
<br/>  Überstürzte Entwicklung unter Zeitdruck resultiert in schlecht designtem, schwerer wartbarem Code.
- [Testschulden](testschulden.md)
<br/>  Testen ist oft die erste Aktivität, die geopfert wird, wenn Teams unter Zeitdruck stehen, was zu angehäuften Testschulden führt.
- [Oberflächliche Code-Reviews](oberflaechliche-code-reviews.md)
<br/>  Unter Zeitdruck werden Code-Reviews oberflächlich, während Reviewer sie hastig durcharbeiten, um die Lieferung freizugeben, was eine direkte Konsequenz ist.

## Causes ▼

- [Unrealistische Termine](unrealistische-termine.md)
<br/>  Das Management, das Termine setzt, die den tatsächlich erforderlichen Aufwand nicht berücksichtigen, ist ein primärer Treiber von Zeitdruck.
- [Marktdruck](marktdruck.md)
<br/>  Externe Wettbewerbskräfte treiben Organisationen dazu, Teams zu drängen, schneller zu liefern, was Zeitdruck schafft.
- [Sich änderndes Projekt-Scope](sich-aenderndes-projekt-scope.md)
<br/>  Wenn sich der Scope ausdehnt, ohne Zeitpläne anzupassen, muss dieselbe Zeitmenge mehr Arbeit abdecken, was den Zeitdruck verstärkt.
- [Ständiges Feuerlöschen](staendiges-feuerloeschen.md)
<br/>  Wenn Teams den Großteil ihrer Zeit mit dringenden Fixes verbringen, wird geplante Arbeit in weniger Zeit gequetscht, was Zeitdruck für Feature-Lieferung schafft.

## Detection Methods ○
- **Überstunden verfolgen:** Überwachung der Anzahl der Stunden, die das Team über seinen normalen Zeitplan hinaus arbeitet.
- **Bug-Berichte analysieren:** Suche nach einer Zunahme der Anzahl von Bugs, besonders solcher, die mit mehr Zeit für Testen und Review hätten verhindert werden können.
- **Team-Retrospektiven:** Diskussion der Auswirkung von Terminen auf die Fähigkeit des Teams, hochqualitative Arbeit zu produzieren.

## Examples
Ein Team steht unter Druck, ein neues Feature bis Ende des Quartals zu liefern. Um den Termin einzuhalten, entscheiden sie sich, das Schreiben von Unit-Tests zu überspringen und nur einen oberflächlichen manuellen Test durchzuführen. Das Feature wird pünktlich geliefert, ist aber voller Bugs, die erst von Nutzern in Produktion entdeckt werden. Das Team muss dann die nächsten mehreren Wochen damit verbringen, die Bugs zu beheben, was letztlich das nächste Feature-Release verzögert.
