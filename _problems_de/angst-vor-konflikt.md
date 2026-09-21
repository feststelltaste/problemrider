---
title: Angst vor Konflikt
description: Reviewer vermeiden es, komplexe Logik oder Designentscheidungen zu
  hinterfragen, und geben stattdessen einfacheres, weniger konfrontatives Feedback.
category:
- Communication
- Process
related_problems:
- slug: reviewer-anxiety
  similarity: 0.75
- slug: author-frustration
  similarity: 0.75
- slug: conflicting-reviewer-opinions
  similarity: 0.75
- slug: review-process-avoidance
  similarity: 0.7
- slug: fear-of-change
  similarity: 0.7
- slug: nitpicking-culture
  similarity: 0.7
solutions:
- psychological-safety-practices
- team-working-agreements
- blameless-postmortems
- code-review-guidelines
- decision-rights-and-escalation
- structured-communication-protocols
- team-retrospectives
- written-first-communication
- communities-of-practice
- collaborative-problem-solving
layout: problem
lang: de
en_slug: fear-of-conflict
---

## Description
Angst vor Konflikt in Code-Reviews ist die Zurückhaltung von Reviewern, kritisches Feedback zu geben, aus Angst, Kollegen zu beleidigen oder Spannungen im Team zu erzeugen. Diese Vermeidung schwieriger Gespräche führt zu einer Kultur, in der Höflichkeit über Qualität priorisiert wird, und erhebliche Probleme im Code bleiben unadressiert. Sie untergräbt den Zweck von Code-Reviews und macht sie zu einer Formalität statt einer echten Praxis der Qualitätssicherung und des Wissensaustauschs.

## Indicators ⟡
- Code-Reviews werden durchgängig mit wenig bis keiner Diskussion genehmigt, selbst bei komplexen Änderungen.
- Reviewer nutzen vage oder übermäßig positive Sprache und vermeiden direkte Kritik.
- Teammitglieder äußern Bedenken zur Codequalität privat, aber nicht in öffentlichen Code-Reviews.

## Symptoms ▲

- [Oberflächliche Code-Reviews](oberflaechliche-code-reviews.md)
<br/>  Reviewer konzentrieren sich nur auf oberflächliche Probleme, um Konfrontation zu vermeiden, und übersehen wichtige Design- und Logikprobleme.
- [Unzureichende Code-Reviews](unzureichende-code-reviews.md)
<br/>  Der Review-Prozess versäumt es, kritische Probleme zu identifizieren, weil Reviewer es vermeiden, das schwierige Feedback zu geben, das für Qualitätssicherung nötig ist.
- [Geringere Codequalität](geringere-codequalitaet.md)
<br/>  Wenn erhebliche Probleme in Reviews unhinterfragt bleiben, verschlechtert sich die Codequalität, während fehlerhafte Designs und Implementierungen in die Codebasis gelangen.
- [Hohe Rate an neu eingeführten Fehlern](hohe-rate-an-neu-eingefuehrten-fehlern.md)
<br/>  Unhinterfragte Logikfehler und Designprobleme in Reviews führen dazu, dass mehr Fehler in die Produktion gelangen.
- [Übereilte Genehmigungen](uebereilte-genehmigungen.md)
<br/>  Reviewer genehmigen Pull Requests schnell, um das Unbehagen zu vermeiden, kritisches Feedback zu geben.
- [Reviewer-Angst](reviewer-angst.md)
<br/>  Reviewer, die sich ihrer eigenen Fähigkeiten unsicher sind, vermeiden Konfrontation, weil sie ihre Stellung bezweifeln, andere zu hinterfragen.

## Causes ▼

- [Schuldzuweisungskultur](schuldzuweisungskultur.md)
<br/>  Eine Kultur, die Fehler bestraft, macht Reviewer ängstlich, andere zu hinterfragen, aus Angst, Spannungen oder Vergeltung zu erzeugen.
- [Unzureichende Mentoring-Struktur](unzureichende-mentoring-struktur.md)
<br/>  Ohne ordentliches Mentoring lernen Reviewer nie, wie man konstruktive Kritik wirksam liefert, und greifen standardmäßig auf Vermeidung zurück.

## Detection Methods ○
- **Beobachtung der Code-Review-Dynamik:** Achten auf Ton und Inhalt von Code-Review-Diskussionen. Suche nach fehlendem kritischem Feedback oder einer Tendenz, schwierige Themen zu vermeiden.
- **Team-Umfragen:** Anonyme Befragung von Teammitgliedern zu ihrem Wohlbefinden beim Geben und Empfangen kritischen Feedbacks.
- **Retrospektiven:** Diskussion der Wirksamkeit des Code-Review-Prozesses und ob Teammitglieder das Gefühl haben, offen und ehrlich sein zu können.

## Examples
Ein Senior-Entwickler bemerkt einen erheblichen architektonischen Fehler im Pull Request eines Junior-Entwicklers. Weil er den Junior-Entwickler jedoch nicht entmutigen möchte, genehmigt er den Pull Request mit nur einem kleinen Kommentar zu einem Variablennamen. Der architektonische Fehler wird später entdeckt, nachdem er erhebliche Probleme in der Produktion verursacht hat. Diese Angst vor Konflikt verhindert, dass das Team die notwendigen Gespräche führt, um hochwertige Software zu bauen.
