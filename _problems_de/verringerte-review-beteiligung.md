---
title: Verringerte Review-Beteiligung
description: Viele Teammitglieder vermeiden die Teilnahme an Code-Reviews, was die
  Review-Last auf wenige Personen konzentriert und die Abdeckung verringert.
category:
- Process
- Team
related_problems:
- slug: team-members-not-engaged-in-review-process
  similarity: 0.85
- slug: reviewer-anxiety
  similarity: 0.75
- slug: review-process-avoidance
  similarity: 0.75
- slug: reduced-code-submission-frequency
  similarity: 0.75
- slug: inadequate-code-reviews
  similarity: 0.75
- slug: code-review-inefficiency
  similarity: 0.7
solutions:
- code-review-process-reform
- code-review-guidelines
- small-change-batches
- team-working-agreements
- pair-and-mob-programming
- work-in-progress-limits
- psychological-safety-practices
- team-retrospectives
- fast-feedback-loops
- communities-of-practice
layout: problem
lang: de
en_slug: reduced-review-participation
---

## Description

Verringerte Review-Beteiligung tritt auf, wenn viele Teammitglieder ihre Beteiligung am Code-Review-Prozess vermeiden oder minimieren, sodass die meisten Reviews von einer kleinen Untergruppe des Teams gehandhabt werden. Dies schafft eine ungleiche Verteilung der Review-Arbeitslast, verringert die Vielfalt der Perspektiven auf Codeänderungen und kann zu Review-Engpässen führen, wenn die aktiven Reviewer überwältigt oder nicht verfügbar werden.

## Indicators ⟡

- Nur 2-3 Teammitglieder von 8-10 nehmen regelmäßig an Code-Reviews teil
- Dieselben Personen werden konsequent für Reviews zugewiesen oder melden sich freiwillig
- Junior-Entwickler reviewen selten den Code von Senior-Entwicklern
- Manche Teammitglieder gehen Wochen ohne Durchführung irgendeines Reviews
- Review-Zuweisungen werden von bestimmten Teammitgliedern abgelehnt oder ignoriert

## Symptoms ▲

- [Review-Engpässe](review-engpaesse.md)
<br/>  Die Konzentration der Review-Arbeit auf wenige Personen schafft Engpässe, wenn diese Reviewer nicht verfügbar oder überlastet sind.
- [Unzureichende Code-Reviews](unzureichende-code-reviews.md)
<br/>  Weniger Reviewer bedeutet weniger vielfältige Perspektiven und verringerte Gründlichkeit beim Erfassen von Problemen.
- [Wissenssilos](wissenssilos.md)
<br/>  Nicht teilnehmende Teammitglieder verpassen die Exposition zu Codeänderungen, was Wissensisolation verstärkt.
- [Verringerte Häufigkeit von Code-Einreichungen](verringerte-haeufigkeit-von-code-einreichungen.md)
<br/>  Wenn wenige Reviewer verfügbar sind, verzögern Entwickler Einreichungen, um lange Review-Wartezeiten zu vermeiden.

## Causes ▼

- [Reviewer-Angst](reviewer-angst.md)
<br/>  Angst, falsches Feedback zu geben oder als unqualifiziert wahrgenommen zu werden, entmutigt Teammitglieder davon, an Reviews teilzunehmen.
- [Unerfahrenheit der Reviewer](unerfahrenheit-der-reviewer.md)
<br/>  Fehlende Review-Fähigkeiten und Vertrauen führt dazu, dass Junior- oder Mid-Level-Entwickler sich gegen das Reviewen von Code entscheiden.
- [Vermeidung des Review-Prozesses](vermeidung-des-review-prozesses.md)
<br/>  Allgemeine Abneigung gegen den Review-Prozess führt dazu, dass Teammitglieder sowohl das Einreichen als auch das Reviewen von Code vermeiden.
- [Teammitglieder nicht in den Review-Prozess eingebunden](teammitglieder-nicht-in-den-review-prozess-eingebunden.md)
<br/>  Desengagement vom Review-Prozess als kultureller Standard verringert die Gesamtbeteiligungsraten.

## Detection Methods ○

- **Review-Beteiligungsverfolgung:** Überwachung, wie viele Teammitglieder über die Zeit aktiv an Reviews teilnehmen
- **Analyse der Review-Arbeitslastverteilung:** Messung, wie Review-Verantwortlichkeiten über Teammitglieder verteilt werden
- **Befragungen zu Beteiligungsbarrieren:** Sammlung von Feedback dazu, warum Teammitglieder das Reviewen von Code vermeiden
- **Akzeptanzraten von Review-Zuweisungen:** Nachverfolgung, wie oft Review-Anfragen akzeptiert versus abgelehnt werden
- **Bewertung der Auswirkung auf Kompetenzentwicklung:** Bewertung von Lernresultaten für teilnehmende versus nicht teilnehmende Mitglieder

## Examples

Ein 10-köpfiges Entwicklungsteam hat nur 3 Senior-Entwickler, die 90 % aller Code-Reviews handhaben, während 7 andere Teammitglieder selten am Review-Prozess teilnehmen. Wenn einer der aktiven Reviewer in den Urlaub geht, werden die verbleibenden zwei überwältigt, und die Review-Qualität leidet. Die nicht teilnehmenden Mitglieder verpassen wertvolle Lernmöglichkeiten und bleiben unwissend über Coding-Muster und Design-Entscheidungen, die über die Codebasis hinweg getroffen werden. Ein weiteres Beispiel betrifft ein Team, in dem Junior-Entwickler das Gefühl haben, nicht qualifiziert zu sein, jemandes Code zu reviewen, Mid-Level-Entwickler nur andere Mid-Level-Arbeit reviewen, und Senior-Entwickler alles reviewen. Dies schafft eine Hierarchie, in der die meiste Code nur eine Perspektive erhält statt der vielfältigen Sichtweisen, die Reviews wertvoll machen, und Junior-Entwickler entwickeln keine kritischen Code-Analyse-Fähigkeiten.
