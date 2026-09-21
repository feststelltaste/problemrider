---
title: Vermeidung des Review-Prozesses
description: Teammitglieder suchen aktiv nach Wegen, Code-Review-Anforderungen zu umgehen
  oder zu minimieren, was den Qualitätssicherungsprozess untergräbt.
category:
- Process
- Team
- Testing
related_problems:
- slug: review-process-breakdown
  similarity: 0.75
- slug: reduced-review-participation
  similarity: 0.75
- slug: team-members-not-engaged-in-review-process
  similarity: 0.7
- slug: inadequate-code-reviews
  similarity: 0.7
- slug: insufficient-code-review
  similarity: 0.7
- slug: reviewer-anxiety
  similarity: 0.7
solutions:
- code-review-process-reform
- code-review-guidelines
- psychological-safety-practices
- small-change-batches
- team-working-agreements
- code-quality-gates
- checklists
- pair-and-mob-programming
- team-retrospectives
- fast-feedback-loops
layout: problem
lang: de
en_slug: review-process-avoidance
---

## Description

Vermeidung des Review-Prozesses tritt auf, wenn Teammitglieder aktiv nach Wegen suchen, Code-Review-Anforderungen zu umgehen, zu minimieren oder zu unterlaufen, aufgrund von Frustration mit dem Review-Prozess selbst. Dies kann Änderungen direkt in Produktion, die Nutzung von Notfall-Deployment-Prozeduren für nicht-dringende Änderungen, direkte Commits auf Hauptbranches oder das Finden technischer Schlupflöcher zur Vermeidung von Reviews umfassen. Dieses Verhalten untergräbt die Qualitätssicherungsvorteile, die Code-Reviews eigentlich bieten sollen.

## Indicators ⟡

- Zunehmende Nutzung von „Hotfix"- oder Notfall-Deployment-Prozeduren für unkritische Änderungen
- Direkte Commits auf Hauptbranches, die Review-Anforderungen umgehen
- Änderungen, die außerhalb der Arbeitszeiten vorgenommen werden, um Review-Aufsicht zu vermeiden
- Häufige Anfragen, Review-Anforderungen zu ändern oder Ausnahmen zu machen
- Teammitglieder äußern den Wunsch, „das Review diesmal einfach zu überspringen"

## Symptoms ▲

- [Hohe Fehlerrate in Produktion](hohe-fehlerrate-in-produktion.md)
<br/>  Änderungen, die Reviews umgehen, verpassen das Qualitätstor, wodurch mehr Fehler unentdeckt in Produktion gelangen.
- [Zusammenbruch des Review-Prozesses](zusammenbruch-des-review-prozesses.md)
<br/>  Weit verbreitete Vermeidung untergräbt den Review-Prozess systematisch, sodass er an seinem Qualitätssicherungszweck scheitert.
- [Inkonsistente Coding-Standards](inkonsistente-coding-standards.md)
<br/>  Code, der Reviews überspringt, wird nicht auf Standardkonformität geprüft, was zu inkonsistenten Coding-Praktiken in der Codebasis führt.
- [Wissenssilos](wissenssilos.md)
<br/>  Das Umgehen von Reviews beseitigt einen wichtigen Mechanismus des Wissensaustauschs, sodass Codewissen beim ursprünglichen Autor siloartig bleibt.

## Causes ▼

- [Review-Engpässe](review-engpaesse.md)
<br/>  Wenn der Review-Prozess ein erheblicher Engpass ist, sind Entwickler motiviert, Wege darum herum zu finden, um ihre Arbeit auszuliefern.
- [Frustration der Autoren](frustration-der-autoren.md)
<br/>  Frustration über widersprüchliches oder scheinbar willkürliches Review-Feedback treibt Entwickler dazu, den Prozess ganz zu vermeiden.
- [Zeitdruck](zeitdruck.md)
<br/>  Fristendruck lässt den Review-Prozess wie eine unerschwingliche Verzögerung erscheinen, was Entwickler motiviert, ihn zu umgehen.
- [Reviewer-Angst](reviewer-angst.md)
<br/>  Wenn Reviewer ängstlich sind und oberflächliches oder unhilfreiches Feedback geben, sehen Autoren wenig Wert im Review-Prozess und vermeiden ihn.

## Detection Methods ○

- **Verfolgung von Review-Umgehungen:** Überwachung von Commits, Deployments oder Änderungen, die normale Review-Prozesse umgehen
- **Analyse der Notfallprozedur-Nutzung:** Verfolgung von Häufigkeit und Begründung der Nutzung von Notfall-Deployments
- **Bewertung der Prozess-Compliance:** Messung des Anteils von Änderungen, die tatsächlich das erforderliche Review durchlaufen
- **Team-Verhaltensbefragungen:** Sammlung von Feedback zu Motivationen für die Vermeidung von Review-Prozessen
- **Korrelation der Qualitätsauswirkung:** Analyse, ob umgangene Änderungen höhere Fehlerraten aufweisen

## Examples

Ein Entwickler wird frustriert, nachdem er drei Wochen damit verbracht hat, einen einfachen Bugfix aufgrund umfangreicher Stildebatten und widersprüchlichen Feedbacks durch den Review-Prozess zu bekommen. Als der nächste dringende Bug auftritt, deployt er den Fix über den Notfall-Hotfix-Prozess, um das Review zu vermeiden, obwohl das Problem gar nicht wirklich kritisch ist. Dies schafft einen Präzedenzfall, und bald nutzen mehrere Teammitglieder Notfallprozeduren aus Bequemlichkeit statt für echte Notfälle. Ein weiteres Beispiel betrifft ein Teammitglied, das entdeckt, dass es kleine Änderungen direkt in der Deployment-Konfiguration vornehmen kann, die Code-Review-Anforderungen umgeht. Es beginnt, zunehmend bedeutsamere Änderungen über diesen Weg vorzunehmen, einschließlich Geschäftslogik-Modifikationen, die eigentlich ein Review erhalten sollten, weil es Zeit und Frustration des normalen Review-Prozesses vermeiden möchte.
