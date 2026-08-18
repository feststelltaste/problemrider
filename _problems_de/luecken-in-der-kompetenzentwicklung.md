---
title: Lücken in der Kompetenzentwicklung
description: Teammitglieder entwickeln aufgrund von Vermeidung, Spezialisierung oder
  unzureichenden Lernmöglichkeiten keine Expertise in wichtigen Technologien oder Domänen.
category:
- Team
related_problems:
- slug: knowledge-gaps
  similarity: 0.8
- slug: inappropriate-skillset
  similarity: 0.7
- slug: legacy-skill-shortage
  similarity: 0.7
- slug: knowledge-silos
  similarity: 0.65
- slug: inexperienced-developers
  similarity: 0.65
- slug: knowledge-dependency
  similarity: 0.65
solutions:
- pair-and-mob-programming
- structured-onboarding-program
- refactoring-katas
- security-training
- code-reading-sessions
- internal-technical-coaching
- communities-of-practice
- technical-skills-development
- cross-functional-skill-development
- knowledge-rotation
layout: problem
lang: de
en_slug: skill-development-gaps
---

## Description

Lücken in der Kompetenzentwicklung treten auf, wenn Teammitglieder es versäumen, notwendige Expertise in wichtigen Technologien, Geschäftsdomänen oder Methodologien zu entwickeln, die für den Erfolg der Organisation kritisch sind. Dies kann aus bewusster Vermeidung schwieriger Themen, übermäßiger Spezialisierung auf enge Bereiche, mangelnden Lernmöglichkeiten oder dem Fehlen strukturierter Kompetenzentwicklungsprogramme resultieren. Diese Lücken schaffen Verwundbarkeiten, wenn Expertise benötigt wird, und beschränken die Fähigkeit des Teams, sich an sich ändernde Anforderungen anzupassen.

## Indicators ⟡

- Teammitglieder vermeiden es, mit bestimmten Technologien oder Systemen zu arbeiten
- Fähigkeiten bleiben bei wenigen Spezialisten konzentriert, während andere keinen Kontakt damit haben
- Neue Technologien oder Methodologien werden ohne angemessenes Team-Training übernommen
- Teammitglieder äußern Unbehagen oder Angst bezüglich bestimmter technischer Bereiche
- Wissenstransfer-Sitzungen sind selten oder ineffektiv

## Symptoms ▲

- [Single Points of Failure](single-points-of-failure.md)
<br/>  Wenn nur wenige Teammitglieder bestimmte Fähigkeiten entwickeln, werden sie zu den alleinigen Experten und Single Points of Failure für diese Bereiche.
- [Wissenssilos](wissenssilos.md)
<br/>  Kompetenzlücken verursachen, dass Wissen bei Spezialisten siloartig bleibt, statt im Team verteilt zu werden.
- [Langsamer Wissenstransfer](langsamer-wissenstransfer.md)
<br/>  Wenn Fähigkeiten nicht breit entwickelt werden, gibt es weniger Personen, die effektiv mentorieren und Wissen weitergeben können.
- [Mangel an Legacy-Fachkräften](mangel-an-legacy-fachkraeften.md)
<br/>  Die Vermeidung, Legacy-Technologien zu lernen, schafft einen Mangel an Personen, die ältere Systeme warten und weiterentwickeln können.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Das Fehlen von Trainingsprogrammen und Mentoring-Möglichkeiten verhindert, dass Junior-Entwickler ihre Fähigkeiten entwickeln.

## Causes ▼

- [Zeitdruck](zeitdruck.md)
<br/>  Konstanter Lieferdruck lässt keine Zeit für Lern- und Kompetenzentwicklungsaktivitäten.
- [Widerstand gegen Veränderung](widerstand-gegen-veraenderung.md)
<br/>  Teammitglieder widersetzen sich dem Lernen neuer Technologien oder Ansätze und bleiben lieber in ihrer Komfortzone.

## Detection Methods ○

- **Fähigkeitsbewertungsmatrix:** Regelmäßige Bewertung der Fähigkeiten von Teammitgliedern über verschiedene Bereiche hinweg
- **Verfolgung von Lernzielen:** Überwachung des Fortschritts bei individuellen und Team-Kompetenzentwicklungszielen
- **Technologieübernahmemuster:** Analyse, welche Technologien vom Team vermieden versus übernommen werden
- **Wissensverteilungsanalyse:** Bewertung, wie gleichmäßig Expertise über Teammitglieder verteilt ist
- **Trainingsteilnahmemetriken:** Verfolgung des Engagements bei Lernmöglichkeiten und beruflicher Weiterentwicklung

## Examples

Ein Entwicklungsteam arbeitet primär mit Java und relationalen Datenbanken, aber ihre Anwendungen müssen zunehmend mit modernen Cloud-Diensten und NoSQL-Datenbanken integriert werden. Die meisten Teammitglieder vermeiden es jedoch, Cloud-Technologien zu lernen, weil sie komplex und anders als vertraute On-Premises-Systeme erscheinen. Über zwei Jahre wird der Mangel an Cloud-Expertise des Teams zu einer erheblichen Einschränkung, da Geschäftsanforderungen zunehmend Cloud-native Lösungen verlangen. Neue Projekte werden entweder verzögert, während externe Berater hinzugezogen werden, oder sie werden schlecht mit veralteten Mustern implementiert, die die Cloud-Fähigkeiten nicht nutzen. Ein weiteres Beispiel betrifft ein Team, bei dem Frontend-Entwicklungsfähigkeiten bei einem Senior-Entwickler konzentriert sind, der die gesamte UI-Arbeit erledigt, während andere Teammitglieder sich ausschließlich auf Backend-Services konzentrieren. Wenn der Senior-Entwickler das Unternehmen verlässt, sieht sich das Team einer Krise gegenüber, weil niemand sonst die Benutzeroberfläche warten oder erweitern kann, was sie zwingt, entweder externe Auftragnehmer einzustellen oder die Feature-Entwicklung erheblich zu verzögern, während Teammitglieder sich abmühen, Frontend-Technologien zu lernen, die sie jahrelang vermieden haben.
