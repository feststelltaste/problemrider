---
title: Entscheidungsvermeidung
description: Wichtige technische Entscheidungen werden wiederholt aufgeschoben, was
  Fortschritt verhindert und Engpässe in der Entwicklungsarbeit erzeugt.
category:
- Process
- Team
related_problems:
- slug: delayed-decision-making
  similarity: 0.85
- slug: accumulated-decision-debt
  similarity: 0.8
- slug: decision-paralysis
  similarity: 0.75
- slug: analysis-paralysis
  similarity: 0.75
- slug: avoidance-behaviors
  similarity: 0.7
- slug: work-blocking
  similarity: 0.7
solutions:
- architecture-decision-records
- architecture-review-board
- technical-spike
- decision-rights-and-escalation
- psychological-safety-practices
- blameless-postmortems
- team-retrospectives
- prototypes
- team-autonomy-and-empowerment
layout: problem
lang: de
en_slug: decision-avoidance
---

## Description

Entscheidungsvermeidung entsteht, wenn Entwicklungsteams durchgängig das Treffen wichtiger technischer Entscheidungen aufschieben oder verzögern, die für den Fortschritt notwendig sind. Diese Vermeidung kann aus Angst vor falschen Entscheidungen, fehlender klarer Entscheidungsautorität oder übermäßigem Perfektionismus hinsichtlich vollständiger Information entstehen. Das Ergebnis sind Projekte, die stillstehen, während sie auf Entscheidungen warten, angehäufte Entscheidungsschulden, die im Laufe der Zeit schwerer zu lösen werden, und frustrierte Teammitglieder, die mit ihrer Arbeit nicht fortfahren können.

## Indicators ⟡

- Wichtige technische Entscheidungen bleiben wochen- oder monatelang ungetroffen
- Team-Meetings enden häufig, ohne zentrale Entscheidungen zu klären
- Mehrere Alternativen werden kontinuierlich bewertet, ohne dass eine ausgewählt wird
- Entwicklungsarbeit ist blockiert und wartet auf architektonische oder Design-Entscheidungen
- Die Verantwortung für Entscheidungen ist unklar oder wird ständig auf andere verschoben

## Symptoms ▲

- [Angehäufte Entscheidungsschulden](angehaeufte-entscheidungsschulden.md)
<br/>  Jede aufgeschobene Entscheidung trägt zum Rückstau ungetroffener Entscheidungen bei, was kumulative Komplexität erzeugt, die künftige Entscheidungen noch schwieriger macht.
- [Arbeitsblockade](arbeitsblockade.md)
<br/>  Entwicklungsaufgaben, die von ungetroffenen Entscheidungen abhängen, können nicht fortschreiten, was Engpässe im Entwicklungsworkflow erzeugt.
- [Verzögerte Projektzeitpläne](verzoegerte-projektzeitplaene.md)
<br/>  Projekte geraten in Zeitverzug, während die Umsetzungsarbeit stillsteht und auf aufgeschobene architektonische und Design-Entscheidungen wartet.
- [Demoralisierung des Teams](demoralisierung-des-teams.md)
<br/>  Teammitglieder verlieren die Motivation, wenn sie wiederholt nicht mit ihrer Arbeit fortfahren können, weil kritische Entscheidungen ungetroffen bleiben.
- [Stagnierende Architektur](stagnierende-architektur.md)
<br/>  Das Vermeiden architektonischer Entscheidungen verhindert, dass sich das System weiterentwickelt, um sich ändernde Bedürfnisse zu erfüllen, was dazu führt, dass es weiter zurückfällt.

## Causes ▼

- [Schuldzuweisungskultur](schuldzuweisungskultur.md)
<br/>  Wenn Fehler bestraft statt als Lerngelegenheiten behandelt werden, vermeiden Menschen es, Entscheidungen zu treffen, um potenzieller Schuldzuweisung zu entgehen.
- [Analyse-Lähmung](analyse-laehmung.md)
<br/>  Übermäßige Analyse und Recherche ohne zu Schlussfolgerungen zu kommen, verhindert, dass sich Teams auf Entscheidungen festlegen.
- [Mikromanagement-Kultur](mikromanagement-kultur.md)
<br/>  Wenn das Management Genehmigung für Routineentscheidungen verlangt, lernen Teammitglieder, alle Entscheidungen nach oben zu verschieben, statt Eigenverantwortung zu übernehmen.
- [Wissenslücken](wissensluecken.md)
<br/>  Fehlendes ausreichendes Verständnis der technischen Domäne macht Menschen zurückhaltend, sich auf Entscheidungen festzulegen, für die sie sich nicht qualifiziert fühlen.
- [Angst vor Konflikt](angst-vor-konflikt.md)
<br/>  Angst vor Konflikt kann dazu führen, dass Menschen es vermeiden, Entscheidungen zu treffen, die zu Meinungsverschiedenheiten oder Konfrontationen mit Kollegen führen könnten.

## Detection Methods ○

- **Entscheidungsprotokoll-Tracking:** Beobachtung, wie lange wichtige Entscheidungen ungelöst bleiben
- **Meeting-Ergebnis-Analyse:** Nachverfolgung, welcher Prozentsatz entscheidungsfokussierter Meetings zu tatsächlichen Entscheidungen führt
- **Analyse blockierter Arbeit:** Messung, wie viel Entwicklungsarbeit blockiert ist und auf Entscheidungen wartet
- **Entscheidungsqualitätsbewertung:** Bewertung der Auswirkung und Wirksamkeit von Entscheidungen, die letztlich getroffen werden
- **Team-Umfragen:** Befragung zu Frustration mit Entscheidungsprozessen und Engpässen

## Examples

Ein Entwicklungsteam verbringt drei Monate damit, zu debattieren, ob es Microservices oder einen modularen Monolithen für seine neue Anwendung nutzen soll. Mehrere Proof-of-Concepts werden gebaut, umfangreiche Dokumentation wird erstellt, die die Ansätze vergleicht, und wöchentliche Meetings werden abgehalten, um die Entscheidung zu diskutieren, aber es wird keine Wahl getroffen, weil das Team "absolut sicher" sein will, die richtige Wahl zu treffen. In der Zwischenzeit kann die Feature-Entwicklung ohne die architektonische Grundlage nicht fortschreiten, was dazu führt, dass das Projekt Monate in Zeitverzug gerät. Ein weiteres Beispiel betrifft ein Team, das sich sechs Wochen lang nicht auf ein Frontend-Framework entscheiden kann, kontinuierlich neue Optionen recherchiert und sich Sorgen macht, eine Wahl zu treffen, die möglicherweise veraltet, während die Entwicklung der Benutzeroberfläche vollständig blockiert bleibt und die Frustration der Stakeholder wächst.
