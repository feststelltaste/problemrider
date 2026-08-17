---
title: Doppelter Recherche-Aufwand
description: Mehrere Teammitglieder recherchieren unabhängig voneinander dieselben
  Themen, was Zeit verschwendet und den Aufbau von kollektivem Wissen verhindert.
category:
- Communication
- Process
- Team
related_problems:
- slug: duplicated-effort
  similarity: 0.85
- slug: duplicated-work
  similarity: 0.85
- slug: extended-research-time
  similarity: 0.7
- slug: knowledge-silos
  similarity: 0.7
- slug: team-silos
  similarity: 0.65
- slug: knowledge-gaps
  similarity: 0.65
solutions:
- knowledge-sharing-practices
- knowledge-rotation
- knowledge-base
- documentation-as-code
- architecture-decision-records
- living-documentation
- communities-of-practice
- written-first-communication
layout: problem
lang: de
en_slug: duplicated-research-effort
---

## Description

Doppelter Recherche-Aufwand entsteht, wenn mehrere Teammitglieder unabhängig voneinander dieselben Themen, Technologien oder Problemdomänen untersuchen, ohne ihre Erkenntnisse zu teilen oder ihre Rechercheaktivitäten zu koordinieren. Diese Duplizierung verschwendet wertvolle Entwicklungszeit und versäumt es, institutionelles Wissen aufzubauen, das dem gesamten Team zugutekommen könnte. Das Problem entspringt oft schlechter Kommunikation, fehlenden Wissensmanagementsystemen oder unklarer Koordination der Rechercheverantwortlichkeiten.

## Indicators ⟡

- Teammitglieder stellen zu unterschiedlichen Zeiten ähnliche Rechercheanfragen
- Mehrere Entwickler recherchieren unabhängig voneinander dieselben Technologien oder Ansätze
- Wiederholte Diskussionen über Themen, die zuvor bereits untersucht wurden
- Ähnliche Dokumentation oder Proof-of-Concept-Code, erstellt von unterschiedlichen Teammitgliedern
- Rechercheergebnisse werden nicht geteilt oder sind anderen Teammitgliedern nicht zugänglich

## Symptoms ▲

- [Verlängerte Rechercheszeit](verlaengerte-recherchezeit.md)
<br/>  Wenn Recherche über Teammitglieder hinweg dupliziert statt geteilt wird, steigt die insgesamt aufgewendete Rechercheszeit dramatisch.
- [Verschwendeter Entwicklungsaufwand](verschwendeter-entwicklungsaufwand.md)
<br/>  Mehrere Personen, die unabhängig voneinander dasselbe Thema recherchieren, stellen direkt verschwendete Entwicklungskapazität dar.
- [Verringerte Teamproduktivität](verringerte-teamproduktivitaet.md)
<br/>  Der Team-Durchsatz sinkt, wenn mehrere Mitglieder Zeit mit Recherche verbringen, die einmal hätte durchgeführt und geteilt werden können.
- [Informationsfragmentierung](informationsfragmentierung.md)
<br/>  Wenn mehrere Personen unabhängig voneinander recherchieren, landen ihre Erkenntnisse verstreut in unterschiedlichen Dokumenten und persönlichen Notizen, statt konsolidiert zu werden.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Der vervielfachte Rechercheaufwand verlangsamt direkt die Fähigkeit des Teams, Features und Fixes zu liefern.

## Causes ▼

- [Zusammenbruch des Wissensaustauschs](zusammenbruch-des-wissensaustauschs.md)
<br/>  Wenn Wissensaustauschprozesse unwirksam sind, werden Rechercheergebnisse nicht verbreitet, was andere dazu bringt, dieselben Untersuchungen zu wiederholen.
- [Wissenssilos](wissenssilos.md)
<br/>  In individuellen Silos gefangene Rechercheexpertise bedeutet, dass andere nicht auf frühere Erkenntnisse zugreifen können und die Recherche wiederholen müssen.
- [Kommunikationszusammenbruch](kommunikationszusammenbruch.md)
<br/>  Schlechte Kommunikation verhindert, dass Teammitglieder wissen, dass andere bereits dieselben Themen recherchiert haben.
- [Team-Silos](team-silos.md)
<br/>  Isolierte Teams duplizieren Recherche naturgemäß, weil ihnen der Einblick fehlt, was andere Teams bereits untersucht haben.
- [Unklare Erwartungen beim Teilen von Informationen](unklare-erwartungen-beim-teilen-von-informationen.md)
<br/>  Wenn nicht klar ist, welche Informationen mit dem Team geteilt werden sollten, bleiben Rechercheergebnisse ungeteilt und werden dupliziert.

## Detection Methods ○

- **Recherchethemen-Tracking:** Beobachtung, welche Themen Teammitglieder recherchieren, zur Identifikation von Überlappungen
- **Fragemuster-Analyse:** Nachverfolgung wiederkehrender Fragen, die auf wiederholte Recherche hindeuten
- **Dokumentations-Review:** Suche nach mehreren Dokumenten oder Codebeispielen, die dieselben Themen behandeln
- **Zeittracking-Analyse:** Vergleich der Rechercheszeit mit der Komplexität der untersuchten Themen
- **Team-Umfragen:** Befragung zu Erfahrungen mit Recherchekoordination und Wissensaustausch

## Examples

Drei unterschiedliche Entwickler verbringen jeweils eine Woche damit, zu recherchieren, wie die Anwendung mit einer bestimmten Drittanbieter-API integriert werden kann, jeder stößt auf dieselben Authentifizierungsherausforderungen und kommt zu ähnlichen Schlussfolgerungen über Implementierungsansätze. Keiner von ihnen kommuniziert seine Rechercheaktivitäten oder teilt seine Erkenntnisse, was zu drei Wochen doppeltem Aufwand führt, der mit ordentlicher Koordination auf eine Woche hätte reduziert werden können. Ein weiteres Beispiel betrifft ein Team, in dem mehrere Mitglieder über mehrere Monate hinweg unabhängig voneinander dieselben Datenbank-Performance-Optimierungstechniken recherchieren, jeder erstellt seine eigenen Testaufbauten und kommt zu ähnlichen Schlussfolgerungen über Abfrageoptimierungsstrategien, teilt aber nie seine Erkenntnisse, was dazu führt, dass jedes neue Performance-Problem denselben RechercheZyklus auslöst.
