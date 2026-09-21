---
title: Usability-Tests
description: Durchführung von Tests mit repräsentativen Nutzern.
category:
- Testing
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/usability-tests/
problems:
- poor-user-experience-ux-design
- user-frustration
- user-confusion
- negative-user-feedback
- customer-dissatisfaction
- feature-gaps
- shadow-systems
- stakeholder-developer-communication-gap
- misaligned-deliverables
- negative-brand-perception
- stakeholder-dissatisfaction
- declining-business-metrics
- user-trust-erosion
layout: solution
lang: de
en_slug: usability-tests
related_solutions:
- slug: user-centered-design
  similarity: 0.8
- slug: intuitive-navigation
  similarity: 0.75
- slug: user-acceptance-tests
  similarity: 0.75
- slug: exploratory-testing
  similarity: 0.75
- slug: prototyping
  similarity: 0.75
- slug: prototypes
  similarity: 0.75
---

## Description

Ein Usability-Test beobachtet echte, repräsentative Nutzer, die genuine Aufgaben mit dem System versuchen, unter Nutzung eines Think-aloud-Protokolls, das nicht nur offenbart, wo sie kämpfen, sondern warum — Evidenz, die grundlegend anders und zuverlässiger ist als die eigene Intuition eines Entwicklers über die Schnittstelle. Genau diese Intuition versagt in einem Legacy-System: Die Personen, die es bauen und pflegen, haben sich über Jahre an seine Eigenheiten angepasst und können die Verwirrung, die ein neuer oder gelegentlicher Nutzer innerhalb der ersten fünf Minuten erlebt, nicht mehr sehen. Fünf bis acht Teilnehmer pro Sitzung reichen aus, um ungefähr achtzig Prozent der Usability-Probleme zutage zu bringen, und dies periodisch statt einmalig durchzuführen deckt oft Schattensysteme und Workarounds auf, von denen das Team nie wusste, dass sie existierten — Evidenz von Lücken, die die Schnittstelle selbst nie von sich aus offenbart hätte.

## How to Apply ◆

> Legacy-Systeme werden selten mit echten Nutzern getestet, sodass sich Usability-Probleme jahrelang unentdeckt ansammeln. Systematisches Usability-Testing offenbart Probleme, die Entwickler und Product Owner nicht sehen können, weil sie mit dem System zu vertraut sind.

- Rekrutieren Sie fünf bis acht repräsentative Nutzer pro Testsitzung. Forschung zeigt, dass fünf Nutzer ungefähr achtzig Prozent der Usability-Probleme aufdecken. Wählen Sie Teilnehmer, die die tatsächliche Nutzerbasis in Bezug auf Rolle, Erfahrungsstufe und technischen Komfort repräsentieren.
- Gestalten Sie aufgabenbasierte Testszenarien, die echte Workflows widerspiegeln, statt künstlicher Übungen. Bitten Sie Nutzer, Aufgaben zu erledigen, die sie normalerweise ausführen würden, wie einen Bericht einreichen, einen Auftrag verarbeiten oder einen Kundendatensatz nachschlagen.
- Nutzen Sie das Think-aloud-Protokoll, bei dem Teilnehmer ihren Gedankenprozess während der Arbeit verbalisieren. Dies offenbart nicht nur, was Nutzer tun, sondern warum sie es tun und wo sie verwirrt werden.
- Zeichnen Sie Sitzungen mit Bildschirmaufnahme und Audio auf, sodass das Team Beobachtungen nach dem Test überprüfen kann. Direkte Beobachtung während der Sitzung erfasst unmittelbare Reaktionen, aber Aufzeichnungen offenbaren Details, die leicht zu übersehen sind.
- Analysieren Sie Ergebnisse nach Schweregrad und Häufigkeit. Priorisieren Sie Probleme, die Aufgabenfehlschlag oder erhebliche Verzögerung verursachen, gegenüber kosmetischen oder Präferenzproblemen.
- Führen Sie Usability-Tests in regelmäßigen Abständen durch, nicht nur einmal. Führen Sie Tests vor und nach größeren Schnittstellenänderungen durch, um Verbesserung zu messen und Regressionen zu erfassen.

## Tradeoffs ⇄

> Usability-Testing bietet die direkteste Evidenz von Nutzererfahrungsproblemen, erfordert aber Zeit, Planung und Zugang zu repräsentativen Nutzern.

**Vorteile:**

- Offenbart Usability-Probleme, die für das Entwicklungsteam unsichtbar sind, weil es sich über die Zeit an die Eigenheiten des Systems angepasst hat.
- Bietet konkrete, beobachtbare Evidenz für die Priorisierung von UX-Verbesserungen, was es leichter macht, Investition in Usability-Arbeit zu rechtfertigen.
- Identifiziert Schattensysteme und Workarounds, die Nutzer entwickelt haben, und deckt versteckte Anforderungen und Feature-Lücken auf.
- Validiert, dass vorgeschlagene Verbesserungen Nutzern tatsächlich helfen, statt neue Probleme einzuführen, und verhindert verschwendeten Entwicklungsaufwand.

**Kosten und Risiken:**

- Die Rekrutierung repräsentativer Nutzer braucht Zeit, und aus ihrer regulären Arbeit gezogene Nutzer könnten die Unterbrechung übelnehmen, wenn der Prozess nicht gut organisiert ist.
- Usability-Tests produzieren qualitative Daten, die geschickte Interpretation erfordern. Unerfahrene Beobachter könnten sich auf subjektive Präferenzen statt genuine Usability-Probleme konzentrieren.
- Das Testen eines Legacy-Systems mit schweren Usability-Problemen kann eine überwältigende Anzahl von Befunden produzieren, was Disziplin zur Priorisierung erfordert, statt zu versuchen, alles auf einmal zu beheben.
- Es besteht das Risiko, sich zu sehr auf das Verhalten einer kleinen Anzahl von Testteilnehmern zu verlassen, was es wichtig macht, zwischen individuellen Präferenzen und genuinen Designproblemen zu unterscheiden.

## How It Could Be

> Organisationen, die nie Usability-Tests an ihren Legacy-Systemen durchgeführt haben, sind konsistent überrascht von dem, was sie entdecken.

Ein Legacy-Personalwesen-System ist seit acht Jahren im Einsatz, und das Entwicklungsteam glaubt, die Schnittstelle sei angemessen, weil das Support-Ticket-Volumen handhabbar ist. Ein Usability-Test mit sechs HR-Generalisten offenbart ein anderes Bild: Jeder Teilnehmer kämpft mit denselben drei Workflow-Schritten, alle sechs entwickeln unterschiedliche Workarounds für dasselbe Navigationsproblem, und zwei Teilnehmer scheitern daran, eine Routineaufgabe innerhalb des Zeitlimits abzuschließen, weil sie den korrekten Bildschirm nicht finden können. Der Test offenbart auch, dass Nutzer ein inoffizielles Wiki mit annotierten Screenshots erstellt haben, das erklärt, wie gängige Aufgaben durchzuführen sind — ein Schattendokumentationssystem, von dem das Entwicklungsteam nicht wusste, dass es existierte. Die Testergebnisse liefern spezifische, priorisierte Verbesserungsziele, die das Team über zwei Sprints adressiert. Ein Follow-up-Usability-Test drei Monate später zeigt messbare Verbesserung bei der Aufgabenabschlusszeit und eine Reduzierung von Fehlern für die drei Problemworkflows.
