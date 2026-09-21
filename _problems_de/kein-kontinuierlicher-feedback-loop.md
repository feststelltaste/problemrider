---
title: Kein kontinuierlicher Feedback-Loop
description: Stakeholder werden nicht während des gesamten Entwicklungsprozesses
  einbezogen, und Feedback wird erst ganz am Ende gesammelt, was zu fehlausgerichteten
  Liefergegenständen führt.
category:
- Communication
- Process
related_problems:
- slug: feedback-isolation
  similarity: 0.8
- slug: stakeholder-developer-communication-gap
  similarity: 0.7
- slug: misaligned-deliverables
  similarity: 0.7
- slug: team-members-not-engaged-in-review-process
  similarity: 0.6
- slug: work-blocking
  similarity: 0.6
- slug: feature-gaps
  similarity: 0.6
solutions:
- continuous-feedback
- evolutionary-requirements-development
- requirements-analysis
- stakeholder-feedback-loops
- direct-feedback
- feedback-mechanisms
- on-site-customer
- user-communities
layout: problem
lang: de
en_slug: no-continuous-feedback-loop
---

## Description
Ein kontinuierlicher Feedback-Loop ist essenziell für agile Entwicklung und erlaubt Teams, ihren Prozess regelmäßig zu überprüfen und anzupassen. Wenn dieser Loop fehlt, operieren Teams im Vakuum, ohne zu wissen, wie ihre Arbeit von Nutzern aufgenommen wird oder ob sie auf dem richtigen Weg sind, ihre Ziele zu erreichen. Dies kann zu einer Trennung zwischen Entwicklungsteam und Geschäft, dem Versäumnis, Probleme zeitnah anzugehen, und einem Produkt führen, das die Nutzerbedürfnisse nicht erfüllt. Die Etablierung eines regelmäßigen Feedback-Rhythmus ist entscheidend für jedes Team, das sich verbessern möchte.

## Indicators ⟡
- Das Team erhält kein regelmäßiges Feedback von Stakeholdern.
- Das Team nutzt keinen Prototyp oder Mockup zur Klärung von Anforderungen.
- Das Team erhält während des Entwicklungsprozesses kein Feedback von Nutzern.
- Das Team führt keine regelmäßigen Demos oder Reviews durch.

## Symptoms ▲

- [Fehlausgerichtete Liefergegenstände](fehlausgerichtete-liefergegenstaende.md)
<br/>  Ohne regelmäßiges Feedback erfolgt Entwicklung auf Grundlage von Annahmen, was Liefergegenstände produziert, die nicht den Stakeholder-Erwartungen entsprechen.
- [Ressourcenverschwendung](ressourcenverschwendung.md)
<br/>  Erheblicher Entwicklungsaufwand wird verschwendet, um Features zu bauen, die aufgrund später Entdeckung der Fehlausrichtung neu designt oder verworfen werden müssen.
- [Scope Creep](scope-creep.md)
<br/>  Ohne laufendes Feedback zur Validierung der Richtung häufen sich Anforderungen ungebremst an, während Stakeholder Anfragen erst am Ende hinzufügen.
- [Frustration der Stakeholder](frustration-der-stakeholder.md)
<br/>  Stakeholder werden frustriert, wenn sie das finale Produkt sehen und es aufgrund fehlender Einbeziehung nicht ihren Erwartungen entspricht.

## Causes ▼

- [Kommunikationslücke zwischen Stakeholdern und Entwicklern](kommunikationsluecke-zwischen-stakeholdern-und-entwicklern.md)
<br/>  Schlechte Kommunikationskanäle zwischen Entwicklern und Stakeholdern verhindern regelmäßigen Feedback-Austausch während der Entwicklung.
- [Zeitdruck](zeitdruck.md)
<br/>  Unter engen Terminen überspringen Teams Feedback-Sitzungen und Demos, um sich auf Entwicklung zu fokussieren, was Möglichkeiten zur Kurskorrektur eliminiert.
- [Team-Silos](team-silos.md)
<br/>  Organisatorische Silos zwischen Geschäfts- und Entwicklungsteams schaffen strukturelle Barrieren für laufende Zusammenarbeit und Feedback.

## Detection Methods ○

- **Projekt-Audits:** Überprüfung von Projektplänen und Kommunikationsprotokollen, um die Häufigkeit von Stakeholder-Engagement und Feedback-Sitzungen zu sehen.
- **Post-Mortems/Retrospektiven:** Analyse von Projekten, bei denen Liefergegenstände fehlausgerichtet waren, zur Identifikation des Timings und der Wirksamkeit von Feedback-Loops.
- **Fehler-Tracking-Metriken:** Nachverfolgung der Phase, in der Fehler oder Änderungsanfragen eingeführt werden (z. B. während der Entwicklung vs. nach dem Release).
- **Stakeholder-Interviews:** Befragung von Stakeholdern zu ihrer Einbeziehung in den Entwicklungsprozess und ihrer Zufriedenheit mit den Feedback-Möglichkeiten.

## Examples
Ein Team verbringt sechs Monate mit der Entwicklung eines komplexen Reporting-Moduls. Sie zeigen es den Geschäfts-Stakeholdern erst eine Woche vor dem geplanten Launch. Die Stakeholder identifizieren sofort mehrere kritische Mängel und fehlende Features, die den Nutzen des Moduls grundlegend verändern, was eine komplette Neugestaltung erzwingt und den Launch um mehrere Monate verzögert. In einem anderen Fall wird eine Webanwendung entwickelt. Das Design-Team erstellt zu Beginn Mockups, und das Entwicklungsteam baut die UI basierend darauf. Es gibt jedoch keine regelmäßigen Check-ins mit dem Design-Team oder Endnutzern. Als die UI schließlich integriert wird, wird entdeckt, dass ein wichtiger Interaktionsablauf verwirrend ist und komplett neu implementiert werden muss. Kontinuierliche Feedback-Loops sind ein Eckpfeiler agiler und iterativer Entwicklungsmethoden. Ihr Fehlen führt zu erheblicher Verschwendung, erhöhtem Risiko und einer höheren Wahrscheinlichkeit, ein Produkt zu liefern, das Markt- oder Geschäftsbedürfnisse nicht erfüllt, besonders im Kontext sich entwickelnder Legacy-Systeme.
