---
title: Kommunikationsrisiko innerhalb des Projekts
description: Missverständnisse und unklare Botschaften verringern Koordination und
  Vertrauen zwischen Projektbeteiligten.
category:
- Communication
- Process
- Team
related_problems:
- slug: communication-risk-outside-project
  similarity: 0.75
- slug: communication-breakdown
  similarity: 0.7
- slug: team-confusion
  similarity: 0.7
- slug: poor-communication
  similarity: 0.6
- slug: poor-planning
  similarity: 0.6
- slug: duplicated-work
  similarity: 0.6
solutions:
- structured-communication-protocols
- team-boundaries-aligned-to-architecture
- team-working-agreements
- knowledge-sharing-practices
- documentation-as-code
- ubiquitous-language
- written-first-communication
- team-retrospectives
- communities-of-practice
- lightweight-design-review
layout: problem
lang: de
en_slug: communication-risk-within-project
---

## Description

Kommunikationsrisiko innerhalb des Projekts entsteht, wenn Teammitglieder nicht in der Lage sind, Informationen wirksam zu teilen, die Botschaften der anderen zu verstehen oder ihre Aktivitäten zu koordinieren. Dies umfasst unklare Anforderungen, mehrdeutige technische Diskussionen, verpasste Nachrichten und Annahmen, die zu Missverständnissen führen. Schlechte interne Projektkommunikation schafft Verwirrung über Prioritäten, doppelten Aufwand und Entscheidungen auf Basis unvollständiger oder falscher Informationen.

## Indicators ⟡

- Teammitglieder bitten häufig um Klärung zuvor besprochener Themen
- Unterschiedliche Teammitglieder haben ein unterschiedliches Verständnis derselben Anforderungen
- Wichtige Entscheidungen werden getroffen, ohne alle relevanten Stakeholder zu informieren
- Nachrichten und Dokumentation sind mehrdeutig oder mehrfach interpretierbar
- Team-Meetings beinhalten häufig Verwirrung darüber, was zuvor vereinbart wurde

## Symptoms ▲

- [Doppelte Arbeit](doppelte-arbeit.md)
<br/>  Missverständnisse über Aufgabenzuweisungen führen dazu, dass mehrere Teammitglieder an denselben Problemen arbeiten.
- [Implementierungs-Nacharbeit](implementierungs-nacharbeit.md)
<br/>  Missverstandene Anforderungen aus unklarer interner Kommunikation zwingen dazu, Features neu zu bauen.
- [Teamverwirrung](teamverwirrung.md)
<br/>  Mehrdeutige Botschaften und verpasste Informationen erzeugen Verwirrung über Projektziele, Prioritäten und Entscheidungen.
- [Anforderungsmehrdeutigkeit](anforderungsmehrdeutigkeit.md)
<br/>  Schlechte interne Kommunikation führt dazu, dass unterschiedliche Teammitglieder dieselben Anforderungen unterschiedlich interpretieren.
- [Kommunikationsrisiko außerhalb des Projekts](kommunikationsrisiko-ausserhalb-des-projekts.md)
<br/>  Interne Fehlkommunikation über den Projektstatus pflanzt sich nach außen als inkonsistente oder ungenaue Botschaft an externe Stakeholder fort.

## Causes ▼

- [Sprachbarrieren](sprachbarrieren.md)
<br/>  Unterschiede in Sprache oder technischer Terminologie erzeugen Missverständnisse in der Teamkommunikation.
- [Team-Silos](team-silos.md)
<br/>  Isolierte Teams haben keine natürlichen Kommunikationskanäle, was den Informationsfluss innerhalb des Projekts verringert.
- [Unklare Erwartungen beim Teilen von Informationen](unklare-erwartungen-beim-teilen-von-informationen.md)
<br/>  Ohne klare Normen darüber, welche Informationen geteilt werden sollen, werden wichtige Details häufig weggelassen.
- [Ineffiziente Prozesse](ineffiziente-prozesse.md)
<br/>  Schlecht strukturierte Prozesse scheitern daran, regelmäßige Kontaktpunkte für Teammitglieder zu schaffen, um sich auszutauschen und abzustimmen.

## Detection Methods ○

- **Kommunikationsmuster-Analyse:** Nachverfolgung von Häufigkeit und Wirksamkeit unterschiedlicher Kommunikationsmethoden
- **Bewertung der Meeting-Wirksamkeit:** Bewertung, ob Meetings zu klarem Verständnis und Entscheidungen führen
- **Test der Nachrichtenklarheit:** Überprüfung von Dokumentation und Nachrichten auf Mehrdeutigkeit oder Verwirrung
- **Review der Entscheidungs-Nachvollziehbarkeit:** Bewertung, ob Teammitglieder verstehen, wie und warum Entscheidungen getroffen wurden
- **Team-Verständnis-Umfragen:** Regelmäßige Check-ins zur Klarheit der Kommunikation und zum gemeinsamen Verständnis

## Examples

Ein Entwicklungsteam erhält von der Produktverantwortlichen die Anforderung, dass "Nutzer effizient suchen können sollen." Das Backend-Team interpretiert dies als Notwendigkeit, Datenbankabfragen zu optimieren, das Frontend-Team konzentriert sich auf die Reaktionsfähigkeit der Benutzeroberfläche, und die Produktverantwortliche meinte eigentlich, dass Nutzer Ergebnisse schnell finden können sollen, unabhängig von der technischen Umsetzung. Jedes Team arbeitet wochenlang an seiner Interpretation, bevor die Fehlausrichtung während einer Demo entdeckt wird, was erhebliche Nacharbeit erfordert, um eine kohärente Lösung zu schaffen. Ein weiteres Beispiel betrifft ein verteiltes Team, in dem Entwickler in unterschiedlichen Zeitzonen E-Mail für die gesamte Kommunikation nutzen. Ein kritischer Fehler wird per E-Mail gemeldet, aber der zuständige Entwickler sieht die Nachricht erst am nächsten Tag, weil sie unter anderen E-Mails begraben war. In der Zwischenzeit beginnen andere Teammitglieder, an demselben Fehler zu arbeiten, weil sie annehmen, dass sich niemand darum kümmert, was zu doppeltem Aufwand und Verwirrung darüber führt, welche Behebung verwendet werden soll.
