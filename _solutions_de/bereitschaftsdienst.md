---
title: Bereitschaftsdienst
description: Sicherstellung, dass Mitarbeiter verfügbar sind, um schnell auf
  Vorfälle und Probleme zu reagieren.
category:
- Process
- Operations
problems:
- slow-incident-resolution
- constant-firefighting
- system-outages
- knowledge-silos
- poorly-defined-responsibilities
- developer-frustration-and-burnout
- overworked-teams
- increased-stress-and-burnout
- mental-fatigue
- lack-of-ownership-and-accountability
layout: solution
lang: de
en_slug: on-call-duty
related_solutions:
- slug: runbooks
  similarity: 0.75
- slug: incident-management
  similarity: 0.75
- slug: sustainable-pace-practices
  similarity: 0.7
- slug: clear-roles-and-ownership
  similarity: 0.7
- slug: security-incident-handling
  similarity: 0.7
- slug: cross-functional-skill-development
  similarity: 0.7
---

## Description

Bereitschaftsdienst ist eine formale Rotation, die bestimmten Personen die Verantwortung überträgt, auf Produktionsvorfälle außerhalb der normalen Arbeitszeit zu reagieren, und die informelle Regelung ersetzt, bei der dieselben ein oder zwei Personen, die das Legacy-System verstehen, jedes Mal gerufen werden, wenn etwas kaputtgeht. Die Einrichtung einer Rotation mit klaren Eskalationspfaden, dokumentierten Runbooks und definierten Reaktionserwartungen verteilt operatives Wissen über das Team, statt es in wenigen überlasteten Personen zu konzentrieren. In Legacy-Systemen, in denen institutionelles Wissen oft dünn und ungleich verteilt ist, erzwingt eine gut geführte Bereitschaftsrotation, dass dieses Wissen als Teil des Onboardings neuer Rotationsmitglieder aufgeschrieben und geteilt wird, während sie gleichzeitig Verantwortlichkeit dafür schafft, die wiederkehrenden Probleme zu beheben, die wiederholte Alarmierungen erzeugen, statt sie unbegrenzt zu tolerieren.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Richten Sie einen fairen Rotationsplan ein, der die Bereitschaftslast auf alle Teammitglieder verteilt
- Stellen Sie klare Eskalationspfade und Runbooks bereit, damit Bereitschaftsingenieure Legacy-System-Probleme effektiv handhaben können
- Definieren Sie Reaktionszeiterwartungen für jede Schweregradstufe und kommunizieren Sie diese an Stakeholder
- Statten Sie Bereitschaftsingenieure mit dem nötigen Zugang, Werkzeugen und der Dokumentation für die Legacy-System-Fehlersuche aus
- Vergüten Sie Bereitschaftsdienst angemessen, um Teammoral und Bereitschaft zur Teilnahme zu erhalten
- Führen Sie regelmäßige Bereitschaftsübergaben durch, die Kontext zu jüngsten Änderungen und bekannten Problemen einschließen
- Überprüfen Sie Bereitschafts-Metriken (Alarmierungshäufigkeit, Alarmierungen außerhalb der Arbeitszeit, MTTR) und beheben Sie Quellen übermäßiger Belastung

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Stellt schnelle Reaktion auf Produktionsvorfälle zu jeder Tages- und Nachtzeit sicher
- Verteilt operatives Wissen über das Team, statt sich auf wenige Experten zu verlassen
- Schafft Verantwortlichkeit für Produktionsqualität unter Entwicklern
- Bietet eine strukturierte Alternative zu Ad-hoc-Feuerwehrübungen

**Kosten und Risiken:**
- Bereitschaftsdienst verursacht Stress und kann zu Burnout beitragen, wenn er nicht gut gemanagt wird
- Häufige Alarmierungen stören die persönliche Zeit und beeinträchtigen die Work-Life-Balance
- Teams mit begrenztem Legacy-System-Wissen können während Bereitschaftsschichten Schwierigkeiten haben
- Unterbesetzte Bereitschaftsrotationen konzentrieren die Last auf zu wenige Personen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Softwareunternehmen hatte sich auf zwei leitende Ingenieure verlassen, die das Legacy-System am besten kannten, um alle Produktionsprobleme zu behandeln, unabhängig von der Uhrzeit. Beide brannten aus und waren zu Single Points of Failure für operatives Wissen geworden. Durch die Einführung einer formalen Bereitschaftsrotation mit umfassenden Runbooks und einem Buddy-System, das junior und senior Ingenieure paarte, verteilte das Team die Vorfallreaktion auf acht Personen. Das Alarmierungsvolumen der Bereitschaft wurde ebenfalls um 60 % reduziert, weil die Rotation das Team motivierte, wiederkehrende Probleme zu beheben, statt sie wiederholt zu umgehen.
