---
title: Vermeidungsverhalten
description: Komplexe Aufgaben werden aufgeschoben oder ganz vermieden, aufgrund
  kognitiver Überlastung, Angst oder wahrgenommener Schwierigkeit.
category:
- Management
- Process
- Team
related_problems:
- slug: procrastination-on-complex-tasks
  similarity: 0.85
- slug: decision-avoidance
  similarity: 0.7
- slug: cognitive-overload
  similarity: 0.7
- slug: fear-of-failure
  similarity: 0.7
- slug: increased-cognitive-load
  similarity: 0.7
- slug: mental-fatigue
  similarity: 0.65
solutions:
- blameless-postmortems
- decision-rights-and-escalation
- psychological-safety-practices
- team-working-agreements
- work-in-progress-limits
- pair-and-mob-programming
- team-retrospectives
- clear-roles-and-ownership
- pilot-projects
- defect-triage-process
layout: problem
lang: de
en_slug: avoidance-behaviors
---

## Description

Vermeidungsverhalten entsteht, wenn Entwickler durchgängig komplexe oder herausfordernde Aufgaben aufschieben, verzögern oder vermeiden, aufgrund psychologischer Barrieren wie kognitiver Überlastung, Angst vor Scheitern oder mentaler Erschöpfung. Dieses Verhalten äußert sich als Prokrastination bei schwierigen Features, als Präferenz für einfache statt komplexe Aufgaben, oder als das Finden von Gründen, an anderen Tätigkeiten zu arbeiten, statt sich herausfordernden Problemen zu widmen. Im Laufe der Zeit kann Vermeidungsverhalten zu einem Rückstau schwieriger Arbeit und verringerter Teamfähigkeit führen.

## Indicators ⟡

- Entwickler wählen durchgängig leichtere Aufgaben gegenüber anspruchsvolleren
- Komplexe Features verbleiben viel länger im Backlog als einfache
- Teammitglieder finden Gründe, an anderen Aufgaben zu arbeiten, wenn ihnen schwierige Arbeit zugewiesen wird
- Wichtige, aber herausfordernde Aufgaben werden wiederholt aufgeschoben oder neu zugewiesen
- Entwickler äußern Angst oder Widerwillen, wenn über komplexe Features gesprochen wird

## Symptoms ▲

- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Wenn Entwickler es vermeiden, komplexe Grundprobleme anzugehen, schaffen sie stattdessen Workarounds, was zu angehäuften technischen Abkürzungen führt.
- [Verzögerte Projektzeitpläne](verzoegerte-projektzeitplaene.md)
<br/>  Das durchgängige Aufschieben komplexer Aufgaben führt dazu, dass Projektzeitpläne verrutschen, da kritische Arbeit unerledigt bleibt.
- [Zunehmende Brüchigkeit](zunehmende-bruechigkeit.md)
<br/>  Die Vermeidung notwendiger Refactorings und komplexer Wartungsarbeit lässt die Codebasis im Laufe der Zeit zunehmend brüchig werden.
- [Aufstauung von Arbeitswarteschlangen](aufstauung-von-arbeitswarteschlangen.md)
<br/>  Komplexe Aufgaben stauen sich im Backlog an, da Entwickler sie wiederholt zugunsten einfacherer Arbeit verschieben.
- [Verringerte Innovation](verringerte-innovation.md)
<br/>  Wenn Teammitglieder herausfordernde Aufgaben vermeiden, verliert das Team seine Fähigkeit, zu innovieren und schwierige Probleme zu lösen.

## Causes ▼

- [Kognitive Überlastung](kognitive-ueberlastung.md)
<br/>  Wenn Entwickler von der Systemkomplexität mental überwältigt sind, vermeiden sie schwierige Aufgaben, um die kognitive Belastung zu verringern.
- [Angst vor Scheitern](angst-vor-scheitern.md)
<br/>  Die Angst, Fehler zu machen oder für Misserfolge verantwortlich gemacht zu werden, treibt Entwickler dazu, riskante oder komplexe Arbeit zu vermeiden.
- [Schuldzuweisungskultur](schuldzuweisungskultur.md)
<br/>  Wenn Fehler bestraft statt als Lerngelegenheiten behandelt werden, vermeiden Entwickler herausfordernde Aufgaben, um das Risiko des Scheiterns zu minimieren.
- [Brüchige Codebasis](bruechige-codebasis.md)
<br/>  Eine brüchige Codebasis macht komplexe Änderungen riskant und unvorhersehbar, was Entwickler davon abhält, sie zu versuchen.

## Detection Methods ○

- **Analyse der Aufgabenerledigungsmuster:** Vergleich der Erledigungsraten für einfache vs. komplexe Aufgaben
- **Backlog-Alter-Analyse:** Nachverfolgung, wie lange komplexe Aufgaben unbegonnen bleiben
- **Entwickler-Umfragen:** Befragung zu Aufgabenpräferenzen und Angstniveaus für verschiedene Arbeitsarten
- **Beobachtungen bei der Sprint-Planung:** Beobachtung, wie Aufgaben während der Planung ausgewählt und vermieden werden
- **Einzelgespräche:** Diskussion individueller Bedenken zu bestimmten Arten von Arbeit

## Examples

Ein Entwicklungsteam hat drei komplexe Features im Backlog, die über sechs Monate wiederholt in künftige Sprints verschoben wurden. Jedes davon beinhaltet das Refactoring eng gekoppelten Legacy-Codes, und Entwickler entscheiden sich durchgängig dafür, an neuen Feature-Ergänzungen zu arbeiten, selbst wenn das komplexe Refactoring mehr Wert liefern würde. Die vermiedene Arbeit erzeugt zunehmende technische Schulden und erschwert künftige Entwicklung. Ein weiteres Beispiel betrifft Entwickler, die es vermeiden, bestimmte Produktionsprobleme zu debuggen, weil sie komplexe Wechselwirkungen zwischen mehreren Microservices beinhalten. Stattdessen konzentrieren sie sich auf leichtere Fehlerbehebungen und Feature-Arbeit, wodurch die schwierigen Probleme ungelöst bleiben und anhaltende Stabilitätsprobleme des Systems verursachen, die sich im Laufe der Zeit summieren.
