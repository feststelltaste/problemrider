---
title: Angehäufte Entscheidungsschulden
description: Aufgeschobene Entscheidungen erzeugen für künftige Entscheidungen zusätzliche
  Komplexität und machen das System schwerer verständlich und veränderbar.
category:
- Architecture
- Management
- Process
related_problems:
- slug: decision-avoidance
  similarity: 0.8
- slug: delayed-decision-making
  similarity: 0.75
- slug: decision-paralysis
  similarity: 0.7
- slug: high-technical-debt
  similarity: 0.7
- slug: delayed-issue-resolution
  similarity: 0.7
- slug: accumulation-of-workarounds
  similarity: 0.65
solutions:
- architecture-decision-records
- decision-rights-and-escalation
- architecture-reviews
- technical-debt-backlog
- living-documentation
- architecture-governance
- written-first-communication
- team-retrospectives
- lightweight-design-review
- application-portfolio-inventory
- modernization-options-comparison
- no-regret-moves
layout: problem
lang: de
en_slug: accumulated-decision-debt
---

## Description

Angehäufte Entscheidungsschulden entstehen, wenn wichtige architektonische, gestalterische oder technische Entscheidungen durchgängig aufgeschoben werden, was einen kumulativen Effekt erzeugt, bei dem jede verschobene Entscheidung künftige Entscheidungen komplexer und eingeschränkter macht. Diese Schulden häufen sich ähnlich an wie technische Schulden: Das Vermeiden schwieriger Entscheidungen im Kurzfristigen schafft langfristig zunehmend teurere Probleme. Irgendwann kann das Gewicht der angehäuften Entscheidungen ein Projekt lähmen oder suboptimale Entscheidungen erzwingen, die bei früherer Entscheidungsfindung hätten vermieden werden können.

## Indicators ⟡

- Aktuelle Entscheidungen werden durch mehrere frühere Nicht-Entscheidungen eingeschränkt
- Das Team diskutiert häufig darüber, wie vergangene Unentschlossenheit aktuelle Optionen einschränkt
- Einfache Entscheidungen werden aufgrund angehäufter Unsicherheit komplex
- Mehrere voneinander abhängige Entscheidungen müssen gleichzeitig getroffen werden
- Das Team äußert das Gefühl, durch frühere Vermeidung von Entscheidungen "gefangen" zu sein

## Symptoms ▲

- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Wenn Entscheidungen aufgeschoben werden, schaffen Teams temporäre Workarounds, um voranzukommen, und diese häufen sich im Laufe der Zeit an.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Die kumulative Komplexität vieler aufgeschobener Entscheidungen verlangsamt alle künftige Entwicklung, da jede Änderung ungelöste Einschränkungen navigieren muss.
- [Stagnierende Architektur](stagnierende-architektur.md)
<br/>  Architektur kann sich nicht weiterentwickeln, wenn zentrale Entscheidungen dauerhaft aufgeschoben werden, was zu Stagnation führt.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Jede aufgeschobene Entscheidung trägt zu den gesamten technischen Schulden des Systems bei, da temporäre Lösungen dauerhaft werden.
- [Suboptimale Lösungen](suboptimale-loesungen.md)
<br/>  Wenn angehäufte aufgeschobene Entscheidungen schließlich unter Druck gelöst werden müssen, sind die resultierenden Lösungen aufgrund eingeschränkter Optionen oft suboptimal.
- [Architektonische Fehlpassung](architektonische-fehlpassung.md)
<br/>  Aufgeschobene architektonische Entscheidungen schränken das System ein, bis es sich ändernden Anforderungen nicht mehr anpassen kann.

## Causes ▼

- [Entscheidungsvermeidung](entscheidungsvermeidung.md)
<br/>  Systematische Vermeidung von Entscheidungen ist das direkte Verhalten, das die Anhäufung von Entscheidungsschulden verursacht.
- [Verzögerte Entscheidungsfindung](verzoegerte-entscheidungsfindung.md)
<br/>  Entscheidungen durchgängig aufzuschieben, statt sie zeitnah zu treffen, führt direkt zur Anhäufung von Entscheidungsschulden.
- [Entscheidungslähmung](entscheidungslaehmung.md)
<br/>  Wenn Teams sich nicht zwischen Optionen entscheiden können, werden Entscheidungen nie getroffen, und die entstehenden Schulden häufen sich im Laufe der Zeit an.

## Detection Methods ○

- **Entscheidungsabhängigkeits-Mapping:** Visualisierung, wie aufgeschobene Entscheidungen künftige Optionen einschränken
- **Entscheidungs-Zeitleisten-Analyse:** Nachverfolgung, wie lange wichtige Entscheidungen ungelöst bleiben
- **Bewertung von Entscheidungseinschränkungen:** Bewertung, wie frühere Unentschlossenheit aktuelle Optionen einschränkt
- **Nachverfolgung von Entscheidungskaskaden:** Beobachtung, wann die Lösung einer Entscheidung mehrere andere auslöst
- **Team-Retrospektiven:** Diskussion, wie vergangene Entscheidungsvermeidung die aktuelle Arbeit beeinflusst

## Examples

Ein Entwicklungsteam schob die Entscheidung zwischen Microservices und monolithischer Architektur monatelang auf, verschob dann die Auswahl der Datenbanktechnologie und stellte API-Design-Entscheidungen zurück. Als es schließlich die Benutzerauthentifizierung implementieren muss, stellt es fest, dass die Datenbankwahl das API-Design beeinflusst, was wiederum die Architekturwahl beeinflusst, was wiederum die Deployment-Strategie beeinflusst. Was vier unabhängige, über die Zeit verteilte Entscheidungen hätten sein sollen, ist zu einer komplexen, voneinander abhängigen Entscheidungsmatrix geworden, die auf einmal gelöst werden muss, was ihre Optionen erheblich einschränkt und Kompromisse erzwingt. Ein weiteres Beispiel betrifft ein Team, das die Entscheidung über Fehlerbehandlungsmuster, Logging-Standards und Monitoring-Ansätze vermied. Als Produktionsprobleme auftreten, erkennt es, dass diese Entscheidungen miteinander verknüpft sind und alle drei gleichzeitig unter Druck gelöst werden müssen, was zu inkonsistenten Implementierungen führt, die mehr Probleme schaffen, als sie lösen.
