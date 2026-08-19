---
title: Unrealistische Termine
description: Das Management setzt aggressive Termine, die den tatsächlich erforderlichen
  Aufwand nicht berücksichtigen, was zu kompromittierter Qualität und nicht nachhaltigen
  Arbeitspraktiken führt.
category:
- Management
- Process
related_problems:
- slug: time-pressure
  similarity: 0.75
- slug: missed-deadlines
  similarity: 0.75
- slug: deadline-pressure
  similarity: 0.75
- slug: unrealistic-schedule
  similarity: 0.7
- slug: constantly-shifting-deadlines
  similarity: 0.7
- slug: delayed-project-timelines
  similarity: 0.7
solutions:
- iterative-development
- requirements-analysis
- short-iteration-cycles
- capacity-based-planning
- explicit-prioritization-framework
- regular-stakeholder-demonstrations
- story-mapping
- value-stream-mapping
- definition-of-ready
layout: problem
lang: de
en_slug: unrealistic-deadlines
---

## Description

Unrealistische Termine treten auf, wenn Projektzeitpläne ohne ordentliche Berücksichtigung des tatsächlich erforderlichen Aufwands, der Systemkomplexität, verfügbarer Ressourcen oder potenzieller Risiken gesetzt werden. Diese Termine entstehen oft aus Geschäftsdruck, Wettbewerbsbedenken oder Missverständnis der Entwicklungskomplexität. Unrealistische Termine schaffen eine Kaskade von Problemen, einschließlich Qualitätskompromissen, Entwicklerstress und letztlich verpassten Lieferzielen.

## Indicators ⟡

- Termine werden gesetzt, bevor Entwicklungsschätzungen vorliegen
- Projektzeitpläne stimmen nicht mit der Teamkapazität oder historischer Velocity überein
- Termine bleiben fest, selbst wenn der Scope zunimmt oder Komplexität entdeckt wird
- Teams arbeiten konsequent Überstunden, um Termine einzuhalten
- Qualitätsstandards werden kompromittiert, um Zeitplanbeschränkungen einzuhalten

## Symptoms ▲

- [Termindruck](termindruck.md)
<br/>  Unrealistische Termine schaffen intensiven Druck auf Entwicklungsteams, schneller zu liefern, als machbar ist.
- [Verpasste Termine](verpasste-termine.md)
<br/>  Wenn Termine den tatsächlich erforderlichen Aufwand nicht berücksichtigen, werden sie trotz Teambemühungen häufig verpasst.
- [Qualitätsverschlechterung](qualitaetsverschlechterung.md)
<br/>  Teams machen Abstriche bei Testen, Code-Reviews und Design, um unrealistische Zeitpläne einzuhalten, was direkt die Qualität verschlechtert.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Anhaltende Überstundenarbeit, die zur Einhaltung unrealistischer Termine erforderlich ist, führt zu Entwicklerburnout und Erschöpfung.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Enge Termine zwingen Entwickler, schnelle Workarounds statt ordentlicher Lösungen zu implementieren.
- [Erhöhte Fehlerraten](erhoehte-fehlerraten.md)
<br/>  Das Hetzen zur Einhaltung unrealistischer Termine verursacht, dass Entwickler Testen überspringen und mehr Bugs einführen.
- [Zeitdruck](zeitdruck.md)
<br/>  Externer Geschäftsdruck von Stakeholdern drängt das Management, sich zu aggressiven Zeitplänen zu verpflichten, unabhängig von technischer Machbarkeit.

## Causes ▼

- [Schlechte Planung](schlechte-planung.md)
<br/>  Termine, die ohne ordentliche Schätzung oder Verständnis der tatsächlich erforderlichen Arbeit gesetzt werden, führen zu unrealistischen Zeitplänen.
- [Anforderungsmehrdeutigkeit](anforderungsmehrdeutigkeit.md)
<br/>  Wenn Anforderungen vage sind, ist die Aufwandsschätzung ungenau, was jeden auf diesen Schätzungen basierenden Termin unrealistisch macht.
- [Marktdruck](marktdruck.md)
<br/>  Marktdruck ist bereits als Ursache von Zeitdruck gelistet, ist aber auch eine direkte Ursache unrealistischer Termine, wenn Wettbewerbszwänge das Management dazu treiben, Termine ohne technische Validierung zu setzen.

## Detection Methods ○

- **Schätzung-vs.-Termin-Analyse:** Vergleich von Entwicklungsschätzungen mit auferlegten Terminen
- **Historische Zeitplananalyse:** Verfolgung der Genauigkeit von Terminvorhersagen über die Zeit
- **Teamkapazitätsbewertung:** Messung tatsächlicher Teamkapazität gegenüber geplanter Arbeit
- **Überstundenverfolgung:** Überwachung der Arbeitsstunden, die zur Einhaltung von Terminen erforderlich sind
- **Qualitätsauswirkungsanalyse:** Bewertung der Korrelation zwischen engen Terminen und Fehlerraten

## Examples

Ein Marketing-Team verpflichtet sich, ein neues E-Commerce-Feature auf einer Messe in 6 Wochen zu launchen, aber das Entwicklungsteam schätzt, dass die Arbeit 12 Wochen benötigt, basierend auf der Komplexität der Integration mit bestehenden Bestands- und Zahlungssystemen. Das Management besteht darauf, dass der Termin nicht verhandelbar ist, was das Team zwingt, eine vereinfachte Version mit erheblichen Einschränkungen und mehreren Workarounds zu implementieren. Die überstürzte Implementierung führt zu Bugs, die 3 Wochen Post-Launch-Fixes erfordern, was letztlich länger dauert als die ursprüngliche Schätzung. Ein weiteres Beispiel betrifft eine mobile App, bei der das Management Investoren ein größeres Update in 2 Monaten verspricht, aber das Entwicklungsteam 4 Monate braucht, um die neuen Features ordentlich zu implementieren und Kompatibilität über Gerätetypen hinweg sicherzustellen. Das Team liefert pünktlich, indem es umfassendes Testen überspringt, was zu Abstürzen auf 30 % der Geräte führt und Notfall-Patches erfordert, die den Ruf des Unternehmens schädigen.
