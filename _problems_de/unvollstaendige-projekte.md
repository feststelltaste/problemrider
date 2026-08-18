---
title: Unvollständige Projekte
description: Features werden begonnen, aber aufgrund sich verschiebender Prioritäten
  nie fertiggestellt, was zu erheblich verschwendetem Aufwand und einem Gefühl der
  Frustration für das Entwicklungsteam führt.
category:
- Process
related_problems:
- slug: gold-plating
  similarity: 0.7
- slug: stakeholder-developer-communication-gap
  similarity: 0.65
- slug: large-feature-scope
  similarity: 0.65
- slug: slow-feature-development
  similarity: 0.65
- slug: frequent-changes-to-requirements
  similarity: 0.65
- slug: development-disruption
  similarity: 0.65
solutions:
- architecture-roadmap
- impact-mapping
- mikado-method
- work-in-progress-limits
- definition-of-ready
- explicit-prioritization-framework
- small-change-batches
- walking-skeleton
- regular-stakeholder-demonstrations
- executive-sponsorship
- staged-investment-with-decision-gates
- large-scale-refactoring
layout: problem
lang: de
en_slug: incomplete-projects
---

## Description
Unvollständige Projekte sind ein verbreitetes Problem in der Softwareentwicklung. Sie treten auf, wenn Features begonnen, aber nie fertiggestellt werden, aufgrund sich verschiebender Prioritäten, fehlender klarer Anforderungen oder anderer unvorhergesehener Umstände. Dies kann zu einer Reihe von Problemen führen, einschließlich erheblich verschwendeten Aufwands, eines Gefühls der Frustration für das Entwicklungsteam und eines Glaubwürdigkeitsverlusts für die Organisation. Unvollständige Projekte sind oft ein Zeichen für ein schlecht gemanagtes Projekt.

## Indicators ⟡
- Das Team hat eine große Anzahl teilweise abgeschlossener Features.
- Das Team beginnt ständig neue Features, bevor es die alten fertiggestellt hat.
- Das Team liefert keinen Wert an die Nutzer.
- Das Team ist mit seiner Arbeit nicht zufrieden.

## Symptoms ▲

- [Verschwendeter Entwicklungsaufwand](verschwendeter-entwicklungsaufwand.md)
<br/>  In unfertige Features investierte Arbeit ist effektiv verschwendet, da teilweise abgeschlossener Code keinen Nutzerwert bietet.
- [Demoralisierung des Teams](demoralisierung-des-teams.md)
<br/>  Wiederholt zu sehen, wie ihre Arbeit aufgegeben wird, demoralisiert Entwickler und verringert ihre Motivation und ihr Engagement.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Teilweise abgeschlossene Features hinterlassen toten Code, halb implementierte Muster und architektonische Kompromisse, die zu technischen Schulden beitragen.
- [Vertrauensverlust bei Stakeholdern](vertrauensverlust-bei-stakeholdern.md)
<br/>  Wiederholtes Versäumnis, Projekte fertigzustellen, untergräbt das Vertrauen der Stakeholder in die Lieferfähigkeit des Teams.

## Causes ▼

- [Prioritäten-Thrashing](prioritaeten-thrashing.md)
<br/>  Häufig wechselnde Prioritäten ziehen Entwickler von aktueller Arbeit weg, bevor sie abgeschlossen werden kann.
- [Unklare Ziele und Prioritäten](unklare-ziele-und-prioritaeten.md)
<br/>  Ohne klare Richtung beginnen Teams neue Initiativen, bevor sie bestehende fertigstellen, weil Prioritäten mehrdeutig sind.
- [Schlechte Projektsteuerung](schlechte-projektsteuerung.md)
<br/>  Fehlendes Projekt-Monitoring erlaubt es dem Umfang, sich auszuweiten, und Zeitpläne, zu verrutschen, bis Projekte schließlich aufgegeben werden.
- [Sich änderndes Projekt-Scope](sich-aenderndes-projekt-scope.md)
<br/>  Ständige Umfangsänderungen machen es unmöglich, Projekte fertigzustellen, während sich das Ziel ständig verschiebt.

## Detection Methods ○
- **Work in Progress (WIP):** Nachverfolgung der Menge an Arbeit, die zu einem bestimmten Zeitpunkt in Bearbeitung ist.
- **Zykluszeit:** Nachverfolgung der Zeit, die zur Fertigstellung eines Features benötigt wird.
- **Durchsatz:** Nachverfolgung der Anzahl der Features, die über die Zeit fertiggestellt werden.
- **Team-Retrospektiven:** Diskussion der Gefühle des Teams zu unvollständigen Projekten in Retrospektiven.

## Examples
Ein Team arbeitet an einer neuen mobilen App. Das Team wird ständig gebeten, neue Features zu beginnen, bevor es die alten fertiggestellt hat. Infolgedessen hat das Team eine große Anzahl teilweise abgeschlossener Features. Das Team liefert keinen Wert an die Nutzer und wird zunehmend frustriert. Das Projekt wird schließlich abgesagt.
