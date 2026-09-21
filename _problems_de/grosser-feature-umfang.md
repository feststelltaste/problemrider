---
title: Großer Feature-Umfang
description: Features sind zu groß, um in kleinere, inkrementelle Änderungen aufgeteilt
  zu werden, was zu langlebigen Branches und Integrationsproblemen führt.
category:
- Code
- Process
related_problems:
- slug: long-lived-feature-branches
  similarity: 0.75
- slug: slow-feature-development
  similarity: 0.7
- slug: feature-creep
  similarity: 0.7
- slug: incomplete-projects
  similarity: 0.65
- slug: large-estimates-for-small-changes
  similarity: 0.65
- slug: gold-plating
  similarity: 0.65
solutions:
- iterative-development
- product-owner
- requirements-analysis
- story-mapping
- user-stories
- feature-driven-development
- lightweight-design-review
- small-change-batches
- definition-of-ready
- walking-skeleton
- mikado-method
layout: problem
lang: de
en_slug: large-feature-scope
---

## Description
Großer Feature-Umfang ist ein Problem, das auftritt, wenn ein Feature zu groß und komplex ist, um in einer einzelnen, kurzen Iteration entwickelt und geliefert zu werden. Dies kann zu einer Reihe von Problemen führen, einschließlich langlebiger Feature-Branches, fehlender Sichtbarkeit in den Fortschritt des Features und einem hohen Risiko für Integrationsprobleme. Das Aufteilen großer Features in kleinere, besser handhabbare Teile ist ein Kernprinzip agiler Entwicklung und essenziell, um Risiko zu verringern und Nutzern schneller Wert zu liefern.

## Indicators ⟡
- Features brauchen durchgängig länger zur Entwicklung als erwartet.
- Das Team beschäftigt sich häufig mit Merge-Konflikten und Integrationsproblemen.
- Es fehlt Sichtbarkeit in den Fortschritt eines Features.

## Symptoms ▲

- [Langlebige Feature-Branches](langlebige-feature-branches.md)
<br/>  Große Features, die nicht aufgeteilt werden können, resultieren in Branches, die Wochen oder Monate leben und vom Haupt-Code auseinanderdriften.
- [Merge-Konflikte](merge-konflikte.md)
<br/>  Langlebige Branches, die durch große Features entstehen, häufen Merge-Konflikte an, während sich der Hauptbranch unabhängig weiterentwickelt.
- [Große Pull Requests](grosse-pull-requests.md)
<br/>  Features, die zu groß sind, um zerlegt zu werden, produzieren natürlich übergroße Pull Requests, die schwer zu reviewen sind.
- [Große, riskante Releases](grosse-riskante-releases.md)
<br/>  Großer Feature-Umfang bündelt viele Änderungen zusammen, was komplexe Deployments schafft, die schwer zu testen und anfällig für Fehlschläge sind.
- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Wenn Features nicht inkrementell geliefert werden können, müssen Nutzer warten, bis das gesamte große Feature fertig ist, bevor sie irgendeinen Wert erhalten.

## Causes ▼

- [Unzureichende Anforderungserhebung](unzureichende-anforderungserhebung.md)
<br/>  Schlechte Anforderungsanalyse versäumt es zu identifizieren, wie Features in kleinere, unabhängig lieferbare Teile zerlegt werden können.
- [Feature-Creep](feature-creep.md)
<br/>  Der Umfang weitet sich schrittweise aus, während neue Anforderungen zu einem bereits großen Feature hinzugefügt werden, was es noch schwerer macht, es aufzuteilen.
- [Einschränkungen durch monolithische Architektur](einschraenkungen-durch-monolithische-architektur.md)
<br/>  Eine monolithische Architektur erschwert es, Teile eines Features unabhängig zu liefern, was große Alles-oder-nichts-Implementierungen erzwingt.

## Detection Methods ○
- **Nachverfolgung der Feature-Durchlaufzeit:** Überwachung der Zeit, die zur Entwicklung und Lieferung eines Features benötigt wird, von der anfänglichen Idee bis zum finalen Release.
- **Analyse der Branching-Strategie:** Suche nach langlebigen Feature-Branches im Versionskontrollsystem.
- **Team-Retrospektiven:** Diskussion der Herausforderungen, mit denen das Team bei großen Features konfrontiert ist, und Identifikation von Wegen, sie in kleinere Teile aufzuteilen.

## Examples
Ein Team wird beauftragt, ein neues Reporting-Modul für eine Anwendung zu bauen. Das Modul ist sehr komplex und hat eine große Anzahl von Features. Das Team entscheidet sich, das gesamte Modul auf einem einzigen Feature-Branch zu bauen. Die Entwicklung dauert mehrere Monate, und als das Team schließlich bereit ist, den Branch zu mergen, stehen sie vor einer massiven Anzahl von Merge-Konflikten und Integrationsproblemen. Es braucht mehrere weitere Wochen, um die Probleme zu lösen und das Feature zu veröffentlichen. Dies ist ein klassisches Beispiel dafür, wie ein großer Feature-Umfang zu erheblichen Verzögerungen und einem hohen Risikoniveau führen kann.
