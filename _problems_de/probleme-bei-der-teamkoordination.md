---
title: Probleme bei der Teamkoordination
description: Eine Situation, in der mehrere Entwickler oder Teams Schwierigkeiten
  haben, an derselben Codebasis zusammenzuarbeiten.
category:
- Process
- Team
related_problems:
- slug: duplicated-work
  similarity: 0.7
- slug: team-silos
  similarity: 0.7
- slug: communication-breakdown
  similarity: 0.65
- slug: merge-conflicts
  similarity: 0.65
- slug: reduced-team-productivity
  similarity: 0.65
- slug: team-dysfunction
  similarity: 0.65
solutions:
- clear-ownership-model
- clear-roles-and-ownership
- structured-communication-protocols
- team-boundaries-aligned-to-architecture
- team-retrospectives
- team-working-agreements
- knowledge-rotation
- value-stream-mapping
- modularization-and-bounded-contexts
layout: problem
lang: de
en_slug: team-coordination-issues
---

## Description
Probleme bei der Teamkoordination entstehen, wenn mehrere Entwickler oder Teams an derselben Codebasis arbeiten müssen und Schwierigkeiten haben, ihre Arbeit zu koordinieren. Dies kann zu Merge-Konflikten, doppeltem Aufwand und einer allgemeinen Verlangsamung des Entwicklungstempos führen. Probleme bei der Teamkoordination sind oft ein Zeichen einer monolithischen Architektur, bei der alles eng gekoppelt ist und es schwierig ist, isoliert an verschiedenen Teilen des Systems zu arbeiten.

## Indicators ⟡
- Häufige Merge-Konflikte.
- Entwickler sind oft blockiert, während sie darauf warten, dass andere Entwickler ihre Arbeit fertigstellen.
- Es gibt viel doppelten Aufwand.
- Es ist schwierig, ein klares Bild des Gesamtstatus des Projekts zu erhalten.

## Symptoms ▲

- [Doppelter Aufwand](doppelter-aufwand.md)
<br/>  Ohne Koordination lösen mehrere Entwickler unabhängig voneinander dieselben Probleme, was Aufwand verschwendet.
- [Merge-Konflikte](merge-konflikte.md)
<br/>  Schlechte Koordination führt dazu, dass Entwickler widersprüchliche Änderungen an denselben Codebereichen vornehmen, was zu häufigen Merge-Konflikten führt.
- [Verringerte Teamproduktivität](verringerte-teamproduktivitaet.md)
<br/>  Entwickler, die blockiert sind, während sie auf andere warten, und Zeit, die mit der Lösung von Konflikten verbracht wird, verringern direkt den Team-Output.
- [Inkonsistente Codebasis](inkonsistente-codebasis.md)
<br/>  Unkoordinierte Entwicklung führt dazu, dass unterschiedliche Ansätze und Muster für ähnliche Probleme über die Codebasis hinweg genutzt werden.
- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Koordinations-Overhead, blockierende Abhängigkeiten und Konfliktlösung verlangsamen das gesamte Liefertempo.

## Causes ▼

- [Einschränkungen durch monolithische Architektur](einschraenkungen-durch-monolithische-architektur.md)
<br/>  Eine monolithische Architektur zwingt alle Teams, in derselben Codebasis zu arbeiten, was Koordination essentiell, aber schwierig macht.
- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Wenn Komponenten eng gekoppelt sind, betreffen Änderungen in einem Bereich häufig andere, was sorgfältige Koordination zwischen Entwicklern erfordert.
- [Team-Silos](team-silos.md)
<br/>  Teams, die isoliert arbeiten, fehlt das Bewusstsein dafür, was andere tun, was die Koordination bei gemeinsam genutzten Codebasen erschwert.
- [Schlecht definierte Verantwortlichkeiten](schlecht-definierte-verantwortlichkeiten.md)
<br/>  Wenn Code-Eigentumsgrenzen und Teamverantwortlichkeiten schlecht definiert sind, wird Koordination schwierig, weil unklar ist, wer für was zuständig ist.

## Detection Methods ○
- **Versionskontrollmetriken:** Nutzung von Werkzeugen zur Messung der Anzahl von Merge-Konflikten und der Zeit, die Entwickler mit deren Lösung verbringen.
- **Entwicklerbefragungen:** Befragung von Entwicklern, ob sie das Gefühl haben, effektiv mit anderen Entwicklern im Team zusammenarbeiten zu können.
- **Projektmanagement-Metriken:** Verfolgung der Zeit, die Entwickler damit verbringen, auf die Fertigstellung der Arbeit anderer Entwickler zu warten.

## Examples
Ein Unternehmen hat eine große, monolithische E-Commerce-Anwendung. Das Frontend-Team und das Backend-Team treten sich konstant gegenseitig auf die Füße. Das Frontend-Team möchte Änderungen an der UI vornehmen, muss aber darauf warten, dass das Backend-Team Änderungen an der API vornimmt. Das Backend-Team ist mit der Arbeit an anderen Features beschäftigt, sodass das Frontend-Team oft blockiert ist. Dies führt zu viel Frustration und einer Verlangsamung des Entwicklungstempos.
