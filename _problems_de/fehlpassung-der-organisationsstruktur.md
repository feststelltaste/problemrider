---
title: Fehlpassung der Organisationsstruktur
description: Eine Situation, in der die Struktur der Organisation nicht mit der
  Architektur des Systems übereinstimmt.
category:
- Architecture
- Process
- Team
related_problems:
- slug: architectural-mismatch
  similarity: 0.75
- slug: team-coordination-issues
  similarity: 0.6
- slug: inadequate-mentoring-structure
  similarity: 0.6
- slug: process-software-misfit
  similarity: 0.6
- slug: misaligned-deliverables
  similarity: 0.6
- slug: capacity-mismatch
  similarity: 0.6
solutions:
- clear-ownership-model
- clear-roles-and-ownership
- domain-aligned-architecture
- team-boundaries-aligned-to-architecture
- team-retrospectives
- modularization-and-bounded-contexts
- value-stream-mapping
- knowledge-rotation
- executive-sponsorship
- large-scale-refactoring
layout: problem
lang: de
en_slug: organizational-structure-mismatch
---

## Description
Eine Fehlpassung der Organisationsstruktur ist eine Situation, in der die Struktur der Organisation nicht mit der Architektur des Systems übereinstimmt. Dies ist ein häufiges Problem in Unternehmen, die eine monolithische Architektur haben, aber in kleine, autonome Teams organisiert sind. Eine Fehlpassung der Organisationsstruktur kann zu einer Reihe von Problemen führen, einschließlich Problemen bei der Teamkoordination, Kommunikationszusammenbrüchen und einer Verlangsamung der Entwicklungsgeschwindigkeit.

## Indicators ⟡
- Die Teams sind um Features herum organisiert, aber die Architektur ist monolithisch.
- Die Teams treten sich ständig gegenseitig auf die Füße.
- Es gibt viel doppelten Aufwand.
- Es ist schwierig, ein klares Bild vom Gesamtstatus des Projekts zu erhalten.

## Symptoms ▲

- [Probleme bei der Teamkoordination](probleme-bei-der-teamkoordination.md)
<br/>  Wenn organisatorische Grenzen nicht mit der Systemarchitektur übereinstimmen, müssen Teams sich ständig über fehlangepasste Grenzen hinweg koordinieren.
- [Merge-Konflikte](merge-konflikte.md)
<br/>  Mehrere Teams, die aufgrund struktureller Fehlpassung an derselben monolithischen Codebasis arbeiten, führen zu häufigen Versionskontrollkonflikten.
- [Doppelter Aufwand](doppelter-aufwand.md)
<br/>  Teams, die in fehlangepassten Strukturen arbeiten, duplizieren unwissentlich Arbeit, weil Eigentumsgrenzen unklar sind.
- [Kommunikationszusammenbruch](kommunikationszusammenbruch.md)
<br/>  Fehlpassung zwischen organisatorischer und System-Struktur schafft unklare Kommunikationskanäle, was zu Informationsverlust und Fehlkoordination führt.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Teams, die sich gegenseitig auf die Füße treten, und exzessive teamübergreifende Koordination verlangsamen das gesamte Entwicklungstempo.

## Causes ▼

- [Einschränkungen durch monolithische Architektur](einschraenkungen-durch-monolithische-architektur.md)
<br/>  Eine monolithische Architektur zwingt mehrere autonome Teams, an derselben Codebasis zu arbeiten, was die Fehlpassung zwischen Teamstruktur und Systemgrenzen schafft.
- [Stagnierende Architektur](stagnierende-architektur.md)
<br/>  Wenn sich die Systemarchitektur nicht zusammen mit organisatorischen Änderungen weiterentwickelt, wächst die Fehlpassung zwischen Struktur und Architektur.
- [Schnelles Teamwachstum](schnelles-teamwachstum.md)
<br/>  Schnelle Expansion von Teams ohne entsprechende architektonische Änderungen schafft Fehlpassung zwischen organisatorischer und System-Struktur.

## Detection Methods ○
- **Architekturdiagramme:** Erstellung eines Diagramms der Systemarchitektur zur Identifikation, wie das System strukturiert ist.
- **Organigramme:** Erstellung eines Diagramms der Organisation zur Identifikation, wie die Teams strukturiert sind.
- **Entwicklerbefragungen:** Befragung von Entwicklern, ob sie das Gefühl haben, effektiv mit anderen Teams arbeiten zu können.

## Examples
Ein Unternehmen hat eine große, monolithische E-Commerce-Anwendung. Das Unternehmen ist in eine Reihe kleiner, autonomer Teams organisiert. Jedes Team ist für ein anderes Feature der Anwendung verantwortlich. Die Teams treten sich ständig gegenseitig auf die Füße, weil sie alle an derselben Codebasis arbeiten. Dies führt zu viel Frustration und einer Verlangsamung des Entwicklungstempos.
