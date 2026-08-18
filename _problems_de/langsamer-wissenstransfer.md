---
title: Langsamer Wissenstransfer
description: Eine Situation, in der es lange dauert, bis neue Teammitglieder produktiv
  werden.
category:
- Communication
- Process
- Team
related_problems:
- slug: tacit-knowledge
  similarity: 0.65
- slug: knowledge-dependency
  similarity: 0.65
- slug: difficult-developer-onboarding
  similarity: 0.65
- slug: knowledge-silos
  similarity: 0.65
- slug: knowledge-gaps
  similarity: 0.65
- slug: knowledge-sharing-breakdown
  similarity: 0.65
solutions:
- knowledge-sharing-practices
- pair-and-mob-programming
- structured-onboarding-program
- collaborative-problem-solving
- code-reading-sessions
- internal-technical-coaching
- communities-of-practice
- written-first-communication
layout: problem
lang: de
en_slug: slow-knowledge-transfer
---

## Description
Langsamer Wissenstransfer ist eine Situation, in der es lange dauert, bis neue Teammitglieder produktiv werden. Dies ist ein häufiges Problem in Teams mit vielen Wissenssilos und mangelnder Dokumentation. Langsamer Wissenstransfer kann eine erhebliche Bremse für die Produktivität sein und auch eine bedeutende Quelle der Frustration für neue Teammitglieder darstellen.

## Indicators ⟡
- Neue Teammitglieder können lange Zeit nicht zum Team beitragen.
- Neue Teammitglieder stellen konstant dieselben Fragen.
- Es gibt viel doppelten Aufwand, da neue Teammitglieder dieselben Informationen neu entdecken müssen.
- Neue Teammitglieder können nicht unabhängig arbeiten.

## Symptoms ▲

- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Neue Teammitglieder bleiben über längere Zeiträume unproduktiv, was die Gesamt-Velocity des Teams verringert.
- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Verlängerte Einarbeitungszeiten für neue Mitarbeiter erhöhen die Kosten und den Aufwand, die zum Onboarding von Teammitgliedern erforderlich sind.
- [Verringerte Teamproduktivität](verringerte-teamproduktivitaet.md)
<br/>  Bestehende Teammitglieder werden von ihrer Arbeit abgezogen, um neuen Mitgliedern zu helfen, was die Gesamtproduktivität des Teams verringert.
- [Hohe Fluktuation](hohe-fluktuation.md)
<br/>  Frustrierende Onboarding-Erfahrungen aufgrund schlechten Wissenstransfers können neue Teammitglieder dazu bringen, zu gehen.
- [Wissenslücken](wissensluecken.md)
<br/>  Langsamer Transfer bedeutet, dass neue Mitglieder ein unvollständiges Verständnis entwickeln, was anhaltende Lücken in ihrem Wissen hinterlässt.

## Causes ▼

- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Ohne Dokumentation müssen neue Teammitglieder alles durch Nachfragen bei Kollegen lernen, was langsam und unzuverlässig ist.
- [Wissenssilos](wissenssilos.md)
<br/>  Wenn Wissen siloartig ist, kann es nur durch die spezifischen Personen weitergegeben werden, die es besitzen, was Engpässe schafft.
- [Implizites Erfahrungswissen](implizites-erfahrungswissen.md)
<br/>  Undokumentiertes Stammeswissen, das nur in den Köpfen von Personen existiert, ist inhärent schwierig und langsam weiterzugeben.
- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Code, der schwer zu verstehen ist, macht es für neue Teammitglieder viel schwieriger, das System unabhängig zu lernen.
- [Unzureichendes Onboarding](unzureichendes-onboarding.md)
<br/>  Unzureichende Onboarding-Prozesse verursachen direkt langsamen Wissenstransfer, da neuen Teammitgliedern strukturierte Wege zum Lernen fehlen.

## Detection Methods ○
- **Onboarding-Zeit:** Messung der Zeit, die neue Teammitglieder brauchen, um produktiv zu werden.
- **Entwicklerbefragungen:** Befragung neuer Teammitglieder zu ihrer Onboarding-Erfahrung.
- **Code-Reviews:** Suche nach Code, der von neuen Teammitgliedern geschrieben wurde und nicht den Coding-Standards des Teams entspricht.

## Examples
Ein Unternehmen stellt einen neuen Entwickler ein, um an einem Legacy-System zu arbeiten. Der Entwickler erhält einen Laptop und einen Link zur Codebasis, aber keine Dokumentation oder Schulung. Der Entwickler verbringt die ersten Wochen damit, herauszufinden, wie das System funktioniert. Er fragt konstant die anderen Entwickler um Hilfe, aber die anderen Entwickler sind mit ihrer eigenen Arbeit beschäftigt. Der neue Entwickler wird frustriert und kündigt schließlich. Dies ist ein häufiges Problem in Unternehmen ohne formalen Onboarding-Prozess.
