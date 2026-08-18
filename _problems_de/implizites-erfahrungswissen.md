---
title: Implizites Erfahrungswissen
description: Wissen, das schwer auf eine andere Person zu übertragen ist, indem man
  es aufschreibt oder verbalisiert.
category:
- Process
- Team
related_problems:
- slug: implicit-knowledge
  similarity: 0.7
- slug: slow-knowledge-transfer
  similarity: 0.65
- slug: knowledge-silos
  similarity: 0.65
- slug: knowledge-dependency
  similarity: 0.6
- slug: knowledge-gaps
  similarity: 0.55
- slug: knowledge-sharing-breakdown
  similarity: 0.55
solutions:
- architecture-decision-records
- documentation-as-code
- knowledge-sharing-practices
- pair-and-mob-programming
- architecture-documentation
- code-comments
- living-documentation
- knowledge-rotation
layout: problem
lang: de
en_slug: tacit-knowledge
---

## Description
Implizites Erfahrungswissen ist Wissen, das schwer auf eine andere Person zu übertragen ist, indem man es aufschreibt oder verbalisiert. Es wird oft durch Erfahrung gelernt und ist oft schwer zu artikulieren. Implizites Erfahrungswissen kann ein bedeutendes Problem in der Softwareentwicklung sein, da es zu Wissenssilos und einer Verlangsamung des Wissenstransfers führen kann.

## Indicators ⟡
- Es gibt viel Wissen, das nicht aufgeschrieben ist.
- Neue Teammitglieder haben Schwierigkeiten, sich einzuarbeiten.
- Das Team ist stark auf wenige Senior-Entwickler angewiesen.
- Es gibt viel doppelten Aufwand, da neue Teammitglieder dieselben Informationen neu entdecken müssen.

## Symptoms ▲

- [Wissenssilos](wissenssilos.md)
<br/>  Wenn Wissen nicht leicht übertragen werden kann, bleibt es bei bestimmten Personen isoliert, die es besitzen.
- [Langsamer Wissenstransfer](langsamer-wissenstransfer.md)
<br/>  Wissen, das schwer zu artikulieren ist, braucht viel länger, um durch lehrlingsähnliches Lernen an neue Teammitglieder weitergegeben zu werden.
- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Neue Teammitglieder haben Schwierigkeiten, produktiv zu werden, wenn kritisches Systemwissen nur in den Köpfen erfahrener Entwickler existiert.
- [Engpassbildung](engpassbildung.md)
<br/>  Träger impliziten Wissens werden zu Engpässen, während andere auf ihre Anleitung bei Entscheidungen und Änderungen warten müssen.
- [Verringerte Teamproduktivität](verringerte-teamproduktivitaet.md)
<br/>  Die Teamproduktivität leidet, wenn nur wenige Mitglieder aufgrund ungeteilten impliziten Wissens effektiv an Teilen des Systems arbeiten können.
- [Implizites Wissen](implizites-wissen.md)
<br/>  Implizites Erfahrungswissen, das nicht leicht artikuliert werden kann, wird zu implizitem Wissen, das in Teampraktiken und Annahmen eingebettet ist.

## Causes ▼

- [Zusammenbruch des Wissensaustauschs](zusammenbruch-des-wissensaustauschs.md)
<br/>  Ineffektive Praktiken des Wissensaustauschs versäumen es, implizites Erfahrungswissen in dokumentierte, übertragbare Formen zu externalisieren.
- [Fehlende Eigenverantwortung und Rechenschaftspflicht](fehlende-eigenverantwortung-und-rechenschaftspflicht.md)
<br/>  Ohne Rechenschaftspflicht für Dokumentation und Wissenstransfer wird implizites Erfahrungswissen nie formalisiert.
- [Informationsverfall](informationsverfall.md)
<br/>  Selbst wenn Wissen dokumentiert ist, zwingt Dokumentation, die veraltet, Entwickler dazu, sich stattdessen auf implizites Erfahrungswissen zu verlassen.
- [Komplexe und unklare Logik](komplexe-und-unklare-logik.md)
<br/>  Code, der schwer zu verstehen ist, kann nicht als Dokumentation dienen, was Verständnis dazu zwingt, implizit in den Köpfen von Entwicklern zu bleiben.
- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Schlechte Dokumentation zwingt Wissen direkt dazu, implizit zu bleiben.

## Detection Methods ○
- **Entwicklerbefragungen:** Befragung von Entwicklern, ob sie das Gefühl haben, das nötige Wissen für ihre Arbeit zu haben.
- **Code-Reviews:** Suche nach Code, der schwer zu verstehen ist.
- **Pair Programming:** Nutzung von Pair Programming, um neuen Teammitgliedern zu helfen, das implizite Erfahrungswissen des Teams zu lernen.

## Examples
Ein Unternehmen hat ein Legacy-System, das von einem einzigen Entwickler geschrieben wurde, der das Unternehmen inzwischen verlassen hat. Der Entwickler schrieb keine Dokumentation, sodass das gesamte Wissen über das System implizites Erfahrungswissen ist. Die neuen Entwickler, die für die Wartung des Systems verantwortlich sind, haben große Schwierigkeiten, sich einzuarbeiten. Sie machen ständig Fehler, und die Anzahl der Bugs nimmt zu. Dies ist ein häufiges Problem in Unternehmen ohne eine Kultur des Wissensaustauschs.
