---
title: Specification by Example
description: Kollaborative Definition von Anforderungen durch konkrete
  Beispiele, die zu ausführbaren Spezifikationen werden.
category:
- Requirements
- Testing
problems:
- requirements-ambiguity
- inadequate-requirements-gathering
- misaligned-deliverables
- implementation-rework
- insufficient-testing
- stakeholder-developer-communication-gap
- legacy-code-without-tests
- inconsistent-behavior
- reduced-feature-quality
- frequent-changes-to-requirements
layout: solution
lang: de
en_slug: specification-by-example
related_solutions:
- slug: behavior-driven-development-bdd
  similarity: 0.85
- slug: evolutionary-requirements-development
  similarity: 0.75
- slug: user-stories
  similarity: 0.75
- slug: acceptance-tests
  similarity: 0.75
- slug: requirements-analysis
  similarity: 0.75
- slug: living-documentation
  similarity: 0.7
---

## Description

Specification by Example ist eine kollaborative Praxis, bei der Entwickler, Tester und Fachexperten das erwartete Systemverhalten durch konkrete, reale Eingabe-Ausgabe-Paare definieren, statt durch abstrakte Prosa-Anforderungen, und diese Beispiele dann in einem strukturierten, automatisierbaren Format ausdrücken, sodass sie gleichzeitig als ausführbare Tests dienen. Die Technik adressiert direkt eines der schwierigsten Probleme in der Legacy-Modernisierung: Die ursprünglichen Anforderungen wurden oft nie dokumentiert, die Personen, die das System geschrieben haben, sind nicht mehr da, und die einzige verbleibende Wahrheitsquelle dafür, was das System eigentlich tun soll, ist sein eigenes beobachtetes Verhalten — gegen das das Legacy-System selbst laufen gelassen werden kann, um den anfänglichen Satz von Beispielen zu generieren. Da jedes Beispiel konkret statt abstrakt ist, beseitigt es die Mehrdeutigkeit, die dazu führt, dass Ersatzsysteme subtil vom Legacy-Verhalten bei Grenzfällen abweichen, die niemand aufzuschreiben dachte, und da die Beispiele automatisiert sind, können sie gleichzeitig gegen das alte und das neue System laufen, um ein direktes, kontinuierlich verifizierbares Maß für die Migrationsparität zu produzieren. Der resultierende Bestand an Beispielen wird zu lebender Dokumentation, die das Legacy-System selbst überdauert und Geschäftsregeln erfasst, die nur als implizites Verhalten existierten. Die Praxis erfordert regelmäßigen, anhaltenden Zugang zu Fachexperten, die die Eigenheiten des Legacy-Systems verstehen, und die richtige Granularität von Beispielen zu finden — genug, um bedeutsame Grenzfälle abzudecken, ohne zu einer unhandhabbaren, brüchigen Masse zu werden — ist eine Fähigkeit, die Iteration braucht, um sich zu entwickeln.

## How to Apply ◆

> In der Legacy-Modernisierung überbrückt Specification by Example die Lücke zwischen undokumentiertem Legacy-Verhalten und klar definierten Ersatzanforderungen, indem konkrete Beispiele als gemeinsame Sprache zwischen Fachexperten und Entwicklern genutzt werden.

- Führen Sie kollaborative Spezifikations-Workshops durch, in denen Entwickler, Tester und Fachexperten gemeinsam das erwartete Systemverhalten durch konkrete Eingabe-Ausgabe-Beispiele aus echter Legacy-Systemnutzung definieren.
- Nutzen Sie das Legacy-System selbst, um Beispiele zu generieren — führen Sie repräsentative Szenarien durch das alte System und protokollieren Sie die Ergebnisse als anfängliche Spezifikation für den Ersatz.
- Drücken Sie Beispiele in einem strukturierten Format aus (wie Given-When-Then), das als ausführbare Tests automatisiert werden kann, um sicherzustellen, dass Spezifikationen während der gesamten Modernisierung verifizierbar bleiben.
- Fokussieren Sie Beispiele auf Geschäftsregeln und Grenzfälle, wo Legacy-Verhalten am komplexesten oder am wenigsten dokumentiert ist, da dies die Bereiche sind, die am wahrscheinlichsten Defekte während des Ersatzes verursachen.
- Automatisieren Sie die Beispiele als Akzeptanztests, die sowohl gegen das Legacy-System (zur Korrektheitsverifikation) als auch gegen das neue System (zur Paritätsverifikation) laufen, was ein klares Maß für den Migrationsfortschritt liefert.
- Pflegen Sie ein lebendes Dokumentations-Repository, in dem Beispiele nach Geschäftsfähigkeit organisiert sind und sowohl als Spezifikation als auch als Testsuite dienen.

## Tradeoffs ⇄

> Specification by Example schafft Ausrichtung und lebende Dokumentation, erfordert aber anhaltende Zusammenarbeit zwischen technischen und geschäftlichen Stakeholdern.

**Vorteile:**

- Beseitigt Mehrdeutigkeit in Anforderungen, indem abstrakte Beschreibungen durch konkrete, verifizierbare Beispiele ersetzt werden, die jeder verstehen kann.
- Schafft ausführbare Tests als Nebenprodukt des Spezifikationsprozesses und stellt sicher, dass sich das Ersatzsystem von Anfang an korrekt verhält.
- Bewahrt kritisches Geschäftswissen, das nur im Verhalten des Legacy-Systems existiert, und erfasst es in einem Format, das das alte System überdauert.
- Bietet eine klare, messbare Definition von "fertig" für jedes migrierte Feature — die Beispiele bestehen entweder oder nicht.

**Kosten und Risiken:**

- Erfordert regelmäßigen Zugang zu Fachexperten, die das Legacy-Systemverhalten verstehen, was schwierig zu sichern sein kann.
- Das Workshop-Format kann zeitaufwändig sein, besonders bei der Spezifikation komplexen Legacy-Verhaltens mit vielen Grenzfällen.
- Zu detaillierte Beispiele können zu brüchigen Tests werden, die bei kleineren Implementierungsänderungen brechen.
- Teams könnten Schwierigkeiten haben, die richtige Abstraktionsebene zu finden — zu wenige Beispiele übersehen kritische Grenzfälle, während zu viele unhandhabbar werden.

## How It Could Be

> Das folgende Szenario demonstriert Specification by Example während einer Legacy-Systemmigration.

Ein Lohnabrechnungsunternehmen ersetzte sein Legacy-System, das Steuerberechnungen für 12 verschiedene Rechtsordnungen handhabte. Statt zu versuchen, traditionelle Anforderungsdokumente für die Tausenden von Steuerregeln zu schreiben, hielt das Team wöchentliche Spezifikations-Workshops mit Lohnsteuerspezialisten ab. In jeder Sitzung lieferten die Spezialisten konkrete Lohnabrechnungsszenarien — spezifische Mitarbeiter, spezifische Lohnperioden, spezifische Abzugskombinationen — und gingen die erwarteten Berechnungen Schritt für Schritt durch. Diese Beispiele wurden als ausführbare Spezifikationen automatisiert, die sowohl gegen das Legacy-System als auch gegen die neue Implementierung liefen. Als die Spezifikationen unterschiedliche Ergebnisse zwischen den Systemen produzierten, untersuchte das Team, ob die Diskrepanz ein Legacy-Fehler oder ein Migrationsdefekt war. Über acht Monate sammelte das Team 2.400 ausführbare Beispiele an, die sowohl als Spezifikation als auch als Regressionstestsuite für die gesamte Migration dienten.
