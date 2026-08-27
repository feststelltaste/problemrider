---
title: Definition von Wertebereichen
description: Definition akzeptabler Wertebereiche für Eingaben und
  Ausgaben.
category:
- Code
- Testing
problems:
- inadequate-error-handling
- inconsistent-behavior
- unpredictable-system-behavior
- silent-data-corruption
- hardcoded-values
- regression-bugs
- increased-risk-of-bugs
- integer-overflow-underflow
layout: solution
lang: de
en_slug: value-range-definition
related_solutions:
- slug: input-validation
  similarity: 0.75
- slug: input-constraints-and-defaults
  similarity: 0.7
- slug: plausibility-checks
  similarity: 0.7
- slug: negative-testing
  similarity: 0.7
- slug: data-quality-checks
  similarity: 0.65
- slug: transactions
  similarity: 0.65
---

## Description

Die Definition von Wertebereichen macht die akzeptablen Grenzen für ein Eingabe- oder Ausgabefeld zu einer expliziten, durchgesetzten Regel — ein Minimum, ein Maximum, ein Satz gültiger Zustände — statt zu einer impliziten Annahme, die nur in welchem Verhalten auch immer der Code zufällig zeigt, lebt. In Legacy-Systemen sind diese Grenzen häufig nicht durchgesetzt oder inkonsistent über verschiedene Eintrittspunkte hinweg durchgesetzt, sodass dasselbe Feld in einem Bildschirm strikt validiert werden könnte, aber durch einen Batch-Import oder ein direktes Datenbank-Update ungeprüft akzeptiert wird, ohne einen einzigen Ort, an dem die tatsächliche Regel niedergeschrieben ist. Den Bereich explizit zu machen erfordert zunächst zu beobachten, welche Werte Produktionsdaten tatsächlich enthalten, da Dokumentation oft veraltet oder fehlend ist, und dann Validierung an jeder Grenze zu kodifizieren, durch die der Wert eintreten kann, sodass ungültige Daten am Rand abgelehnt werden, statt nach innen zu propagieren, wo sie zunehmend schwerer zu ihrem Ursprung zurückzuverfolgen werden. Dies zählt besonders für Modernisierungsanstrengungen, wo die stille Toleranz eines Legacy-Systems für technisch ungültige Werte — negative Bestandsmengen, die als informeller Rückstands-Mechanismus genutzt werden, zum Beispiel — sich oft als Kodierung einer undokumentierten Geschäftspraxis herausstellt, die ein strikteres Ersatzsystem rundweg ablehnen wird, es sei denn, diese Praxis wird bewusst modelliert statt nur blockiert. Wertebereiche richtig zu machen erfordert daher, zwischen Werten zu unterscheiden, die tatsächlich ungültig sind, und Werten, die eine echte, wenn auch undokumentierte, Geschäftsregel widerspiegeln, die das Legacy-System still berücksichtigte.

## How to Apply ◆

> In Legacy-Systemen werden Wertebereiche oft inkonsistent oder gar nicht durchgesetzt — sie explizit zu machen verhindert Datenbeschädigung und erfasst Integrationsfehler an Systemgrenzen.

- Prüfen Sie das Legacy-System, um alle Eingabe- und Ausgabewerte zu identifizieren, und dokumentieren Sie die in Produktionsdaten tatsächlich beobachteten Bereiche, statt sich auf potenziell veraltete Dokumentation zu verlassen.
- Definieren Sie explizite Validierungsregeln für jede Systemgrenze — API-Endpunkte, Nutzeroberflächen, Datei-Importe, Datenbankschreibvorgänge —, die Werte außerhalb akzeptabler Bereiche mit klaren Fehlermeldungen ablehnen.
- Achten Sie besonders auf Werte, die das Legacy-System still akzeptierte, aber falsch behandelte, wie negative Beträge in Feldern, die nur positiv sein sollten, oder Daten außerhalb geschäftlich bedeutsamer Bereiche.
- Implementieren Sie Validierung so nah wie möglich an der Systemgrenze, um zu verhindern, dass ungültige Daten durch das System propagieren.
- Erstellen Sie umfassende Testfälle für Grenzwerte, einschließlich Minimum, Maximum, knapp innerhalb, knapp außerhalb und Null-/Leer-Werte für jeden definierten Bereich.
- Dokumentieren Sie Wertebereichsentscheidungen und ihre Begründung, besonders wenn sich die Bereiche des Ersatzsystems vom impliziten Verhalten des Legacy-Systems unterscheiden.

## Tradeoffs ⇄

> Explizite Wertebereiche verhindern Datenbeschädigung und klären Systemverhalten, erfordern aber Aufwand zur korrekten Definition und könnten Daten ablehnen, die das Legacy-System akzeptierte.

**Vorteile:**

- Verhindert stille Datenbeschädigung, indem ungültige Werte an Systemgrenzen erfasst werden, bevor sie in die Verarbeitungspipeline eintreten.
- Macht Systemverhalten vorhersagbar und dokumentiert, statt sich auf implizite Annahmen über gültige Daten zu verlassen.
- Vereinfacht Debugging, indem sichergestellt wird, dass interne Verarbeitung nur Werte innerhalb bekannt-guter Bereiche handhabt.
- Bietet klare Dokumentation von Systembeschränkungen für API-Konsumenten und Integrationspartner.

**Kosten und Risiken:**

- Die Definition von Bereichen für ein Legacy-System mit Jahren angesammelter Daten könnte offenbaren, dass Produktionsdaten bereits außerhalb des Bereichs liegende Werte enthalten, die bereinigt oder mit Bestandsschutz versehen werden müssen.
- Übermäßig restriktive Bereiche können legitime Daten ablehnen, die das Legacy-System akzeptierte, was Nutzerfrustration und Workflow-Störung verursacht.
- Die Pflege von Wertebereichsdefinitionen erfordert laufende Governance, während sich Geschäftsregeln ändern.
- Validierungslogik kann über die Codebasis verstreut werden, wenn sie nicht in einer Validierungsschicht zentralisiert ist.

## How It Could Be

> Das folgende Szenario zeigt, wie die Definition von Wertebereichen Datenqualitätsprobleme während der Legacy-Migration verhindert.

Ein Lieferkettenunternehmen entdeckte während der Migration, dass sein Legacy-Bestandssystem negative Bestandsmengen akzeptierte, die 10 Jahre lang als informeller Rückstands-Mechanismus genutzt worden waren. Die strikte Nicht-negativ-Validierung des neuen Systems lehnte diese Einträge anfänglich ab, was den Bestellworkflow brach. Durch die Definition einer expliziten Wertebereichsrichtlinie — Bestandsmengen müssen nicht-negativ sein, mit einem separaten Rückstandsmengenfeld — bewahrte das Team die Geschäftsfähigkeit, während die Datenmehrdeutigkeit beseitigt wurde, die jahrelang Berichtsfehler verursacht hatte. Die Migration erforderte einen Datenbereinigungsschritt, der 8.000 negative Bestandsdatensätze in ordentliche Rückstandseinträge konvertierte, was Bestandsberichte endlich akkurat machte.
