---
title: Property-Based Testing
description: Verifikation von Software durch zufällige Eingaben und
  Eigenschaften.
category:
- Testing
problems:
- insufficient-testing
- regression-bugs
- poor-test-coverage
- quality-blind-spots
- legacy-code-without-tests
- increased-risk-of-bugs
- integer-overflow-underflow
- null-pointer-dereferences
- race-conditions
layout: solution
lang: de
en_slug: property-based-testing
related_solutions:
- slug: mutation-testing
  similarity: 0.75
- slug: automated-tests
  similarity: 0.75
- slug: integration-tests
  similarity: 0.7
- slug: functional-tests
  similarity: 0.7
- slug: test-coverage-strategy
  similarity: 0.7
- slug: cross-version-testing
  similarity: 0.7
---

## Description

Property-Based Testing generiert eine große Anzahl zufälliger Eingaben und prüft, dass eine allgemeine Eigenschaft — eine Invariante, die für alle gültigen Eingaben gelten muss, wie Idempotenz, eine Roundtrip-Garantie oder eine Bereichsbeschränkung — wahr bleibt, statt spezifische erwartete Ausgaben für eine feste, handverlesene Menge von Beispieleingaben zu behaupten. Wenn eine Eigenschaft fehlschlägt, reduziert der Shrinking-Mechanismus des Frameworks die fehlschlagende Eingabe automatisch auf ihre kleinste reproduzierende Form, was ein zufälliges, möglicherweise großes Gegenbeispiel in einen minimalen, debugbaren Testfall verwandelt. Dies ist besonders effektiv für Legacy-Code, weil solcher Code häufig nur von einer Handvoll beispielbasierter Tests durchlaufen wird, die vor Jahren für die Szenarien geschrieben wurden, an die der ursprüngliche Autor zufällig dachte, was breite Bereiche des Eingaberaums — und die Grenzfälle und Randbedingungen, die die ursprünglichen Tests nie abdeckten — effektiv unverifiziert lässt. Da zufällige Eingabegenerierung aktiv nach Eingaben sucht, die die festgelegten Eigenschaften verletzen, legt sie routinemäßig Defekte offen, die jahrelang in Produktionscode vorhanden waren, ohne je durch die enge Menge manuell geschriebener Beispiele ausgelöst zu werden, wie ein Integer-Overflow in einem selten genutzten Codepfad. Die anfänglichen Kosten sind konzeptuell: Eigenschaften als universelle Aussagen über Verhalten zu artikulieren erfordert eine andere Denkweise als das Schreiben von Beispiel-Assertionen, und nicht jedes Stück Legacy-Code hat leicht formulierbare Eigenschaften, was einschränkt, wo die Technik ohne benutzerdefinierte Eingabegeneratoren für domänenspezifische Typen angewendet werden kann.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie reine Funktionen und Datentransformationen im Legacy-Code, die wohldefinierte Eigenschaften haben (z. B. Idempotenz, Umkehrbarkeit, Invarianten)
- Verwenden Sie ein für die Sprache geeignetes Property-Based-Testing-Framework (z. B. QuickCheck, jqwik, Hypothesis, fast-check)
- Definieren Sie Eigenschaften als universelle Wahrheiten über den Code statt als spezifische Eingabe-Ausgabe-Paare
- Beginnen Sie mit Serialisierungs-/Deserialisierungs-Roundtrip-Tests und mathematischen Eigenschaften als leichte Erfolge
- Nutzen Sie Shrinking-Fähigkeiten, um automatisch die minimale fehlschlagende Eingabe zu finden, wenn eine Eigenschaftsverletzung entdeckt wird
- Kombinieren Sie Property-Based-Tests mit traditionellen beispielbasierten Tests für umfassende Abdeckung

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Entdeckt Grenzfälle und Randbedingungen, an deren manuelles Testen Entwickler nicht denken würden
- Bietet breitere Abdeckung als handgeschriebene Beispieltests mit weniger zu pflegenden Testfällen
- Shrinking produziert automatisch minimale Reproduktionsfälle und vereinfacht Debugging
- Zwingt Entwickler, über Invarianten und Verträge nachzudenken statt über spezifische Szenarien

**Kosten und Risiken:**
- Gute Eigenschaften zu schreiben erfordert eine andere Denkweise und kann für Teams anfangs herausfordernd sein
- Zufallsgenerierung produziert möglicherweise keine relevanten Eingaben ohne benutzerdefinierte Generatoren für Domänentypen
- Instabile Ergebnisse können auftreten, wenn Eigenschaften nicht deterministisch sind oder Seed-Verwaltung vernachlässigt wird
- Nicht der gesamte Legacy-Code hat leicht ausdrückbare Eigenschaften, was die Anwendbarkeit einschränkt

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Finanzdienstleistungsanwendung hatte ein Währungsumrechnungsmodul mit handgeschriebenen Tests, die ein Dutzend spezifischer Währungspaare abdeckten. Property-Based Testing wurde mit Eigenschaften wie „die Umrechnung von A nach B und zurück nach A sollte den ursprünglichen Betrag innerhalb der Rundungstoleranz zurückgeben" und „Umrechnungskurse sollten immer positiv sein" eingeführt. Der Zufallsgenerator fand sofort einen Fall, in dem die Umrechnung zwischen zwei selten genutzten Währungen aufgrund eines Integer-Overflows in einer Zwischenberechnung einen negativen Betrag produzierte. Dieser Fehler war jahrelang vorhanden gewesen, wurde aber nie durch die spezifischen Testfälle ausgelöst, die das Team geschrieben hatte.
