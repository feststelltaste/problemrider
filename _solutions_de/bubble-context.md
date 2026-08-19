---
title: Bubble Context
description: Klare Abgrenzung von Erweiterungen gegenüber bestehenden Code-Teilen.
category:
- Architecture
- Code
problems:
- legacy-business-logic-extraction-difficulty
- high-coupling-low-cohesion
- spaghetti-code
- fear-of-change
- brittle-codebase
- monolithic-architecture-constraints
- inconsistent-codebase
- strangler-fig-pattern-failures
layout: solution
lang: de
en_slug: bubble-context
related_solutions:
- slug: domain-patterns
  similarity: 0.7
- slug: strangler-fig-pattern
  similarity: 0.7
- slug: modularization-and-bounded-contexts
  similarity: 0.7
- slug: separation-of-concerns
  similarity: 0.7
- slug: anti-corruption-layer
  similarity: 0.7
- slug: facades
  similarity: 0.7
---

## Description

Ein Bubble Context ist ein bewusst isolierter Bereich neuen Codes, mit einem eigenen sauberen Domänenmodell und modernen Codierungskonventionen, verbunden mit dem umgebenden Legacy-System nur durch explizite Adapter- oder Übersetzungsklassen, die an der Grenze der Blase platziert sind. Der Mechanismus funktioniert, indem er sich weigert, das Datenmodell, die Namenskonventionen oder die Architekturmuster des Legacy-Systems in den neuen Code durchsickern zu lassen — alles, was die Blase betritt oder verlässt, wird an der Grenze übersetzt, sodass Entwickler innerhalb der Blase so arbeiten können, als hätte das Legacy-System nicht die Probleme, die es hat. Dies ist spezifisch wertvoll, weil eine vollständige Neuschreibung eines Legacy-Systems selten in einem Schritt machbar ist, während der Bau neuer Features direkt im Legacy-Code dazu tendiert, diese Features zu zwingen, das denormalisierte Datenmodell und die inkonsistenten Muster des Legacy-Systems zu erben, einfach weil das bereits vorhanden ist, worauf aufgebaut werden kann. Der Bubble Context erlaubt es einem Team, neue Funktionalität sofort mit aktuellen Praktiken hinzuzufügen, ohne auf den Abschluss einer breiteren Modernisierungsbemühung zu warten, und schafft einen natürlichen Keim, der sich schrittweise ausdehnen kann — während mehr Funktionalität in die Blase wandert, wächst ihre Grenze effektiv, bis sie schließlich das Legacy-System ersetzen kann, mit dem sie einst nur koexistierte. Die Kosten sind die Übersetzungsschicht selbst: Sie muss gepflegt werden, während sich beide Modelle weiterentwickeln, und wenn mehrere Blasen inkonsistent über ein System hinweg eingeführt werden, kann das Ergebnis ein Flickwerk sein, das schwerer zu verstehen ist, als es entweder eine vollständig alte oder vollständig moderne Architektur allein gewesen wäre.

## How to Apply ◆

> In Legacy-Systemen schafft Bubble Context eine saubere Grenze um neuen Code und verhindert, dass er durch Legacy-Muster kontaminiert wird, während er mit dem alten System koexistiert.

- Definieren Sie eine klare Grenze (die „Blase") um neue Funktionalität, mit expliziten Übersetzungsschichten an jedem Punkt, an dem neuer Code mit Legacy-Code interagieren muss.
- Implementieren Sie neue Features mit modernen Praktiken und Mustern innerhalb der Blase, ohne durch die Codierungskonventionen, Datenmodelle oder Architekturmuster des Legacy-Systems eingeschränkt zu sein.
- Erstellen Sie Adapterklassen oder -module an der Blasengrenze, die zwischen den Datenformaten des Legacy-Systems und dem internen Modell der Blase übersetzen.
- Nutzen Sie das Bubble-Context-Muster beim Hinzufügen neuer Features zu einem Legacy-System, das noch nicht vollständig zerlegt werden kann — es erlaubt schrittweise Verbesserung, ohne eine vollständige Neuschreibung zu versuchen.
- Halten Sie das interne Modell der Blase sauber und domänengetrieben, selbst wenn das Modell des Legacy-Systems denormalisiert, inkonsistent oder schlecht benannt ist.
- Testen Sie die Blase unabhängig vom Legacy-System, indem Sie die Adapter als Nähte für Test Doubles nutzen, was sicherstellt, dass neuer Code ohne Legacy-System-Abhängigkeiten verifiziert werden kann.

## Tradeoffs ⇄

> Bubble Context ermöglicht saubere neue Entwicklung innerhalb von Legacy-Einschränkungen, schafft aber ein Zwei-Modell-System, das verwaltet werden muss.

**Vorteile:**

- Erlaubt den Bau neuer Features mit modernen Praktiken, ohne auf die vollständige Modernisierung des gesamten Legacy-Systems zu warten.
- Verhindert „Legacy-Kontamination", bei der neuer Code die schlechten Muster des Legacy-Systems übernimmt, einfach weil er mit ihnen interagieren muss.
- Schafft natürliche Migrationsgrenzen — die Blase kann schließlich wachsen, um das Legacy-System zu ersetzen, während mehr Funktionalität in sie wandert.
- Ermöglicht dem Team, neue Features schneller zu entwickeln und zu testen, indem sie von Legacy-Komplexität isoliert werden.

**Kosten und Risiken:**

- Die Übersetzungsschicht an der Blasengrenze fügt Komplexität hinzu und muss gepflegt werden, während sich sowohl das Legacy-System als auch die Blase weiterentwickeln.
- Mehrere Blasen innerhalb desselben Legacy-Systems können eine Flickwerk-Architektur schaffen, die schwerer zu verstehen ist als entweder ein reines Legacy- oder ein reines modernes System.
- Teams könnten sich uneinig sein, wo Blasengrenzen gezogen werden sollten, was zu inkonsistenter Anwendung des Musters führt.
- Das saubere Modell der Blase und das Legacy-Modell können auseinanderdriften, was die Übersetzungsschicht über die Zeit zunehmend komplex macht.

## How It Could Be

> Das folgende Szenario veranschaulicht, wie Bubble Context saubere Entwicklung innerhalb einer Legacy-Codebasis ermöglicht.

Ein Energieunternehmen musste ein neues Echtzeit-Preisgestaltungsfeature zu seinem Legacy-Abrechnungssystem hinzufügen. Das Legacy-System nutzte ein flaches relationales Modell mit 200 Zeichen langen Spaltennamen nach einer Namenskonvention aus den 1990ern, und die gesamte Geschäftslogik war in Stored Procedures eingebettet. Statt das Preisgestaltungsfeature im selben Stil zu bauen, erstellte das Team einen Bubble Context mit einem sauberen Domänenmodell (unter Nutzung von Klassen wie `PricingPlan`, `RateSchedule` und `ConsumptionTier`) und modernen Codemustern. Adapter an der Grenze übersetzten zwischen dem `CUST_RATE_SCHED_EFF_DT`-Format der Legacy-Datenbank und dem `RateSchedule.effectiveDate` der Blase. Das Preisgestaltungsfeature wurde in der Hälfte der Zeit entwickelt und getestet, die es mit Legacy-Mustern gebraucht hätte, und die saubere Architektur der Blase diente später als Grundlage für den Ersatz weiterer Abrechnungsmodule.
