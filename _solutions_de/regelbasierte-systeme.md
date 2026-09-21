---
title: Regelbasierte Systeme
description: Definition von Regeln, die das Verhalten der Software steuern.
category:
- Architecture
- Code
problems:
- complex-and-obscure-logic
- legacy-business-logic-extraction-difficulty
- difficult-code-comprehension
- hardcoded-values
- spaghetti-code
- poor-domain-model
- maintenance-overhead
layout: solution
lang: de
en_slug: rule-based-systems
related_solutions:
- slug: decision-tables
  similarity: 0.8
- slug: domain-specific-languages
  similarity: 0.7
- slug: incremental-refactoring
  similarity: 0.7
- slug: static-analysis-and-linting
  similarity: 0.7
- slug: architecture-decision-records
  similarity: 0.7
- slug: domain-patterns
  similarity: 0.7
---

## Description

Ein regelbasiertes System extrahiert Geschäftslogik, die sonst in tief verschachtelten Bedingungen, ausufernden Switch-Anweisungen oder prozeduralem Code vergraben wäre, in eine explizite Sammlung diskreter Regeln, jede ausgedrückt als klares Bedingung-und-Aktion-Paar, typischerweise ausgewertet durch eine dedizierte Regel-Engine statt inline im Anwendungscode eingebettet zu sein. Dies macht die Logik nach eigenen Bedingungen lesbar, unabhängig von der Struktur des umgebenden Codes, und — abhängig vom gewählten Regelformat — potenziell überprüfbar oder sogar bearbeitbar von Fachexperten statt nur von Entwicklern, die die ursprüngliche Implementierung lesen können. Die Technik ist besonders wertvoll bei der Modernisierung von Legacy-Systemen, deren Geschäftslogik sich über viele Jahre hinweg über Tausende von Zeilen prozeduralen Codes angesammelt hat, da diese Logik häufig die einzelne größte Quelle sowohl von Risiko als auch von Wert im System ist: Risiko, weil niemand vollständig versteht, was sie alles tut oder warum; und Wert, weil sie Jahre angesammelter Geschäftsentscheidungen, regulatorischer Anpassungen und Grenzfallbehandlung kodiert, die nicht einfach verworfen werden können. Diese Logik in explizite Regeln zu extrahieren, wobei Fachexperten jede gegen ihr Verständnis des Geschäfts validieren, deckt oft Regeln auf, deren ursprünglicher Zweck vergessen wurde, Regeln, die nun miteinander in Konflikt stehen, und Regeln, die aktuelle Vorschriften nicht mehr widerspiegeln — Entdeckungen, die selbst wertvolle Eingaben für den Modernisierungsaufwand sind. Da die Extraktion inkrementell, Regel für Regel, fortschreiten kann, ohne eine Big-Bang-Neuschreibung des umgebenden Systems zu erfordern, bietet sie einen praktischen Pfad, verwickelte Legacy-Geschäftslogik zu entwirren, die eine vollständige Neuarchitektur unerschwinglich riskant machen würde.

## How to Apply ◆

> In Legacy-Systemen macht die Extraktion verwickelter Geschäftslogik in explizite Regeln Verhalten sichtbar, testbar und von Fachexperten modifizierbar, statt tiefe Code-Archäologie zu erfordern.

- Identifizieren Sie Geschäftslogik im Legacy-System, die als tief verschachtelte Bedingungen, ausufernde Switch-Anweisungen oder prozeduraler Code, gemischt mit Infrastrukturbelangen, implementiert ist.
- Extrahieren Sie diese Entscheidungspunkte in eine Regel-Engine oder ein deklaratives Regelformat, wo jede Regel eine klare Bedingung und Aktion hat, was die Logik lesbar macht, ohne den umgebenden Code zu verstehen.
- Beziehen Sie Fachexperten in die Validierung extrahierter Regeln gegen ihr Verständnis des Geschäfts ein, da Legacy-Code oft Regeln enthält, deren ursprünglicher Zweck vergessen wurde.
- Implementieren Sie Regeln in einem Format, das Nicht-Entwicklern erlaubt, sie zu überprüfen und potenziell zu modifizieren — dies reduziert den Engpass, für jede Geschäftsregeländerung Entwicklereingriff zu erfordern.
- Fügen Sie umfassende Tests für jede extrahierte Regel isoliert hinzu, dann testen Sie Regelinteraktionen, um Konflikte oder Lücken zu erfassen.
- Pflegen Sie einen Regelkatalog, der Herkunft, Zweck und Begründung für jede Regel dokumentiert und künftigen Wissensverlust verhindert.

## Tradeoffs ⇄

> Regelbasierte Systeme machen Geschäftslogik explizit und wartbar, führen aber eine neue Komplexitätsschicht ein, die gemanagt werden muss.

**Vorteile:**

- Macht Geschäftslogik für Fachexperten sichtbar und verständlich, die die Legacy-Codebasis nicht lesen können.
- Ermöglicht Geschäftsregeländerungen ohne Modifikation des Anwendungscodes, was die Änderungszykluszeit für regulatorische oder richtlinienbezogene Updates reduziert.
- Vereinfacht Testing, indem individuelle Regeln unabhängig verifiziert werden können.
- Unterstützt schrittweise Extraktion von Logik aus dem Legacy-System — Regeln können inkrementell migriert werden, ohne eine Big-Bang-Neuschreibung.

**Kosten und Risiken:**

- Regel-Engines führen eine neue Technologieabhängigkeit ein und erfordern Team-Expertise, um sie effektiv zu managen.
- Komplexe Regelinteraktionen können emergentes Verhalten erzeugen, das schwer vorherzusagen und zu debuggen ist, besonders wenn Hunderte von Regeln interagieren.
- Der Performance-Overhead der Regelauswertung kann für Systeme mit großen, in Echtzeit ausgeführten Regelmengen erheblich sein.
- Übermäßig enthusiastische Übernahme kann dazu führen, Logik in Regeln zu verschieben, die besser in konventionellem Code ausgedrückt würde, was das System schwerer verständlich macht.

## How It Could Be

> Das folgende Szenario zeigt, wie regelbasierte Extraktion die Komplexität von Legacy-Geschäftslogik zähmt.

Eine Krankenversicherung hatte ein Schadensregulierungssystem, in dem Preislogik über 50.000 Zeilen COBOL-Code verstreut war, mit Hunderten verschachtelter IF-ELSE-Blöcke, die verschiedene Plantypen, Anbieternetzwerke und regulatorische Ausnahmen repräsentierten. Das Team extrahierte diese Entscheidungen in eine moderne Regel-Engine und erstellte ungefähr 800 einzelne, nach Geschäftsdomäne organisierte Regeln. Zum ersten Mal konnte das Compliance-Team Regeländerungen direkt überprüfen und genehmigen, statt sich darauf zu verlassen, dass Entwickler den COBOL interpretierten. Die Extraktion offenbarte außerdem 34 Regeln, die miteinander im Konflikt standen, und 12, die aufgrund regulatorischer Änderungen, die nie vollständig im Code widergespiegelt worden waren, nicht mehr anwendbar waren. Das regelbasierte System reduzierte die Zeit zur Implementierung jährlicher regulatorischer Updates von drei Monaten auf drei Wochen.
