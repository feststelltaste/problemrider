---
title: Domänenmodellierung
description: Abbildung von Domänenkonzepten und -beziehungen in einem Domänenmodell.
category:
- Architecture
- Requirements
problems:
- poor-domain-model
- complex-domain-model
- legacy-business-logic-extraction-difficulty
- architectural-mismatch
- requirements-ambiguity
- stakeholder-developer-communication-gap
- over-reliance-on-utility-classes
layout: solution
lang: de
en_slug: domain-modeling
related_solutions:
- slug: domain-patterns
  similarity: 0.8
- slug: domain-driven-design
  similarity: 0.8
- slug: data-modeling
  similarity: 0.8
- slug: domain-aligned-architecture
  similarity: 0.8
- slug: ubiquitous-language
  similarity: 0.75
- slug: event-storming
  similarity: 0.75
---

## Description

Domänenmodellierung erzeugt eine explizite Repräsentation — Diagramme, CRC-Karten oder ähnliche Artefakte — der Konzepte, Attribute und Beziehungen der Geschäftsdomäne, gemeinsam mit Domänenexperten gebaut und bewusst unabhängig davon, wie das System sie derzeit implementiert. Ihr diagnostischer Wert in Legacy-Systemen kommt daher, dieses unabhängig abgeleitete Modell mit den tatsächlichen Datenstrukturen und der Code-Organisation des Systems zu vergleichen: Legacy-Code nutzt häufig generische technische Abstraktionen wie „Record" oder „Transaction", die sich über die Zeit angehäuft haben, während das Geschäft selbst in Begriffen spezifischer, bedeutungsvoller Konzepte wie „Bestellung" oder „Sendung" denkt, und der Vergleich macht diese Divergenz sichtbar und konkret, statt eines vagen Gefühls, dass „der Code nicht ganz zum Geschäft passt". Diese Sichtbarkeit ist das, was Domänenmodellierung für Modernisierung handlungsfähig macht — sie beschreibt nicht nur die Domäne, sie lokalisiert präzise, wo und wie die Implementierung von ihr abgedriftet ist, was dann leitet, wo Refactoring-Aufwand zuerst investiert werden sollte. Ein so gebautes Domänenmodell wird auch zu einem gemeinsamen Kommunikationsartefakt, das Entwicklern und Stakeholdern erlaubt, während der Planung zu verifizieren, dass sie mit demselben Begriff dasselbe meinen, was eine Lücke schließt, die sonst Anforderung für Anforderung wieder auftaucht. Weil sich Geschäftsverständnis weiterentwickelt, muss das Modell als lebendes Artefakt behandelt werden, das über die Zeit überarbeitet wird, statt als einmaliges Dokument, wobei es gut zu bauen echte Zeitinvestition von Domänenexperten erfordert, die gegen das Risiko übermäßiger Modellierung abgewogen werden muss, bevor irgendein Refactoring-Nutzen realisiert wird.

## How to Apply ◆

- Arbeiten Sie mit Domänenexperten zusammen, um die Schlüsselgeschäftskonzepte, ihre Attribute und Beziehungen in der Domäne des Legacy-Systems zu identifizieren.
- Erstellen Sie visuelle Domänenmodelle (UML-Klassendiagramme, CRC-Karten oder informelle Diagramme), die die Geschäftsdomäne unabhängig von der aktuellen Implementierung repräsentieren.
- Vergleichen Sie das Domänenmodell mit den tatsächlichen Datenstrukturen und der Code-Organisation des Legacy-Systems, um Unstimmigkeiten zu identifizieren.
- Nutzen Sie das Domänenmodell, um Refactoring zu leiten: Strukturieren Sie Legacy-Code so um, dass Klassen und Module Domänenkonzepten entsprechen.
- Iterieren Sie am Domänenmodell, während sich das Verständnis vertieft; behandeln Sie es als lebendes Artefakt, nicht als einmaliges Dokument.
- Nutzen Sie Domänenmodelle als Kommunikationswerkzeug während Planungssitzungen, um sicherzustellen, dass Entwickler und Stakeholder dasselbe Verständnis teilen.

## Tradeoffs ⇄

**Vorteile:**
- Schafft ein gemeinsames Verständnis der Geschäftsdomäne, das die Lücke zwischen technischen und geschäftlichen Stakeholdern überbrückt.
- Offenbart, wo die Struktur des Legacy-Systems von der Geschäftsrealität abweicht, der es dient.
- Leitet Refactoring- und Umstrukturierungsbemühungen hin zu einer domänenausgerichteteren Codebasis.
- Dient als Grundlage für die Anwendung von Domain-Driven-Design-Mustern.

**Kosten:**
- Der Bau eines genauen Domänenmodells erfordert erhebliche Zeit mit Domänenexperten.
- Domänenmodelle können veralten, wenn sie nicht gepflegt werden, während sich das Geschäft weiterentwickelt.
- Die Lücke zwischen dem Domänenmodell und der Legacy-Implementierung könnte zu groß sein, um schrittweise überbrückt zu werden.
- Übermäßige Modellierung kann die Entwicklung verlangsamen, wenn das Team zu viel Zeit mit der Perfektionierung des Modells verbringt.

## How It Could Be

Ein Legacy-Lieferkettenmanagementsystem nutzt technische Abstraktionen („Record", „Transaction", „Item"), die nicht darauf abbilden, wie Logistikmanager über ihre Domäne denken („Bestellung", „Sendung", „Lagereinheit"). Das Entwicklungsteam führt Domänenmodellierungs-Workshops mit Logistikmanagern durch und erstellt ein Domänenmodell mittels Geschäftsterminologie. Der Vergleich dieses Modells mit dem Legacy-Code enthüllt, dass eine einzelne „Transaction"-Tabelle Bestellungen, Sendungsdatensätze und Bestandsanpassungen speichert, unterschieden nur durch einen Typencode. Diese Erkenntnis leitet einen Refactoring-Aufwand, der diese Konzepte in separate Domänenobjekte trennt, was den Code für neue Entwickler verständlich macht und dem Logistikteam erlaubt, Anforderungen mittels Begriffen zu kommunizieren, die direkt auf Codestrukturen abbilden.
