---
title: Event Storming
description: Entdeckung von Domain Events, Commands und Aggregates in kollaborativen
  Workshops.
category:
- Requirements
- Architecture
problems:
- legacy-business-logic-extraction-difficulty
- implicit-knowledge
- requirements-ambiguity
- stakeholder-developer-communication-gap
- poor-domain-model
- monolithic-architecture-constraints
layout: solution
lang: de
en_slug: event-storming
related_solutions:
- slug: domain-modeling
  similarity: 0.75
- slug: business-process-modeling
  similarity: 0.7
- slug: architecture-workshops
  similarity: 0.7
- slug: domain-driven-design
  similarity: 0.7
- slug: story-mapping
  similarity: 0.65
- slug: bounded-contexts
  similarity: 0.65
---

## Description

Event Storming ist ein kollaboratives Workshop-Format, entwickelt innerhalb der Domain-Driven-Design-Community, in dem Entwickler, Domänenexperten und Stakeholder gemeinsam einen Geschäftsprozess mittels Klebezetteln rekonstruieren: zuerst Domain Events in chronologischer Reihenfolge platzieren, dann die Commands hinzufügen, die sie auslösen, die Aggregates, die für die Behandlung dieser Commands verantwortlich sind, und die Policies, die ein Event automatisch mit dem nächsten verbinden. Die Technik ist besonders effektiv gegen ein spezifisches Legacy-Problem — dass das tatsächliche Verhalten eines alten Systems oft nur fragmentarisch bekannt ist, verstreut über die Köpfe einiger weniger langjähriger Mitarbeiter, ohne ein einzelnes Artefakt, das den vollständigen Prozess Ende-zu-Ende beschreibt. Weil das Workshop-Format dieses Wissen kollektiv und visuell zutage fördert, in einer Angelegenheit von Stunden statt der Wochen, die ein schriftlicher Spezifikationsaufwand brauchen könnte, tendiert es dazu, Widersprüche, undokumentierte Nebenkanäle und Lücken im gemeinsamen Verständnis des Teams aufzudecken, von denen kein einzelner Teilnehmer zuvor wusste. Die Cluster von Events und Aggregates, die aus der Sitzung entstehen, verdoppeln sich auch als natürliche Kandidatengrenzen für die Zerlegung eines Monolithen, was Event Storming sowohl für die Planung der Zielarchitektur einer Modernisierung als auch für das Verständnis der aktuellen wertvoll macht — obwohl seine Ausgabe nur so dauerhaft ist wie der Aufwand, sie danach zu formalisieren, da Klebezettel an einer Wand keine dauerhafte Dokumentation sind.

## How to Apply ◆

- Organisieren Sie Workshops mit Entwicklern, Domänenexperten und Stakeholdern mittels Klebezetteln an einer großen Wand oder einem digitalen Whiteboard.
- Beginnen Sie damit, Domain Events zu identifizieren (Dinge, die im Geschäft geschehen) und ordnen Sie sie chronologisch an.
- Fügen Sie Commands (was Events auslöst), Aggregates (Entitäten, die für die Behandlung von Commands verantwortlich sind) und Policies (automatisierte Reaktionen auf Events) hinzu.
- Nutzen Sie den resultierenden Event-Fluss, um die tatsächlichen Geschäftsprozesse des Legacy-Systems abzubilden, was versteckte Komplexität und undokumentierte Abläufe enthüllt.
- Identifizieren Sie Bounded-Context-Grenzen, an denen unterschiedliche Gruppen von Events und Aggregates kohärente Cluster bilden.
- Nutzen Sie die Event-Storming-Ausgabe, um die Zerlegung monolithischer Legacy-Systeme in wohldefinierte Module oder Services zu leiten.

## Tradeoffs ⇄

**Vorteile:**
- Fördert implizites Domänenwissen, das nur in den Köpfen der Menschen existiert, schnell zutage.
- Schafft gemeinsames Verständnis über geschäftliche und technische Teilnehmer hinweg in Stunden statt Wochen.
- Offenbart Lücken und Widersprüche im aktuellen Verständnis des Legacy-System-Verhaltens.
- Erzeugt natürliche Grenzen für Systemzerlegung und Teamorganisation.

**Kosten:**
- Erfordert Verfügbarkeit von Schlüssel-Domänenexperten und Entwicklern für konzentrierte Workshop-Zeit.
- Die Workshop-Ausgabe muss formalisiert und gepflegt werden; Klebezettel allein sind keine dauerhafte Dokumentation.
- Moderationsfähigkeiten sind wichtig; schlecht moderierte Sitzungen können unproduktiv sein.
- Große Legacy-Systeme benötigen möglicherweise mehrere Sitzungen, um sie ausreichend abzudecken.

## How It Could Be

Ein Legacy-Auftragsabwicklungssystem muss für die Modernisierung zerlegt werden, aber niemand hat ein vollständiges Bild davon, wie alle Teile zusammenpassen. Das Team führt einen zweitägigen Event-Storming-Workshop mit Lagermanagern, Kundenservice-Mitarbeitern und Entwicklern durch. Sie entdecken über sechzig Domain Events und identifizieren drei unterschiedliche Bounded Contexts: Auftragsannahme, Lagerbetrieb und Versandkoordination. Der Workshop enthüllt, dass das Legacy-System Retouren über einen undokumentierten Nebenkanal handhabt, der den Hauptauftragsfluss umgeht, ein kritischer Geschäftsprozess, der dem Entwicklungsteam unbekannt war. Die Event-Storming-Ausgabe wird zum Bauplan für den Zerlegungsaufwand, und die entdeckten Bounded Contexts leiten sowohl die technische Architektur als auch die Teamstruktur für das Modernisierungsprojekt.
