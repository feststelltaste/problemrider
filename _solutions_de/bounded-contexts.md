---
title: Bounded Contexts
description: Trennung von Geschäftsbereichen mit unterschiedlichen Begriffen und
  Regeln voneinander.
category:
- Architecture
problems:
- monolithic-architecture-constraints
- complex-domain-model
- poor-domain-model
- tight-coupling-issues
- high-coupling-low-cohesion
- spaghetti-code
- ripple-effect-of-changes
- shared-database
layout: solution
lang: de
en_slug: bounded-contexts
related_solutions:
- slug: domain-driven-design
  similarity: 0.75
- slug: domain-aligned-architecture
  similarity: 0.75
- slug: domain-modeling
  similarity: 0.7
- slug: modularization-and-bounded-contexts
  similarity: 0.7
- slug: high-cohesion
  similarity: 0.7
- slug: separation-of-concerns
  similarity: 0.65
---

## Description

Ein Bounded Context ist eine explizit gezogene Grenze um einen Teil eines Systems, innerhalb derer ein bestimmtes Domänenmodell, seine Terminologie und seine Geschäftsregeln konsistent gelten, wobei Übersetzung bewusst an der Grenze geschieht, wann immer Information in einen anderen Kontext übertritt. Der Mechanismus funktioniert, indem akzeptiert wird, dass ein einziges universelles Modell, das überall dasselbe bedeutet, für jedes System echter Größe unrealistisch ist — dasselbe Wort, wie „Kunde", bedeutet für die Abrechnung und für den Support legitim unterschiedliche Dinge — und statt eine gemeinsame Definition zu erzwingen, partitioniert es das System, sodass jeder Teil sein eigenes Modell definieren kann, ohne die anderen zu korrumpieren. Legacy-Systeme sind häufig das Gegenteil davon: Eine einzelne Entität oder Tabelle hat Felder und bedingte Logik von jeder Abteilung angehäuft, die jemals etwas leicht anderes davon brauchte, was ein aufgeblähtes, tief gekoppeltes Modell produziert, das kein einzelnes Team ändern kann, ohne jedes andere Team zu beeinflussen, das ebenfalls davon abhängt. Bounded Contexts in ein solches System einzuführen bedeutet, zu identifizieren, wo diese impliziten Grenzen bereits in der Praxis existieren, sie mit expliziten Schnittstellen und Anti-Corruption Layers zu formalisieren und jedem Kontext das Eigentum an seinen eigenen Daten zu geben, statt sie direkt Tabellen teilen zu lassen. Das Ergebnis ist, dass sich jeder Kontext unabhängig weiterentwickeln, deployen und über ihn nachgedacht werden kann, was auch der Grund ist, warum Bounded Contexts die natürliche Zerlegungsgrenze sind, wenn ein Monolith schließlich in separate Services aufgebrochen wird.

## How to Apply ◆

- Identifizieren Sie unterschiedliche Geschäftsdomänen im Legacy-System, wo dieselben Begriffe unterschiedliche Bedeutungen haben oder wo Geschäftsregeln sich unterscheiden (z. B. „Kunde" in der Abrechnung vs. im Support).
- Ziehen Sie explizite Grenzen um diese Domänen und definieren Sie, wie sie durch gut spezifizierte Schnittstellen kommunizieren.
- Kartieren Sie bestehende Legacy-Code-Module auf Bounded Contexts, um zu verstehen, wo Grenzen verletzt werden.
- Führen Sie Anti-Corruption Layers an Kontextgrenzen ein, um zwischen unterschiedlichen Domänenmodellen zu übersetzen.
- Refaktorieren Sie gemeinsam genutzte Datenbanktabellen, die mehrere Kontexte umspannen, indem Sie jedem Kontext das Eigentum an seinen eigenen Daten geben.
- Nutzen Sie Kontextkarten, um Beziehungen zwischen Bounded Contexts zu dokumentieren (Shared Kernel, Customer-Supplier, Conformist).

## Tradeoffs ⇄

**Vorteile:**
- Jeder Kontext kann sich unabhängig mit seinem eigenen Domänenmodell und seinen eigenen Regeln weiterentwickeln.
- Verringert die kognitive Last, indem Komplexität auf eine handhabbare Grenze beschränkt wird.
- Verhindert Terminologieverwirrung, die zu Bugs führt, wenn unterschiedliche Domänen dieselbe Codebasis teilen.
- Schafft natürliche Zerlegungsgrenzen zum Aufbrechen von Monolithen.

**Kosten:**
- Die Identifikation korrekter Grenzen erfordert tiefes Domänenwissen, das in Legacy-Systemen teilweise verloren gegangen sein könnte.
- Die Einführung von Grenzen in einen eng gekoppelten Monolithen ist ein schrittweiser, aufwendiger Prozess.
- Datenduplizierung über Kontexte hinweg erfordert Synchronisationsmechanismen.
- Über-Zerlegung kann zu übermäßigem Kommunikations-Overhead zwischen Kontexten führen.

## How It Could Be

Ein Legacy-Universitätsverwaltungssystem nutzt eine einzelne „Student"-Entität über Einschreibung, Notenvergabe, Studienfinanzierung und Wohnheim hinweg. Jede Abteilung hat unterschiedliche Regeln und Attribute dafür, was ein „Student" bedeutet, was zu einem aufgeblähten Modell mit Hunderten von Feldern und komplexer bedingter Logik führt. Das Team identifiziert vier Bounded Contexts und erstellt separate Studentenmodelle für jeden, verbunden durch eine gemeinsame Studenten-Kennung. Ein Anti-Corruption Layer übersetzt zwischen Kontexten, wenn sie Informationen austauschen müssen. Der Einschreibungskontext kann nun neue Registrierungs-Workflows hinzufügen, ohne die komplexen Berechtigungsberechnungen des Studienfinanzierungsmoduls zu beeinträchtigen, und jedes Team kann unabhängig über sein Domänenmodell nachdenken.
