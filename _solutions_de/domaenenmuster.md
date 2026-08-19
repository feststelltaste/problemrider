---
title: Domänenmuster
description: Anwendung bewährter Lösungen für wiederkehrende Geschäftsprobleme.
category:
- Architecture
- Code
problems:
- complex-and-obscure-logic
- poor-domain-model
- legacy-business-logic-extraction-difficulty
- suboptimal-solutions
- accumulation-of-workarounds
layout: solution
lang: de
en_slug: domain-patterns
related_solutions:
- slug: pattern-language
  similarity: 0.85
- slug: domain-modeling
  similarity: 0.8
- slug: domain-driven-design
  similarity: 0.75
- slug: ubiquitous-language
  similarity: 0.75
- slug: incremental-refactoring
  similarity: 0.75
- slug: consistent-terminology
  similarity: 0.75
---

## Description

Domänenmuster sind bewährte, gut dokumentierte Lösungsstrukturen für wiederkehrende Probleme innerhalb einer spezifischen Geschäftsdomäne — etwa das Doppik-Muster der Buchhaltungsbranche oder etablierte Enterprise-Integration-Muster —, die ein Team anwenden kann, statt sich weiterhin auf Ad-hoc-, selbst entwickelte Logik zu verlassen, die lokal erfunden wurde, um dasselbe wiederkehrende Problem zu lösen. Legacy-Systeme häufen über die Zeit genau diese Art selbst entwickelter Logik an: Ein Problem, das die Branche bereits sauber gelöst hat, wird intern als eine Menge verstreuter Validierungsprüfungen oder Abgleichskripte neu erfunden, oft weil niemand im damaligen Team wusste, dass ein etabliertes Muster existierte. Solche Ad-hoc-Logik durch das entsprechende Domänenmuster zu ersetzen tut mehr als den Code zu vereinfachen — es kann ganze Kategorien von Fehlern strukturell unmöglich machen, wie bei einem Buchhaltungsmuster, das garantiert, dass jede Transaktion konstruktionsbedingt ausgeglichen ist, statt durch verstreute Prüfungen, die umgangen oder verpasst werden können. Diese Substitution zahlt auch eine Verständnisdividende: Ein Entwickler, der das Standardmuster bereits aus früherer Erfahrung in der Branche kennt, kann sofort produktiv im refaktorierten Code arbeiten, ohne zunächst die benutzerdefinierte Implementierung, die es ersetzt, per Reverse Engineering erschließen zu müssen. Das Hauptrisiko ist, ein Muster gewaltsam auf ein Problem zu passen, zu dem es nicht wirklich passt, was eine Art unnötiger Komplexität gegen eine andere tauscht, sodass die Anwendung von Domänenmustern immer noch echte Domänenkompetenz erfordert, statt bloßem Mustervergleich anhand des Namens.

## How to Apply ◆

- Studieren Sie domänenspezifische Muster, die für die Branche des Legacy-Systems relevant sind (z. B. Martin Fowlers Analysis Patterns, Enterprise-Integration-Muster, Buchhaltungsmuster).
- Identifizieren Sie, wo das Legacy-System Ad-hoc-Lösungen für Probleme hat, für die gut bekannte Domänenmuster existieren.
- Refaktorieren Sie Legacy-Code schrittweise, um etablierte Domänenmuster zu nutzen, beginnend mit den problematischsten Bereichen.
- Schulen Sie das Entwicklungsteam in Domänenmustern, die auf den Geschäftsbereich ihres Systems anwendbar sind.
- Nutzen Sie Domänenmuster als gemeinsames Vokabular bei der Diskussion von Designentscheidungen mit dem Team.
- Dokumentieren Sie, welche Domänenmuster wo genutzt werden, und erstellen Sie eine Musterkarte für das Legacy-System.

## Tradeoffs ⇄

**Vorteile:**
- Ersetzt Ad-hoc-, selbst entwickelte Lösungen durch bewährte, in der Branche gut verstandene Ansätze.
- Macht die Codebasis für neue Entwickler, die die Standardmuster kennen, vertrauter.
- Reduziert das Risiko subtiler Fehler, die aus der Neuerfindung von Lösungen für gut bekannte Probleme entstehen.
- Bietet ein Vokabular für die Diskussion wiederkehrender Geschäftskonzepte.

**Kosten:**
- Das richtige Muster zu finden erfordert Domänenwissen und Musterkompetenz.
- Ein Muster gewaltsam auf ein Problem zu passen, zu dem es nicht passt, erzeugt unnötige Komplexität.
- Das Refactoring bestehenden Codes, um zu einem Muster zu passen, erfordert Aufwand und sorgfältiges Testing.
- Manche Domänenmuster lassen sich möglicherweise nicht sauber auf die bestehende Struktur des Legacy-Systems abbilden.

## How It Could Be

Ein Legacy-Buchhaltungssystem implementiert Doppik durch verstreute Validierungsprüfungen und Abgleichskripte, statt das gut etablierte Buchungsmuster zu nutzen. Diskrepanzen zwischen Konten sind ein wiederkehrendes Problem, und das Debugging erfordert das Verfolgen durch mehrere Codepfade. Das Team refaktoriert die Kern-Transaktionsbehandlung, um das Standard-Buchungsmuster zu nutzen, bei dem jedes Finanzereignis ausgeglichene Soll- und Habenbuchungen als atomare Operation erzeugt. Das Muster macht es strukturell unmöglich, unausgeglichene Buchungen zu erstellen, was eine ganze Fehlerkategorie eliminiert. Neue Entwickler mit Buchhaltungs-Domänenwissen erkennen das Muster sofort und können produktiv arbeiten, ohne die benutzerdefinierte Implementierung zu studieren, die es ersetzt.
