---
title: Geschäftsprozessautomatisierung
description: Abbildung von Geschäftskonzepten und -regeln in einem ausführbaren Modell.
category:
- Business
- Process
problems:
- legacy-business-logic-extraction-difficulty
- complex-and-obscure-logic
- increased-manual-work
- inefficient-processes
- process-design-flaws
- poor-domain-model
layout: solution
lang: de
en_slug: business-process-automation
related_solutions:
- slug: business-process-modeling
  similarity: 0.7
- slug: rule-based-systems
  similarity: 0.7
- slug: decision-tables
  similarity: 0.65
- slug: development-workflow-automation
  similarity: 0.65
- slug: data-modeling
  similarity: 0.65
- slug: business-event-processing
  similarity: 0.65
---

## Description

Geschäftsprozessautomatisierung extrahiert Geschäftsregeln und Workflow-Logik, die aktuell innerhalb von Legacy-Anwendungscode eingebettet sind — oft verstreut über Stored Procedures, bedingte Verzweigungen und manuelle Übergaben — in eine explizite Prozess-Engine, getrieben von BPMN-Prozessmodellen und DMN-Entscheidungstabellen, die nicht nur Entwickler, sondern auch Fachanalysten lesen und ändern können. Der Mechanismus trennt, was ein Geschäftsprozess tun soll, von der Art, wie ein bestimmtes Legacy-System es heute zufällig implementiert, und macht zuvor implizite Regeln sichtbar und gibt ihnen eine Heimat außerhalb des Codes, wo sie überprüft, versioniert und unabhängig von einem Deployment-Zyklus geändert werden können. Dies ist direkt relevant für Legacy-Modernisierung, weil sich Geschäftslogik in alten Systemen häufig über Jahre angehäuft hat, ohne jemals irgendwo explizit modelliert worden zu sein, was bedeutet, dass die „Dokumentation" einer kritischen Geschäftsregel in der Praxis der Code selbst plus die wenigen Menschen ist, die sich noch erinnern, warum sie so geschrieben wurde. Solche Logik schrittweise und beginnend mit gut verstandenen, hochvolumigen Prozessen in eine Prozess-Engine zu migrieren, klärt die Regel zum ersten Mal seit Jahren und entfernt die fragilen manuellen Übergaben (E-Mail, Tabellenkalkulationen), die häufig um Legacy-Systeme herum bestehen bleiben, genau weil das System selbst den vollständigen Prozess nicht ausdrücken konnte. Die Kosten sind der operative Overhead eines neuen Infrastrukturteils und die Schwierigkeit der Extraktion selbst, die genau dort am schwierigsten ist, wo die Geschäftslogik am tiefsten mit der Legacy-technischen Implementierung verwoben ist.

## How to Apply ◆

- Extrahieren Sie Geschäftsregeln, die aktuell in Legacy-Code eingebettet sind, in eine Geschäftsprozess-Engine (Camunda, Flowable oder ähnliche BPMN-basierte Werkzeuge).
- Modellieren Sie bestehende Geschäftsprozesse explizit mit BPMN, bevor Sie sie automatisieren, und machen Sie implizite Logik sichtbar.
- Beginnen Sie mit hochvolumigen, gut verstandenen Prozessen und migrieren Sie sie schrittweise zur Prozess-Engine.
- Definieren Sie Geschäftsregeln in einem Format, das Fachanalysten überprüfen und ändern können (Entscheidungstabellen, DMN).
- Integrieren Sie die Prozess-Engine mit Legacy-Systemen durch Adapter, sodass automatisierte Prozesse bestehende Funktionalität aufrufen können.
- Nutzen Sie Prozessmonitoring, um Engpässe zu identifizieren und Workflows basierend auf tatsächlichen Ausführungsdaten zu optimieren.

## Tradeoffs ⇄

**Vorteile:**
- Macht Geschäftslogik explizit und wartbar, indem sie vom Anwendungscode getrennt wird.
- Ermöglicht Fachanalysten, Prozessabläufe zu verstehen und zu ändern, ohne Entwicklerbeteiligung.
- Bietet Prüfpfade und Prozessmonitoring von Haus aus.
- Verringert manuelle Arbeit und fehleranfällige Übergaben zwischen Systemen.

**Kosten:**
- Die Einführung einer Prozess-Engine fügt Infrastruktur und operative Komplexität hinzu.
- Die Extraktion von Geschäftslogik aus Legacy-Code ist schwierig, wenn sie tief mit technischer Implementierung verwoben ist.
- Über-Automatisierung einfacher Prozesse kann unnötige Komplexität hinzufügen.
- Prozess-Engines haben ihre eigene Lernkurve und Wartungsanforderungen.

## How It Could Be

Ein Legacy-Kreditbearbeitungssystem hat Geschäftsregeln, die über Stored Procedures, Anwendungscode und manuelle Workflows mit E-Mail und Tabellenkalkulationen verstreut sind. Die Bearbeitung eines einzelnen Kreditantrags dauert aufgrund manueller Übergaben Tage. Das Team modelliert den Kreditgenehmigungsprozess in BPMN und extrahiert Entscheidungsregeln in DMN-Tabellen, die Kreditsachbearbeiter überprüfen können. Die Prozess-Engine orchestriert den Workflow und leitet Anträge automatisch durch Bonitätsprüfungen, Dokumentenverifikation und Genehmigungsschritte. Manuelle Eingriffe sind nur für Ausnahmen erforderlich. Die Bearbeitungszeit sinkt von Tagen auf Stunden, und das Geschäft kann Genehmigungsschwellen ändern, ohne Codeänderungen anzufordern.
