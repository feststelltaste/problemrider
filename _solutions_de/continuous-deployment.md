---
title: Continuous Deployment
description: Vollautomatisiertes Deployment von Software-Änderungen in die Produktivumgebung.
category:
- Process
- Operations
problems:
- complex-deployment-process
- manual-deployment-processes
- deployment-risk
- large-risky-releases
- long-release-cycles
- release-anxiety
- release-instability
- frequent-hotfixes-and-rollbacks
layout: solution
lang: de
en_slug: continuous-deployment
related_solutions:
- slug: ci-cd-pipeline
  similarity: 0.9
- slug: blue-green-canary-deployments
  similarity: 0.8
- slug: continuous-delivery
  similarity: 0.8
- slug: continuous-integration-and-delivery
  similarity: 0.75
- slug: development-workflow-automation
  similarity: 0.75
- slug: smoke-testing
  similarity: 0.75
---

## Description

Continuous Deployment erweitert Continuous Delivery um einen weiteren Schritt, indem das manuelle Genehmigungstor vollständig entfernt wird: Jede Änderung, die die automatisierte Pipeline besteht, wird automatisch in Produktion ausgerollt, ohne dass ein Mensch entscheidet, wann oder ob ein bestimmter Build live geht. Für Legacy-Systeme ist dies typischerweise die letzte Stufe einer längeren Reise statt ein Ausgangspunkt, weil es Voraussetzungen erfordert, die viele Legacy-Umgebungen schlicht nicht haben — eine umfassende automatisierte Testsuite, die manuelle Verifikation ersetzen kann, zuverlässiges automatisiertes Rollback und Pipeline-Schritte, die legacy-spezifische Komplikationen wie Datenbankmigrationen oder Koordination mit abhängigen Systemen handhaben. Wo manuelle Deployment-Prozeduren nur als stillschweigendes Wissen oder ein zwischen Operatoren weitergegebenes Runbook existieren, zwingt die für die vollständige Automatisierung nötige Disziplin auch dazu, dieses Wissen explizit zu machen, was selbst ein wertvoller Nebeneffekt unabhängig vom gewonnenen Deployment-Tempo ist. Einmal erreicht, verkürzt Continuous Deployment die Feedback-Schleife zwischen einer Codeänderung und ihrer Produktionsvalidierung von Wochen auf Minuten, und weil jede ausgerollte Änderung klein ist, werden Vorfälle tendenziell leichter zu diagnostizieren und rückgängig zu machen, statt häufiger zu werden. Das Risiko, gegen das dies eingetauscht wird, ist, dass Automatisierung ohne ausreichende Überwachung Fehler schneller in Produktion bringen kann, als ein manueller Prozess es je könnte, sodass die Investition in automatisiertes Testing und Observability mit der zunehmenden Deployment-Frequenz Schritt halten muss.

## How to Apply ◆

> In Legacy-Systemen ist Continuous Deployment oft das Endziel einer langen Reise — Teams müssen zunächst Vertrauen durch Continuous Integration und Continuous Delivery aufbauen, bevor sie Produktions-Deployments vollständig automatisieren.

- Beginnen Sie damit, den Deployment-Prozess in Nicht-Produktionsumgebungen zu automatisieren, bevor Sie Produktionsautomatisierung versuchen — viele Legacy-Systeme haben Deployment-Prozeduren, die nur als stillschweigendes Wissen oder manuelle Runbooks existieren.
- Bauen Sie eine umfassende automatisierte Testsuite, die ausreichend Vertrauen bietet, um ohne manuelle Verifikation zu deployen — dies ist oft die größte Voraussetzung für Legacy-Systeme.
- Implementieren Sie automatisierte Rollback-Fähigkeiten, sodass fehlgeschlagene Deployments schnell ohne manuellen Eingriff rückgängig gemacht werden können.
- Nutzen Sie Feature Flags, um Deployment von Release zu entkoppeln, sodass Code kontinuierlich deployt werden kann, während neue Features Nutzern schrittweise offenbart werden.
- Etablieren Sie automatisierte Smoke Tests, die unmittelbar nach jedem Deployment laufen, um zu verifizieren, dass die Kernfunktionalität funktioniert.
- Überwachen Sie Deployment-Frequenz, Lead Time, Fehlerrate und Wiederherstellungszeit als Schlüsselkennzahlen, um Fortschritt hin zu zuverlässigem Continuous Deployment zu verfolgen.
- Adressieren Sie Legacy-System-Einschränkungen (Datenbankmigrationen, Konfigurationsänderungen, Koordination abhängiger Systeme) mit automatisierten Vor- und Nach-Deployment-Schritten.

## Tradeoffs ⇄

> Continuous Deployment reduziert Deployment-Risiko und Zyklusdauer dramatisch, erfordert aber erhebliche Investition in Automatisierung, Testing und Überwachungsinfrastruktur.

**Vorteile:**

- Eliminiert manuelle Deployment-Fehler, indem jeder Schritt des Deployment-Prozesses automatisiert wird.
- Reduziert Deployment-Risiko, indem kleine, inkrementelle Änderungen statt großer, seltener Releases ausgerollt werden.
- Verkürzt die Feedback-Schleife zwischen Codeänderung und Produktionsvalidierung von Wochen oder Monaten auf Stunden oder Minuten.
- Entfernt Deployment als Engpass, was schnellere Auslieferung von Bugfixes, Sicherheitspatches und Features ermöglicht.

**Kosten und Risiken:**

- Erfordert umfassendes automatisiertes Testing, das vielen Legacy-Systemen fehlt, was eine erhebliche Vorabinvestition darstellt.
- Legacy-Systeme mit gemeinsam genutzten Datenbanken, manuellen Konfigurationsanforderungen oder externen Systemabhängigkeiten benötigen möglicherweise erhebliche Refaktorierung, um automatisiertes Deployment zu unterstützen.
- Automatisierte Deployments ohne ausreichende Überwachung können Fehler schneller in Produktion bringen, als manuelle Prozesse es würden.
- Organisationskultur kann sich gegen vollständig automatisierte Deployments sträuben, besonders bei Systemen, die sensible Daten oder Finanztransaktionen verarbeiten.
- Datenbankschemaänderungen in Legacy-Systemen können besonders herausfordernd sicher zu automatisieren sein.

## How It Could Be

> Das folgende Szenario zeigt die Reise von manuellem zu Continuous Deployment für ein Legacy-System.

Die Legacy-Plattform eines E-Commerce-Unternehmens erforderte einen vierstündigen manuellen Deployment-Prozess mit drei Teams, einem Deployment-Koordinator und einer detaillierten Checkliste. Deployments fanden monatlich statt und liefen regelmäßig bis nach Mitternacht, mit mindestens einem Rollback pro Quartal. Das Team verbrachte 18 Monate auf dem Weg zu Continuous Deployment: zunächst Automatisierung der Build- und Test-Pipeline, dann Automatisierung von Deployments nach Staging, dann Einführung von Feature Flags und automatisierten Datenbankmigrationen. Als sie schließlich Continuous Deployment in Produktion aktivierten, dauerte das durchschnittliche Deployment 12 Minuten ohne manuelle Schritte. Die Deployment-Frequenz stieg von monatlich auf mehrmals täglich, und das monatliche Ausfallfenster wurde vollständig eliminiert. Die Vorfallrate sank tatsächlich, weil kleinere Änderungen leichter zu diagnostizieren und rückgängig zu machen waren, wenn Probleme auftraten.
