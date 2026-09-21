---
title: Kontinuierliche Auslieferung
description: Häufige und inkrementelle Auslieferung von Funktionalität.
category:
- Process
- Operations
problems:
- long-release-cycles
- complex-deployment-process
- manual-deployment-processes
- deployment-risk
- large-risky-releases
- release-anxiety
- immature-delivery-strategy
- delayed-value-delivery
- extended-cycle-times
- increased-time-to-market
- uneven-work-flow
layout: solution
lang: de
en_slug: continuous-delivery
related_solutions:
- slug: continuous-integration-and-delivery
  similarity: 0.85
- slug: ci-cd-pipeline
  similarity: 0.85
- slug: continuous-deployment
  similarity: 0.8
- slug: feature-driven-development
  similarity: 0.75
- slug: rollback-mechanisms
  similarity: 0.75
- slug: trunk-based-development
  similarity: 0.75
---

## Description

Kontinuierliche Auslieferung hält die Codebasis in einem Zustand, in dem sie jederzeit in Produktion ausgerollt werden könnte, indem die Build-, Test- und Verpackungs-Pipeline automatisiert wird, sodass jede Änderung ein auslieferbares Artefakt erzeugt, statt sich in einem nicht veröffentlichten Branch bis zum nächsten geplanten Release anzuhäufen. Legacy-Systeme sind oft in seltene, große „Big Bang"-Releases gesperrt, gerade weil Deployment manuell, fehleranfällig und gefürchtet ist, was einen Teufelskreis erzeugt: Seltene Releases bündeln mehr Änderungen, größere Releases tragen mehr Risiko, und höheres Risiko verstärkt die Zurückhaltung, häufiger auszuliefern. Die Pipeline zu automatisieren und Praktiken wie trunk-basierte Entwicklung, Feature Flags und automatisierte Smoke Tests einzuführen durchbricht diesen Kreislauf, indem jedes Release klein genug wird, um darüber nachzudenken, und sicher genug, um es schnell rückgängig zu machen, falls etwas schiefgeht. Weil Feature Flags Deployment von der Feature-Aktivierung entkoppeln, kann Code für ein unvollständiges Feature durch die Pipeline wandern und ruhend in Produktion liegen, ohne Nutzern ausgesetzt zu sein, was besonders wertvoll ist, wenn die Architektur eines Legacy-Systems langlebige Feature-Branches kostspielig zu pflegen macht. Die Hauptkosten liegen vorab: Zuverlässige Automatisierung und ausreichende automatisierte Testabdeckung für einen Legacy-Build-Prozess aufzubauen, der nie zuvor vollständig automatisiert war, erfordert echte Investition und einen kulturellen Wandel weg davon, Releases als seltene Ereignisse mit hoher Zeremonie zu behandeln.

## How to Apply ◆

- Automatisieren Sie die Build-, Test- und Deployment-Pipeline für das Legacy-System, beginnend mit den fehleranfälligsten manuellen Schritten.
- Implementieren Sie trunk-basierte Entwicklung oder kurzlebige Feature-Branches, um die Merge-Komplexität in der Legacy-Codebasis zu reduzieren.
- Deployen Sie häufig in Produktion in kleinen Schritten, statt in großen, riskanten Releases.
- Nutzen Sie Feature Flags, um Deployment von Feature-Aktivierung zu entkoppeln, sodass Code deployt werden kann, ohne unvollständige Funktionalität freizulegen.
- Bauen Sie automatisierte Smoke Tests, die die Kernfunktionalität des Legacy-Systems nach jedem Deployment verifizieren.
- Erstellen Sie automatisierte Rollback-Fähigkeiten, um das Risiko beim Ausrollen von Änderungen an Legacy-Systemen zu reduzieren.
- Standardisieren Sie Umgebungen mittels Infrastructure as Code, um „funktioniert auf meiner Maschine"-Probleme zu eliminieren.

## Tradeoffs ⇄

**Vorteile:**
- Reduziert Deployment-Risiko, indem jedes Release kleiner und vorhersehbarer wird.
- Verkürzt Feedback-Schleifen, sodass Teams Probleme schneller erkennen und beheben können.
- Eliminiert manuelle Deployment-Fehler, die bei Legacy-System-Releases häufig sind.
- Ermöglicht schrittweise Modernisierung, indem kleine Verbesserungen schnell Produktion erreichen können.

**Kosten:**
- Erfordert erhebliche Vorabinvestition, um Legacy-System-Builds und -Deployments zu automatisieren.
- Legacy-Systeme können Abhängigkeiten oder architektonische Einschränkungen haben, die häufiges Deployment erschweren.
- Erfordert umfassendes automatisiertes Testing, um Vertrauen in häufige Releases aufrechtzuerhalten.
- Der kulturelle Wandel von seltenen „Big Bang"-Releases erfordert Teamanpassung und Managementunterstützung.

## How It Could Be

Ein Legacy-Content-Management-System wird vierteljährlich über einen manuellen, zweitägigen Prozess ausgerollt, der mehrere Teams und Übergabedokumente involviert. Jedes Release bündelt Monate von Änderungen, und Rollbacks erfordern die Wiederherstellung aus einem Backup. Das Team investiert drei Monate in den Aufbau einer CI/CD-Pipeline: automatisierte Builds, Datenbankmigrationsskripte, Umgebungsbereitstellung und Smoke Tests. Sie beginnen, wöchentlich zu releasen, dann zweimal wöchentlich. Deployment-Vorfälle sinken drastisch, weil jedes Release weniger Änderungen enthält, und die automatisierte Pipeline eliminiert die menschlichen Fehler, die manuelle Deployments plagten. Das Team entdeckt und behebt Fehler innerhalb von Tagen statt sie über Monate anzuhäufen, und Stakeholder gewinnen Vertrauen in den Auslieferungsprozess.
