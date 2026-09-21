---
title: Geschäftsprozessmodellierung
description: Erhebung von Geschäftsanforderungen durch Modellierung der zugrunde
  liegenden Geschäftsprozesse.
category:
- Requirements
- Business
problems:
- requirements-ambiguity
- inadequate-requirements-gathering
- legacy-business-logic-extraction-difficulty
- poor-domain-model
- stakeholder-developer-communication-gap
- implicit-knowledge
- process-software-misfit
layout: solution
lang: de
en_slug: business-process-modeling
related_solutions:
- slug: data-modeling
  similarity: 0.75
- slug: requirements-analysis
  similarity: 0.75
- slug: business-process-automation
  similarity: 0.7
- slug: domain-modeling
  similarity: 0.7
- slug: user-stories
  similarity: 0.7
- slug: evolutionary-requirements-development
  similarity: 0.7
---

## Description

Geschäftsprozessmodellierung erfasst, wie ein Geschäftsprozess tatsächlich funktioniert — durch Stakeholder-Interviews und direkte Beobachtung echter Workflows statt Verlass auf bestehende Dokumentation — und stellt ihn visuell dar, typischerweise in BPMN, sodass sowohl Geschäfts- als auch technische Stakeholder dasselbe Verständnis des Prozesses teilen, unabhängig von der Implementierung eines bestimmten Systems. Der Wert des Mechanismus liegt spezifisch in der Lücke, die er offenlegen soll: Dokumentierte Prozeduren und tatsächliche Praxis weichen oft erheblich voneinander ab, und nur durch Beobachtung dessen, was Menschen tatsächlich tun, kann diese Abweichung gefunden und aufgelöst werden. Legacy-Systeme sind eine häufige Quelle genau dieser Abweichung, weil sich Nutzer über Jahre an die Beschränkungen eines Systems anpassen, indem sie informelle Workarounds entwickeln, die nie in eine Spezifikation gelangen, aber genuine Geschäftsbedürfnisse darstellen, die das System nicht direkt erfüllt. Legacy-Systemfunktionalität gegen das resultierende Prozessmodell zu kartieren offenbart, welche Teile des Systems welche Prozesse unterstützen, und bringt die Workarounds als echte Anforderungen ans Licht, statt als Rauschen, das während einer Ersatzbemühung ignoriert wird. Modernisierungsprojekte, die diesen Schritt überspringen, riskieren den häufigsten Fehlermodus des Legacy-Ersatzes: ein neues System zu bauen, das treu altes Softwareverhalten reproduziert, statt des tatsächlichen Geschäftsprozesses, für den die alte Software immer nur ein unvollkommenes Fahrzeug war.

## How to Apply ◆

- Interviewen Sie Geschäfts-Stakeholder und beobachten Sie tatsächliche Workflows, um zu erfassen, wie Geschäftsprozesse wirklich funktionieren, nicht nur, wie Dokumentation sagt, dass sie funktionieren sollten.
- Nutzen Sie BPMN oder ähnliche Notation, um visuelle Prozessmodelle zu erstellen, die sowohl Geschäfts- als auch technische Teams verstehen können.
- Kartieren Sie Legacy-Systemfunktionalität gegen das Geschäftsprozessmodell, um zu identifizieren, welche Teile des Systems welche Prozesse unterstützen.
- Identifizieren Sie Diskrepanzen zwischen dokumentierten Prozessen und tatsächlichem Systemverhalten, die in Legacy-Umgebungen üblich sind.
- Nutzen Sie Prozessmodelle, um Automatisierungsmöglichkeiten und redundante manuelle Schritte zu entdecken.
- Pflegen Sie Prozessmodelle als lebende Dokumente, die aktualisiert werden, wenn sich Prozesse oder Anforderungen ändern.

## Tradeoffs ⇄

**Vorteile:**
- Schafft ein gemeinsames Verständnis von Geschäftsprozessen zwischen Geschäfts- und technischen Stakeholdern.
- Offenbart versteckte Geschäftslogik, die in Legacy-Systemen eingebettet und möglicherweise nirgendwo dokumentiert ist.
- Bietet eine Grundlage für Anforderungserhebung während Modernisierungsbemühungen.
- Identifiziert Ineffizienzen und Redundanzen in aktuellen Prozessen.

**Kosten:**
- Die genaue Modellierung bestehender Prozesse erfordert erhebliche Zeitinvestition und Stakeholder-Zugang.
- Prozessmodelle können schnell veralten, wenn sie nicht aktiv gepflegt werden.
- Stakeholder könnten idealisierte statt tatsächliche Prozesse beschreiben, was Beobachtung zur Validierung erfordert.
- Übermäßig detaillierte Modelle können so schwer verständlich werden wie der Code, den sie beschreiben.

## How It Could Be

Eine Regierungsbehörde plant, ein Legacy-Fallmanagementsystem zu ersetzen, entdeckt aber, dass niemand die aktuellen Geschäftsprozesse vollständig versteht. Das Team führt Workshops mit Sachbearbeitern durch und erstellt BPMN-Diagramme davon, wie Fälle tatsächlich durch das System fließen. Sie entdecken, dass der echte Prozess erheblich von dem offiziellen Verfahrenshandbuch abweicht: Sachbearbeiter haben zahlreiche Workarounds entwickelt, um Systembeschränkungen zu kompensieren. Diese Workarounds stellen genuine Geschäftsbedürfnisse dar, die im Ersatzsystem adressiert werden müssen. Die Prozessmodelle werden zur maßgeblichen Anforderungsquelle für das Modernisierungsprojekt und verhindern den üblichen Fehler, ein neues System zu bauen, das altes Softwareverhalten repliziert statt tatsächlicher Geschäftsbedürfnisse.
