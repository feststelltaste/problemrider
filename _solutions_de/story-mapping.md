---
title: Story Mapping
description: Visualisierung vollständiger Nutzerreisen als zweidimensionale
  Karte von Lücken und Prioritäten.
category:
- Requirements
- Process
problems:
- requirements-ambiguity
- inadequate-requirements-gathering
- feature-gaps
- misaligned-deliverables
- large-feature-scope
- planning-dysfunction
- unclear-goals-and-priorities
- scope-creep
- market-pressure
- changing-project-scope
- gold-plating
- poor-planning
- scope-change-resistance
- stakeholder-dissatisfaction
- unrealistic-deadlines
- unrealistic-schedule
- frequent-changes-to-requirements
layout: solution
lang: de
en_slug: story-mapping
related_solutions:
- slug: user-stories
  similarity: 0.85
- slug: wireframing
  similarity: 0.75
- slug: personas
  similarity: 0.75
- slug: prototypes
  similarity: 0.75
- slug: user-centered-design
  similarity: 0.75
- slug: architecture-roadmap
  similarity: 0.75
---

## Description

Story Mapping ist eine Moderationstechnik, die die User Stories eines Systems in einer zweidimensionalen Karte anordnet — hochrangige Nutzeraktivitäten von links nach rechts in der Reihenfolge ihres Auftretens, mit den detaillierten Aufgaben, die jede Aktivität unterstützen, darunter gestapelt —, sodass die vollständige Form einer Nutzerreise auf einmal sichtbar wird, statt versteckt in einem flachen, undifferenzierten Backlog zu bleiben. Diese räumliche Struktur ist es, was eine Liste von Hunderten von Stories nicht bieten kann: Sie zeigt nicht nur, welche Funktionalität existiert, sondern wie die Teile entlang des Pfads, dem ein echter Nutzer tatsächlich folgt, zueinander in Beziehung stehen, und sie macht Lücken — Stellen, an denen Nutzer derzeit auf manuelle Workarounds oder Schattensysteme angewiesen sind — visuell offensichtlich statt in einer Tabelle vergraben. In Legacy-Ersatzprojekten adressiert dies einen spezifischen und häufigen Fehlermodus, bei dem ein Team Features in einer Reihenfolge baut, die technisch Sinn ergibt, aber keinen Punkt lässt, an dem Nutzer einen gesamten Workflow von Anfang bis Ende abschließen können, weil das Backlog keine Sichtbarkeit darüber gab, welche Stories zur selben Reise gehörten. Das Ziehen einer Release-Linie über die Karte, um einen minimalen tragfähigen Ersatz zu definieren, verwandelt diese Sichtbarkeit dann in einen konkreten, ausgehandelten Lieferplan, einen, den Stakeholder und Entwickler gemeinsam konstruieren, statt einen, der einseitig von einer Seite auferlegt wird. Die Kosten sind, dass die Konstruktion einer anfänglichen Karte für ein großes Legacy-System selbst ein erhebliches Moderationsunterfangen ist, das mehrere Workshops mit verschiedenen Stakeholdern erfordert, und die Karte bleibt nur nützlich, wenn sie aktiv aktuell gehalten wird, während die Migrationsarbeit fortschreitet.

## How to Apply ◆

> In der Legacy-Modernisierung offenbart Story Mapping, welche Teile der Nutzerreise das Legacy-System gut abdeckt, wo es zu kurz kommt und was der Ersatz priorisieren muss.

- Kartieren Sie die vollständige Nutzerreise durch die primären Workflows des Legacy-Systems, indem Sie hochrangige Aktivitäten von links nach rechts und detaillierte Nutzeraufgaben von oben nach unten anordnen.
- Identifizieren Sie Lücken im aktuellen Legacy-System, wo Nutzer auf Workarounds, manuelle Prozesse oder Schattensysteme angewiesen sind, um ihre Arbeit zu erledigen — diese Lücken repräsentieren hochpriore Verbesserungsmöglichkeiten.
- Ziehen Sie eine Release-Linie über die Karte, um den minimalen tragfähigen Ersatz zu definieren: die kleinste Teilmenge von Funktionalität, die das Legacy-System für mindestens eine Nutzergruppe ersetzen kann.
- Nutzen Sie die Karte, um Gespräche zwischen Entwicklern, Product Ownern und Nutzern darüber zu erleichtern, was zuerst gebaut werden soll, und machen Sie Tradeoff-Entscheidungen sichtbar statt in einem flachen Backlog versteckt.
- Aktualisieren Sie die Story Map, während die Modernisierung fortschreitet, um zu verfolgen, welche Bereiche migriert wurden und welche im Legacy-System verbleiben.
- Kennzeichnen Sie Stories farblich nach Migrationsrisiko oder -komplexität, um technische Herausforderungen während Planungsdiskussionen zutage zu bringen.

## Tradeoffs ⇄

> Story Mapping bietet einen ganzheitlichen Blick auf den Modernisierungsumfang, erfordert aber Moderationsfähigkeit und laufende Pflege.

**Vorteile:**

- Verhindert den häufigen Modernisierungsfehler, Features in einer Reihenfolge zu bauen, die technisch Sinn ergibt, aber Nutzer daran hindert, End-to-End-Workflows abzuschließen.
- Macht den vollständigen Umfang eines Legacy-Ersatzes in einer einzigen Ansicht sichtbar und hilft Stakeholdern zu verstehen, warum Modernisierung Zeit braucht.
- Ermöglicht inkrementelle Lieferung durch Identifikation bedeutsamer Release-Scheiben, die Nutzern Wert bieten, bevor das gesamte System fertig ist.
- Bringt versteckte Abhängigkeiten zwischen Features zutage, die ein flaches Backlog verschleiert.

**Kosten und Risiken:**

- Die Erstellung der anfänglichen Story Map für ein großes Legacy-System ist ein erheblicher Moderationsaufwand, der mehrere Workshops mit verschiedenen Stakeholdern erfordert.
- Story Maps können bei sehr großen Systemen unhandlich werden und müssen möglicherweise in mehrere Karten aufgeteilt werden, die die ganzheitliche Perspektive verlieren.
- Ohne regelmäßige Aktualisierungen wird die Karte veraltet und verliert ihren Wert als Planungswerkzeug.
- Teams, die mit der Technik nicht vertraut sind, könnten Schwierigkeiten haben, die richtige Granularitätsebene für Stories zu finden.

## How It Could Be

> Das folgende Szenario zeigt, wie Story Mapping einen phasenweisen Legacy-Ersatz leitet.

Ein Immobilienverwaltungsunternehmen ersetzte ein Legacy-System, das von 200 Immobilienverwaltern genutzt wurde. Ein flaches Backlog von 800 User Stories machte es unmöglich zu bestimmen, was zuerst geliefert werden sollte. Das Team führte einen zweitägigen Story-Mapping-Workshop durch, der die gesamte Funktionalität entlang des täglichen Workflows des Immobilienverwalters organisierte: Immobilien auflisten, Mieter prüfen, Mietverträge verwalten, Wartungsanfragen bearbeiten und Zahlungen verarbeiten. Die Karte offenbarte, dass die Wartungsanfragenverwaltung der schmerzhafteste Bereich im Legacy-System war und als eigenständiges Modul geliefert werden konnte, das Immobilienverwalter sofort übernehmen würden. Durch die zuerst erfolgte Lieferung der Wartungsverwaltung baute das Team Glaubwürdigkeit und Nutzervertrauen auf, das die Übernahme nachfolgender Module erleichterte. Die Story Map offenbarte auch, dass das flache Backlog 120 Stories enthielt, die sich auf ein Berichtsfeature bezogen, das nur fünf Nutzer benötigten, was dem Team half, diese Arbeit auf ein späteres Release zu verschieben.
