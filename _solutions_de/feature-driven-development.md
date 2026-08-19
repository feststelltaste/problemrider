---
title: Feature-Driven Development
description: Strukturierung und Umsetzung von Softwarefunktionalität in Form von
  Features.
category:
- Process
- Management
problems:
- slow-feature-development
- poor-planning
- unclear-goals-and-priorities
- large-feature-scope
- delayed-value-delivery
- planning-dysfunction
layout: solution
lang: de
en_slug: feature-driven-development
related_solutions:
- slug: iterative-development
  similarity: 0.75
- slug: behavior-driven-development-bdd
  similarity: 0.75
- slug: continuous-delivery
  similarity: 0.75
- slug: user-stories
  similarity: 0.75
- slug: architecture-roadmap
  similarity: 0.7
- slug: strangler-fig-pattern
  similarity: 0.7
---

## Description

Feature-Driven Development strukturiert Arbeit um eine laufende Liste kleiner, kundenwertiger Features, jedes in einer konsistenten Form ausgedrückt wie „die Gesamtprämie für eine Vertragsverlängerung berechnen", mit individueller Eigentümerschaft pro Feature zugewiesen und Fortschritt verfolgt durch kurze Iterationen fester Länge, die jeweils eine Menge vollständig abgeschlossener Features End-to-End liefern. Legacy-Modernisierungsvorhaben stocken oft am gegenteiligen Muster: breite, vage umrissene Initiativen wie „das Lagermodul verbessern", die nie eine sichtbare, vorführbare Fortschrittseinheit produzieren und Stakeholdern nichts Konkretes geben, wogegen sie bewerten oder neu priorisieren könnten. Die Zerlegung von Modernisierungsarbeit in eine nach Geschäftsfähigkeit organisierte Feature-Liste gibt Stakeholdern einen Fertigstellungsgrad, den sie tatsächlich interpretieren können, und gibt dem Team einen natürlichen Mechanismus zur Neupriorisierung mitten im Vorhaben — Fokus auf welchen Feature-Bereich auch immer sich als wichtiger herausstellt zu verschieben —, ohne bei jeder Prioritätsverschiebung eine vollständige Neuplanung zu benötigen. Weil Features die Einheit sowohl der Planung als auch der Eigentümerschaft sind, passen übergreifende Belange wie Sicherheit, Performance oder Infrastrukturarbeit nicht sauber ins Modell und müssen separat verfolgt werden, und individuelle Feature-Eigentümerschaft kann selbst neue Wissenssilos schaffen, sofern sie nicht bewusst mit Code-Review- und Wissensaustausch-Praktiken gepaart wird.

## How to Apply ◆

- Zerlegen Sie Legacy-System-Verbesserungen in kundenwertige Features, ausgedrückt als „<Aktion> das <Ergebnis> <für|von|zu> ein(e) <Objekt>" (z. B. „Die Gesamtprämie für eine Vertragsverlängerung berechnen").
- Bauen Sie eine Feature-Liste auf, die als Backlog für Legacy-Modernisierungsarbeit dient, organisiert nach Geschäftsbereich.
- Weisen Sie Feature-Eigentümerschaft einzelnen Entwicklern zu, die für den Entwurf und die Umsetzung jedes Features End-to-End verantwortlich sind.
- Planen Sie Arbeit in Zwei-Wochen-Iterationen, in denen jede Iteration eine Menge vollständig abgeschlossener Features liefert.
- Verfolgen Sie den Fortschritt anhand von Feature-Fertigstellungsgraden und geben Sie Stakeholdern Sichtbarkeit auf den Modernisierungsfortschritt.
- Nutzen Sie Design-by-Feature- und Build-by-Feature-Phasen, um sicherzustellen, dass jedes Feature vor der Umsetzung ordentlich entworfen wird.

## Tradeoffs ⇄

**Vorteile:**
- Hält Modernisierungsvorhaben fokussiert auf die Lieferung greifbaren Geschäftswerts durch abgeschlossene Features.
- Bietet klare Fortschrittsverfolgung, die Stakeholder verstehen können.
- Verhindert Scope Creep, indem Features mit spezifischen, messbaren Ergebnissen definiert werden.
- Weist klare Eigentümerschaft zu, was den Koordinationsaufwand verringert.

**Kosten:**
- Übergreifende Belange (Sicherheit, Performance, Infrastruktur) passen nicht sauber in Features.
- Feature-Zerlegung erfordert Verständnis sowohl der Geschäftsdomäne als auch des Legacy-Systems.
- Individuelle Feature-Eigentümerschaft kann Wissenssilos schaffen, wenn sie nicht mit Code-Reviews und Wissensaustausch kombiniert wird.
- Legacy-Modernisierung beinhaltet oft fundamentale Arbeit, die nicht direkt feature-sichtbar ist.

## How It Could Be

Ein Modernisierungsvorhaben für ein Legacy-Lagerverwaltungssystem stockt, weil das Team an breiten, undefinierten Aufgaben wie „das Lagermodul verbessern" arbeitet. Das Team wechselt zu Feature-Driven Development und erstellt eine Feature-Liste mit 120 spezifischen, nach Geschäftsfähigkeit organisierten Features. In jeder Zwei-Wochen-Iteration wählt das Team Features zur Umsetzung aus, entwirft sie kurz und baut sie bis zur Fertigstellung. Stakeholder können sehen, dass 45 % der Wareneingangs-Features modernisiert sind, 20 % der Einlagerungs-Features und 0 % der Inventurzählungs-Features. Diese Sichtbarkeit erlaubt dem Unternehmen, neu zu priorisieren: Inventurzählung ist dringlicher als Einlagerung, also verschiebt das Team den Fokus. Innerhalb von sechs Monaten sind die geschäftskritischsten Features modernisiert, während weniger priorisierte Bereiche auf dem Legacy-System verbleiben, was den innerhalb des verfügbaren Budgets gelieferten Wert maximiert.
