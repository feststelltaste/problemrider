---
title: User Stories
description: Formulierung von Anforderungen aus Nutzerperspektive.
category:
- Requirements
problems:
- requirements-ambiguity
- inadequate-requirements-gathering
- misaligned-deliverables
- stakeholder-developer-communication-gap
- feature-bloat
- large-feature-scope
- implementation-rework
layout: solution
lang: de
en_slug: user-stories
related_solutions:
- slug: story-mapping
  similarity: 0.85
- slug: personas
  similarity: 0.8
- slug: user-centered-design
  similarity: 0.8
- slug: requirements-analysis
  similarity: 0.8
- slug: behavior-driven-development-bdd
  similarity: 0.75
- slug: prototypes
  similarity: 0.75
---

## Description

Eine User Story rahmt ein Stück benötigter Funktionalität aus der Perspektive der Person, die sie nutzen wird — typischerweise in der Form "Als [Rolle] möchte ich [Fähigkeit], damit [Wert]" — und erzwingt eine explizite Aussage darüber, warum eine Fähigkeit zählt, statt einfach was sie tun soll. Diese Rahmung ist eine bewusste Korrektur für eines der beständigsten Fehlermuster in Legacy-Modernisierungsprojekten: jeden bestehenden Bildschirm, jedes Feld und jeden Batch-Prozess im Legacy-System als unhinterfragte Anforderung zu behandeln, die getreu reproduziert werden muss, unter der Annahme, dass, wenn es vorher existierte, es gebraucht werden muss. Viele Legacy-Features existieren jedoch nicht, weil ein Nutzer sie genuin braucht, sondern weil sie eine technische Einschränkung des alten Systems kompensieren — ein manueller Neuberechnungs-Trigger, der nur existiert, weil der Legacy-Batch-Job nicht häufig genug laufen konnte, zum Beispiel —, und sie pauschal zu reproduzieren trägt Komplexität weiter, die eine moderne Architektur möglicherweise überhaupt nicht braucht. Anforderungen als User Stories zu schreiben, validiert gegen das, was ein Nutzer tatsächlich zu erreichen versucht, bringt diese Fälle zutage und lässt das Team bewusst entscheiden, ob ein Stück Legacy-Funktionalität die Migration aus eigenem Recht überlebt statt standardmäßig. Dieser Ansatz ermöglicht auch inkrementelle Lieferung, da Stories aufgeteilt, nach Wert und Migrationsrisiko priorisiert und unabhängig validiert werden können, was der Modernisierungsanstrengung kontinuierliche Evidenz gibt, dass sie in die richtige Richtung geht, statt alles auf eine einzige Big-Bang-Umschaltung zu setzen.

## How to Apply ◆

> In der Legacy-Modernisierung verschieben User Stories den Fokus vom Replizieren technischer Features zum Liefern von Nutzerwert, was die häufige Falle verhindert, alles neu zu bauen "weil das alte System es hatte".

- Schreiben Sie User Stories für das Ersatzsystem basierend darauf, was Nutzer erreichen müssen, nicht darauf, wie die Bildschirme und Funktionen des Legacy-Systems aussehen.
- Nutzen Sie das Format "Als [Nutzerrolle] möchte ich [Fähigkeit], damit [Geschäftswert]", um das Team zu zwingen zu artikulieren, warum jedes Stück Funktionalität zählt.
- Teilen Sie Legacy-Systemfeatures in User Stories auf, die unabhängig geliefert und validiert werden können, was inkrementelle Migration statt Big-Bang-Ersatz ermöglicht.
- Beziehen Sie Abnahmekriterien für jede Story ein, die klare, testbare Zufriedenheitsbedingungen basierend auf Geschäftsergebnissen definieren.
- Beziehen Sie Nutzer in Story-Schreib-Workshops ein, um Anforderungen zu erfassen, die nur als stillschweigendes Wissen im Legacy-System existieren.
- Priorisieren Sie Stories basierend auf Nutzerwert und Migrationsrisiko statt technischer Bequemlichkeit, um sicherzustellen, dass die kritischsten Nutzerbedürfnisse zuerst adressiert werden.
- Nutzen Sie Story-Splitting-Techniken, um Stories klein genug für Einzel-Iterations-Lieferung zu halten, während bedeutsamer Nutzerwert erhalten bleibt.

## Tradeoffs ⇄

> User Stories halten die Entwicklung auf Nutzerwert fokussiert, erfordern aber laufende Verfeinerung und können herausfordernd zu schreiben sein für komplexe Legacy-Geschäftslogik.

**Vorteile:**

- Verhindert Feature-Bloat während der Modernisierung, indem explizite Rechtfertigung für jede Fähigkeit verlangt wird, statt Legacy-Features blind zu replizieren.
- Ermöglicht inkrementelle Lieferung und Validierung, was Nutzern erlaubt, Feedback zu abgeschlossenen Stories zu geben, bevor das gesamte System gebaut ist.
- Schafft eine gemeinsame Sprache zwischen Entwicklern und Stakeholdern, die sich auf Ergebnisse statt technische Implementierungsdetails konzentriert.
- Macht Priorisierungsentscheidungen transparent, indem jede Story mit einem Nutzerbedürfnis und Geschäftswert verbunden wird.

**Kosten und Risiken:**

- Komplexe Legacy-Geschäftslogik könnte schwer als User Stories auszudrücken sein, ohne wichtige Nuancen und Grenzfälle zu verlieren.
- Stories, die ohne ausreichendes Domänenverständnis geschrieben werden, könnten kritisches Legacy-Verhalten übersehen, das Nutzer für selbstverständlich halten.
- Übermäßiges Aufteilen von Stories, um in Sprint-Zeitfenster zu passen, kann Nutzer-Workflows in zu kleine Stücke fragmentieren, um sie sinnvoll zu validieren.
- Teams könnten Stories schreiben, die dünn verschleierte technische Aufgaben sind, statt genuine Nutzerwert-Ausdrücke.

## How It Could Be

> Das folgende Szenario demonstriert, wie User Stories Legacy-Modernisierungsentscheidungen leiten.

Eine Kreditgenossenschaft ersetzte ihr Legacy-Kreditvergabesystem. Das Legacy-System hatte 47 Bildschirme, und der anfängliche Plan war, jeden Bildschirm neu zu bauen. Als das Team diese als User Stories aus der Perspektive des Kreditsachbearbeiters neu schrieb, entdeckten sie, dass 12 Bildschirme nur existierten, um Einschränkungen der Batch-Verarbeitung des Legacy-Systems zu umgehen — sie wurden genutzt, um manuell Neuberechnungen auszulösen, die das neue System automatisch durchführen konnte. Durch den Fokus auf User Stories statt Bildschirmreplikation eliminierte das Team 25 % der geplanten Arbeit, während es tatsächlich den Workflow des Kreditsachbearbeiters verbesserte. Die Story "Als Kreditsachbearbeiter möchte ich die aktualisierte monatliche Zahlung sofort sehen, wenn ich den Zinssatz ändere, damit ich Optionen mit dem Mitglied in Echtzeit besprechen kann" ersetzte drei Legacy-Bildschirme und einen Batch-Prozess durch eine einzige responsive Berechnung.
