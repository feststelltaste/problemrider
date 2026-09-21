---
title: Konsistente Nutzeroberfläche
description: Vereinheitlichung von Design und Verhalten der Nutzeroberfläche über
  alle Software-Teile hinweg.
category:
- Requirements
- Architecture
quality_tactics_url: https://qualitytactics.de/en/usability/consistent-user-interface/
problems:
- poor-user-experience-ux-design
- inconsistent-behavior
- user-confusion
- user-frustration
- inconsistent-codebase
- negative-user-feedback
- shadow-systems
- increased-cognitive-load
layout: solution
lang: de
en_slug: consistent-user-interface
related_solutions:
- slug: consistent-terminology
  similarity: 0.85
- slug: style-guide
  similarity: 0.85
- slug: user-centered-design
  similarity: 0.85
- slug: intuitive-navigation
  similarity: 0.8
- slug: cognitive-load-minimization
  similarity: 0.8
- slug: a-b-testing
  similarity: 0.75
---

## Description

Eine konsistente Nutzeroberfläche wendet dieselben Komponenten, Interaktionsmuster und Navigationsstrukturen über jeden Teil einer Anwendung an, sodass ein Nutzer, der einen Bildschirm lernt, bereits weiß, wie sich der Rest des Systems verhält. Legacy-Systeme, gebaut von verschiedenen Teams über viele Jahre, enden routinemäßig mit mehreren nicht verwandten Navigationsparadigmen und visuellen Stilen, die im selben Produkt koexistieren, was Nutzer zwingt, die Schnittstelle jedes Mal neu zu lernen, wenn sie zwischen Modulen wechseln. Die Einführung einer gemeinsamen Komponentenbibliothek und eines Style Guides, und ihre rückwirkende Anwendung zuerst auf die am stärksten genutzten Bildschirme, schließt diese Lücke schrittweise, ohne eine vollständige Neuschreibung jedes Moduls auf einmal zu erfordern.

## How to Apply ◆

> Legacy-Systeme, entwickelt über viele Jahre von verschiedenen Teams, haben oft stark inkonsistente Schnittstellen. Die Etablierung von UI-Konsistenz verringert die Lernkurve und baut Nutzervertrauen auf.

- Erstellen Sie eine gemeinsame Komponentenbibliothek oder ein Design-System, das standardisierte UI-Elemente für Buttons, Formulare, Tabellen, Navigation und Dialoge bietet. Jede neue Entwicklung und Änderung an bestehenden Bildschirmen sollte Komponenten aus dieser Bibliothek nutzen.
- Dokumentieren Sie Interaktionsmuster für übliche Aktionen wie Erstellen, Bearbeiten, Löschen, Suchen und Filtern. Stellen Sie sicher, dass diese Muster über alle Module des Legacy-Systems identisch sind.
- Auditieren Sie die bestehende Schnittstelle auf Inkonsistenzen in Layout, Farbnutzung, Typografie, Icon-Bedeutung und Button-Platzierung. Priorisieren Sie die Behebung von Inkonsistenzen in den am häufigsten genutzten Bildschirmen.
- Etablieren Sie einen Style Guide, der Abstände, Ausrichtung, responsives Verhalten und Fehlerdarstellung abdeckt, und machen Sie ihn für alle Entwickler zugänglich, die am Legacy-System arbeiten.
- Standardisieren Sie Navigationsmuster, sodass Nutzer vorhersagen können, wo sie Funktionalität finden, unabhängig davon, in welchem Modul sie sich befinden. Legacy-Systeme haben oft völlig unterschiedliche Navigationsstrukturen in verschiedenen Abschnitten.
- Erhalten Sie bei schrittweiser Modernisierung visuelle Konsistenz zwischen aktualisierten und noch nicht aktualisierten Abschnitten, indem Sie das Design-System rückwirkend auf unveränderte Bereiche anwenden, wo machbar.

## Tradeoffs ⇄

> Eine konsistente UI verbessert die Nutzbarkeit dramatisch und verringert Schulungskosten, aber das Erreichen von Konsistenz in einem großen Legacy-System erfordert anhaltenden Aufwand.

**Vorteile:**

- Nutzer lernen eine Reihe von Interaktionsmustern und können sie über die gesamte Anwendung hinweg anwenden, was kognitive Last und Verwirrung verringert.
- Verringert die Anzahl der Support-Anfragen, verursacht durch Nutzer, die Funktionalität nicht finden können, weil sie in verschiedenen Abschnitten unterschiedlich präsentiert wird.
- Beschleunigt die Entwicklung, weil Entwickler standardisierte Komponenten wiederverwenden, statt UI-Muster für jedes Modul neu zu erfinden.
- Eliminiert Schattensysteme, gebaut, um verwirrende oder inkonsistente offizielle Schnittstellen zu umgehen.

**Kosten und Risiken:**

- Der Bau und die Pflege eines Design-Systems erfordert Vorabinvestition in Design- und Entwicklungsressourcen.
- Die Nachrüstung von Konsistenz auf ein Legacy-System mit vielfältigen Technologie-Stacks könnte erhebliche Refaktorierung in Modulen erfordern, die mit unterschiedlichen UI-Frameworks gebaut wurden.
- Langjährige Nutzer, die sich an die Inkonsistenzen angepasst haben, könnten vorübergehende Störung erleben, wenn vertraute Bildschirme sich ändern.
- Die Durchsetzung von Konsistenz über autonome Teams hinweg erfordert Governance und Zusammenarbeit, die in Organisationen mit starken Team-Silos möglicherweise nicht existiert.

## How It Could Be

> Schnittstelleninkonsistenz in Legacy-Systemen ist oft für das Entwicklungsteam unsichtbar, aber für Nutzer, die über mehrere Module hinweg arbeiten, schmerzhaft offensichtlich.

Das Legacy-ERP-System eines Fertigungsunternehmens wurde von vier separaten Teams über zwölf Jahre gebaut. Das Bestandsmodul nutzt Seitenleisten-Navigation mit ausklappbaren Baummenüs, das Einkaufsmodul nutzt eine obere Symbolleiste mit Dropdown-Menüs, und das Versandmodul nutzt ein Tabbed Interface mit Breadcrumbs. Nutzer, die über alle drei Module hinweg arbeiten, verschwenden Zeit damit, sich jedes Mal neu zu orientieren, wenn sie den Kontext wechseln. Das Team führt eine gemeinsame Komponentenbibliothek ein, basierend auf dem Navigationsmuster des Einkaufsmoduls, das Nutzerforschung als das intuitivste identifizierte. Über drei vierteljährliche Releases übernehmen alle Module die gemeinsame Navigation, Formularlayouts und Tabellenkomponenten. Nutzerzufriedenheitsumfragen zeigen eine messbare Verbesserung, und die Schulungszeit für neue Mitarbeiter sinkt, weil Trainer nicht mehr drei verschiedene Schnittstellenparadigmen erklären müssen.
