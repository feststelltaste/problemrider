---
title: Anpassbare Nutzeroberfläche
description: Nutzern erlauben, die Nutzeroberfläche nach ihren Präferenzen zu ändern.
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/customizable-user-interface/
problems:
- poor-user-experience-ux-design
- user-frustration
- shadow-systems
- feature-gaps
- negative-user-feedback
- increased-cognitive-load
- customer-dissatisfaction
- user-confusion
layout: solution
lang: de
en_slug: customizable-user-interface
related_solutions:
- slug: custom-views
  similarity: 0.8
- slug: intuitive-navigation
  similarity: 0.75
- slug: customizing
  similarity: 0.7
- slug: visual-hierarchy
  similarity: 0.7
- slug: cognitive-load-minimization
  similarity: 0.7
- slug: user-centered-design
  similarity: 0.7
---

## Description

Eine anpassbare Nutzeroberfläche erlaubt einzelnen Nutzern, Layout, Design, Benachrichtigungsverhalten und Tastenkürzel an ihre eigenen Präferenzen und Arbeitsabläufe anzupassen, statt jedem Nutzer die eine starre Anordnung zu präsentieren, die die meisten Legacy-Systeme vorgeben. Genau diese Starrheit treibt Nutzer zu Workarounds — kleine fixe Schriften anstrengen, irrelevante Dashboard-Widgets ignorieren, externe Werkzeuge zum Ausgleich bauen —, weil die offizielle Oberfläche sich nicht daran anpassen kann, wie sie tatsächlich arbeiten. Diese Präferenzen pro Konto zu persistieren, sodass sie dem Nutzer über Sitzungen und Geräte hinweg folgen, erhöht sowohl die Produktivität als auch ein Gefühl der Eigentümerschaft über das Werkzeug, obgleich der kombinatorische Raum der Konfigurationen, den es schafft, gründliches Testing und Support merklich erschwert.

## How to Apply ◆

> Legacy-Systeme erzwingen starre Oberflächen, die sich nicht an individuelle Nutzerpräferenzen anpassen können. Die Erlaubnis zur Anpassung befähigt Nutzer, effizienter innerhalb des Systems zu arbeiten, statt darum herum.

- Erlauben Sie Nutzern, Dashboard-Widgets, Panels und Bereiche per Drag-and-Drop umzuordnen, um die für ihren Arbeitsablauf relevantesten Informationen zu priorisieren.
- Unterstützen Sie Design-Präferenzen einschließlich Schriftgröße, Farbschemata und Kontrastmodi. Legacy-Systeme haben oft kleine, feste Schriftgrößen, die bei langen Arbeitsstunden Belastung verursachen.
- Lassen Sie Nutzer Benachrichtigungspräferenzen konfigurieren, um zu steuern, welche Systemereignisse Alarme erzeugen und wie diese Alarme zugestellt werden.
- Erlauben Sie Anpassung von Tastenkürzeln, sodass Nutzer häufig genutzte Aktionen auf Tastenkombinationen abbilden können, die zu ihren Gewohnheiten aus anderen Werkzeugen passen.
- Bieten Sie Dichte-Einstellungen, die Nutzern erlauben, zwischen kompakten Ansichten für erfahrene Nutzer, die mehr Daten sehen wollen, und geräumigen Ansichten für neue Nutzer, die mehr visuellen Freiraum brauchen, zu wählen.
- Speichern Sie alle Anpassungspräferenzen pro Nutzerkonto, sodass die personalisierte Erfahrung über Sitzungen und Geräte hinweg bestehen bleibt.

## Tradeoffs ⇄

> Anpassbare Oberflächen erfüllen diverse Nutzerbedürfnisse, erhöhen aber die Testfläche und Support-Komplexität.

**Vorteile:**

- Befähigt Nutzer, die Oberfläche für ihre spezifische Rolle und Präferenzen zu optimieren, was Produktivität und Zufriedenheit erhöht.
- Reduziert den Bedarf an Schattensystemen, weil Nutzer das offizielle System an ihre Bedürfnisse anpassen können, statt externe Workarounds zu bauen.
- Berücksichtigt Nutzer mit unterschiedlichen Barrierefreiheitsbedürfnissen durch konfigurierbare Schriftgrößen, Farbschemata und Interaktionsmodi.
- Erhöht das Nutzerengagement mit dem System, weil Nutzer, die Zeit in die Anpassung ihres Arbeitsbereichs investieren, ein Gefühl der Eigentümerschaft entwickeln.

**Kosten und Risiken:**

- Das Testen aller möglichen Anpassungskombinationen ist unpraktikabel, was das Risiko von Layout-Fehlern und visuellen Störungen in ungewöhnlichen Konfigurationen erhöht.
- Support-Personal kann die exakte Oberflächenkonfiguration eines Nutzers bei der Fehlersuche nicht leicht nachbilden, was Support zeitaufwendiger macht.
- Übermäßige Anpassungsoptionen können selbst zu einer Quelle kognitiver Überlastung werden, wenn sie nicht durchdacht organisiert und präsentiert werden.
- Der Aufbau von Anpassungsinfrastruktur in ein Legacy-System mit starrer Frontend-Architektur kann erhebliche Refaktorierung erfordern.
- Nutzer, die übermäßig anpassen, könnten unbeabsichtigt wichtige Funktionalität ausblenden und sie dann als fehlend melden.

## How It Could Be

> Wenn Nutzer das offizielle System nicht anpassen können, bauen sie ihre eigenen Lösungen, was Daten und Arbeitsabläufe fragmentiert.

Ein Legacy-Kundenbeziehungsmanagementsystem hat ein fixes Dashboard, das jedem Nutzer dieselben sechs Kennzahlen zeigt: Vertriebsmitarbeiter, Support-Mitarbeiter und Manager gleichermaßen. Vertriebsmitarbeiter kümmern sich um Pipeline und Quotenfortschritt, Support-Mitarbeiter brauchen Ticket-Warteschlangen und Lösungszeiten, und Manager wollen teambezogene Zusammenfassungen. Weil keiner von ihnen auf einen Blick sieht, was er braucht, haben alle drei Gruppen persönliche Tabellenkalkulationen und Lesezeichensammlungen gebaut, um ihre eigenen Dashboards aus Rohdatenexporten zusammenzustellen. Das Team fügt ein anpassbares Dashboard mit konfigurierbarer Widget-Platzierung und Datenquellenauswahl hinzu. Jede Nutzergruppe erstellt ein auf ihre Rolle zugeschnittenes Dashboard-Layout. Innerhalb von zwei Monaten geht die Nutzung externer Tabellenkalkulations-Dashboards erheblich zurück, und die Datenkonsistenz verbessert sich, weil alle mit derselben Live-Datenquelle arbeiten.
