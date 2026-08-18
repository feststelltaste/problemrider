---
title: Erhöhte kognitive Last
description: Entwickler müssen übermäßig viel mentale Energie aufwenden, um inkonsistenten,
  komplexen oder schlecht strukturierten Code zu verstehen und mit ihm zu arbeiten.
category:
- Code
- Process
related_problems:
- slug: cognitive-overload
  similarity: 0.85
- slug: mental-fatigue
  similarity: 0.75
- slug: difficult-to-understand-code
  similarity: 0.7
- slug: difficult-developer-onboarding
  similarity: 0.7
- slug: difficult-code-comprehension
  similarity: 0.7
- slug: high-technical-debt
  similarity: 0.7
solutions:
- clean-code
- loose-coupling
- separation-of-concerns
- cognitive-load-minimization
- consistent-user-interface
- customizable-user-interface
- form-design
- strategic-code-deletion
- intuitive-navigation
- progressive-disclosure
- visual-hierarchy
layout: problem
lang: de
en_slug: increased-cognitive-load
---

## Description

Erhöhte kognitive Last tritt auf, wenn Entwickler übermäßige mentale Ressourcen nutzen müssen, um Code zu verstehen, zu navigieren und zu ändern. Dies geschieht, wenn Codebasen inkonsistent, übermäßig komplex, schlecht organisiert sind oder klare Muster und Konventionen fehlen. Hohe kognitive Last führt zu Entwicklerermüdung, erhöhten Fehlerraten und langsamerer Entwicklungsgeschwindigkeit. Sie ist besonders problematisch in Legacy-Systemen, in denen sich über die Zeit mehrere Coding-Stile, Muster und architektonische Entscheidungen ohne kohärente Organisation angehäuft haben.

## Indicators ⟡
- Entwickler brauchen länger als erwartet, um scheinbar einfache Aufgaben abzuschließen
- Teammitglieder bitten häufig um Hilfe beim Verständnis bestehenden Codes
- Code-Reviews dauern ungewöhnlich lange, weil Reviewer Schwierigkeiten haben, die Änderungen zu verstehen
- Neue Teammitglieder haben Schwierigkeiten, produktiv zu werden, selbst nach ausgedehntem Onboarding
- Entwickler äußern Frustration über die Schwierigkeit, mit bestimmten Teilen der Codebasis zu arbeiten

## Symptoms ▲

- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Wenn Entwickler übermäßige mentale Energie für das Verständnis von Code aufwenden, schließen sie Aufgaben langsamer ab.
- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Mentale Überlastung erhöht die Wahrscheinlichkeit, dass Entwickler Code missverstehen und Defekte einführen.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Ständiges Kämpfen mit dem Verständnis komplexen und inkonsistenten Codes führt zu Frustration und mentaler Erschöpfung.
- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Hohe kognitive Last macht es besonders schwierig für neue Entwickler, in der Codebasis produktiv zu werden.
- [Verringerte individuelle Produktivität](verringerte-individuelle-produktivitaet.md)
<br/>  Entwickler schließen weniger Aufgaben ab, weil ein Großteil ihres Aufwands ins Verstehen statt ins Erschaffen von Code fließt.
- [Mentale Erschöpfung](mentale-erschoepfung.md)
<br/>  Übermäßige kognitive Anforderungen laugen Entwickler mental aus und lassen sie erschöpft ohne bedeutende Errungenschaft zurück.

## Causes ▼

- [Inkonsistente Coding-Standards](inkonsistente-coding-standards.md)
<br/>  Inkonsistente Konventionen zwingen Entwickler, sich ständig an unterschiedliche Muster über die Codebasis hinweg anzupassen.
- [Komplexe und unklare Logik](komplexe-und-unklare-logik.md)
<br/>  Verworrene Logik, die schwer nachzuvollziehen ist, zwingt Entwickler, zusätzliche mentale Energie aufzuwenden, um das Codeverhalten zu verstehen.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Eng gekoppelte Komponenten erfordern von Entwicklern, viele miteinander verbundene Teile gleichzeitig zu verstehen.
- [Inkonsistente Namenskonventionen](inkonsistente-namenskonventionen.md)
<br/>  Unvorhersehbare Namensmuster fügen unnötigen mentalen Overhead beim Navigieren und Verstehen von Code hinzu.

## Detection Methods ○
- **Zeittracking:** Beobachtung, wie lange einfache Aufgaben im Vergleich zu Schätzungen oder historischen Durchschnitten dauern
- **Entwicklerbefragungen:** Befragung von Teammitgliedern zu ihrer Erfahrung bei der Arbeit mit unterschiedlichen Teilen der Codebasis
- **Code-Komplexitätsmetriken:** Nutzung von Werkzeugen zur Messung zyklomatischer Komplexität, Verschachtelungstiefe und Funktionslänge
- **Onboarding-Zeit:** Nachverfolgung, wie lange neue Entwickler brauchen, um in unterschiedlichen Bereichen des Systems produktiv zu werden
- **Code-Review-Dauer:** Beobachtung, wie lange Code-Reviews dauern, besonders für scheinbar einfache Änderungen

## Examples

Ein Entwickler muss eine einfache Validierungsregel zu einem Nutzerregistrierungsformular hinzufügen. Die bestehende Codebasis hat Validierung auf vier unterschiedliche Arten über unterschiedliche Module hinweg implementiert: manche nutzen eine Drittanbieter-Bibliothek, andere nutzen benutzerdefinierte Validierungsklassen, manche betten Validierungslogik direkt in Controller ein, und ein Modul nutzt ein völlig anderes Framework. Um die neue Validierung konsistent mit dem Registrierungsmodul hinzuzufügen, muss der Entwickler zunächst Stunden damit verbringen zu verstehen, welchen Ansatz dieses spezifische Modul nutzt, und dann die für diesen Ansatz spezifischen Muster und Konventionen lernen. Was eine 15-Minuten-Aufgabe sein sollte, wird zu einer mehrstündigen Untersuchung. Ein weiteres Beispiel betrifft ein Finanzberechnungsmodul, bei dem Geschäftslogik über 12 unterschiedliche Dateien mit variierenden Namenskonventionen verstreut ist, was es nahezu unmöglich macht, den vollständigen Berechnungsfluss zu verstehen, ohne mehrere Dateien gleichzeitig zu öffnen und eine mentale Landkarte davon zu pflegen, wie sie interagieren.
