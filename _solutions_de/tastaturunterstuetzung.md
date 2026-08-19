---
title: Tastaturunterstützung
description: Bedienbarkeit der Software über die Tastatur ermöglichen.
category:
- Requirements
- Code
quality_tactics_url: https://qualitytactics.de/en/usability/keyboard-support/
problems:
- poor-user-experience-ux-design
- user-frustration
- regulatory-compliance-drift
- negative-user-feedback
- feature-gaps
- user-confusion
layout: solution
lang: de
en_slug: keyboard-support
related_solutions:
- slug: assistive-technology-support
  similarity: 0.75
- slug: focus-management
  similarity: 0.7
- slug: intuitive-navigation
  similarity: 0.7
- slug: accessibility-concept
  similarity: 0.7
- slug: responsive-design
  similarity: 0.65
- slug: customizable-user-interface
  similarity: 0.65
---

## Description

Tastaturunterstützung macht jedes interaktive Element einer Anwendung vollständig ohne Maus bedienbar — erreichbar in logischer Tab-Reihenfolge, auslösbar mit Enter oder Leertaste, navigierbar in benutzerdefinierten Steuerelementen mit Pfeiltasten — statt anzunehmen, wie Legacy-Oberflächen, gebaut für Maus-und-Desktop-Nutzung, häufig tun, dass Klicken der einzige Weg hinein ist. Diese Annahme schließt Nutzer mit motorischen Behinderungen vollständig aus und verlangsamt alle anderen, die mit einem Tastaturkürzel schneller wären, weshalb volle Tastaturbedienbarkeit gleichzeitig als Barrierefreiheitsanforderung und als echtes Produktivitätsfeature fungiert. Sie in benutzerdefinierte Legacy-Steuerelemente — eine Drag-and-Drop-Liste, eine maßgeschneiderte Datumsauswahl —, die nur mit Mausinteraktion im Blick gebaut wurden, nachzurüsten ist üblicherweise der schwierigste und refactoring-intensivste Teil der Arbeit.

## How to Apply ◆

> Legacy-Systeme erfordern oft Mausinteraktion für kritische Operationen, was reine Tastaturnutzer ausschließt und Power-User verlangsamt, die Tastaturkürzel bevorzugen. Volle Tastaturunterstützung ist sowohl Barrierefreiheitsanforderung als auch Produktivitätsfeature.

- Prüfen Sie alle interaktiven Elemente im Legacy-System, um zu verifizieren, dass sie über die Tastatur erreichbar und bedienbar sind. Elemente, die nur auf Klick-Events reagieren, müssen auch Tastatur-Events wie Enter und Leertaste handhaben.
- Implementieren Sie logische Tab-Reihenfolge, die dem visuellen Layout der Seite folgt. Legacy-Systeme mit tabellenbasierten Layouts haben oft Tab-Reihenfolgen, die unvorhersehbar über den Bildschirm springen.
- Fügen Sie Tastaturkürzel für häufig genutzte Aktionen wie Speichern, Suchen, Neu erstellen und Navigation zwischen Bereichen hinzu. Zeigen Sie verfügbare Kürzel in einem auffindbaren Hilfe-Overlay an.
- Stellen Sie sicher, dass alle benutzerdefinierten Steuerelemente, einschließlich Dropdown-Menüs, Datumsauswahl, Modal-Dialogen und Datenrastern, vollständig über die Tastatur mit Pfeiltasten, Enter, Escape und Tab bedienbar sind.
- Ersetzen Sie reine Mausinteraktionen wie Drag-and-Drop durch Tastaturalternativen. Bieten Sie Auf/Ab-Buttons oder einen Neuordnungsbefehl für Listen, die derzeit Ziehen erfordern.
- Testen Sie die Anwendung, indem Sie vollständig mit der Tastatur navigieren und die Maus deaktivieren, um Lücken in der Tastaturunterstützung zu identifizieren.

## Tradeoffs ⇄

> Volle Tastaturunterstützung macht die Anwendung für Power-User zugänglich und effizient, erfordert aber Aufmerksamkeit für jedes interaktive Element.

**Vorteile:**

- Sichert Compliance mit Barrierefreiheitsvorschriften, die verlangen, dass alle Funktionalität über die Tastatur bedienbar ist.
- Ermöglicht Nutzern mit motorischen Behinderungen, die keine Maus nutzen können, das System unabhängig zu bedienen.
- Erhöht die Produktivität für Power-User, die mit Tastaturkürzeln schneller sind als mit Mausnavigation.
- Verbessert die Erfahrung für Nutzer in Umgebungen, in denen Mausnutzung unpraktisch ist, wie Callcenter und medizinische Umgebungen.

**Kosten und Risiken:**

- Tastaturunterstützung in Legacy-benutzerdefinierte Steuerelemente nachzurüsten, die mit reiner Mausinteraktion gebaut wurden, kann erhebliches Refactoring erfordern.
- Konflikte zwischen Tastaturkürzeln und Browser-Kürzeln oder Kürzeln assistiver Technologien müssen sorgfältig verwaltet werden.
- Die Pflege der Tastaturunterstützung, während neue Features hinzugefügt werden, erfordert Bewusstsein und Disziplin von allen Entwicklern.
- Komplexe benutzerdefinierte Steuerelemente wie Drag-and-Drop-Oberflächen benötigen möglicherweise völlig alternative Interaktionsmodi für Tastaturnutzer, was den Entwicklungsaufwand erhöht.

## How It Could Be

> Tastaturzugänglichkeit ist oft die wirkungsvollste Barrierefreiheitsverbesserung, weil sie nicht nur reine Tastaturnutzer, sondern auch Screenreader-Nutzer ermöglicht, die auf Tastaturnavigation angewiesen sind.

Eine Legacy-Callcenter-Anwendung verlangt von Agenten, mit der Maus durch mehrere Dropdown-Menüs und Dialogfelder zu klicken, um jeden Anruf zu protokollieren. Agenten bearbeiten Hunderte Anrufe pro Tag, und die ständige Mausinteraktion trägt zu Verschleißerscheinungen bei. Das Team fügt Tastaturkürzel für die zehn häufigsten Anrufprotokollierungsaktionen hinzu: Ein einzelner Tastendruck öffnet den Anruftypauswähler, Pfeiltasten navigieren die Optionen, und Enter bestätigt. Tab wechselt zwischen Feldern, und ein Kürzel speichert und schließt den Anrufdatensatz. Agenten, die den Tastatur-Workflow übernehmen, berichten von spürbar schnellerer Anrufbearbeitung, und mehrere Agenten mit Verschleißproblemen berichten von verringertem Unbehagen. Die Barrierefreiheitsverbesserung bedeutet auch, dass ein sehbehinderter Agent, der einen Screenreader nutzt, nun zum ersten Mal das Anrufprotokollierungssystem unabhängig bedienen kann.
