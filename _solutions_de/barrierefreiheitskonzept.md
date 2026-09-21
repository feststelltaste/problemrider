---
title: Barrierefreiheitskonzept
description: Gestaltung von Software, um sie für Menschen mit Behinderungen zugänglich
  und nutzbar zu machen.
category:
- Requirements
problems:
- poor-user-experience-ux-design
- customer-dissatisfaction
- regulatory-compliance-drift
- negative-user-feedback
- user-frustration
- feature-gaps
layout: solution
lang: de
en_slug: accessibility-concept
related_solutions:
- slug: assistive-technology-support
  similarity: 0.85
- slug: adaptive-behavior
  similarity: 0.75
- slug: a-b-testing
  similarity: 0.7
- slug: intuitive-navigation
  similarity: 0.7
- slug: user-centered-design
  similarity: 0.7
- slug: consistent-user-interface
  similarity: 0.7
---

## Description

Ein Barrierefreiheitskonzept ist eine bewusste Design- und Engineering-Anstrengung, Software für Menschen mit Behinderungen nutzbar zu machen, die Aspekte wie Tastaturnavigation, Screenreader-Kompatibilität, Farbkontrast und alternativen Text für nicht-textuelle Inhalte abdeckt, typischerweise gemessen gegen einen etablierten Standard wie die WCAG-Richtlinien. Statt Barrierefreiheit als nachträgliche Checkliste zu behandeln, etabliert das Konzept Barrierefreiheit als Designbeschränkung, die Markup, Interaktionsmuster und visuelles Design von Anfang an formt. Legacy-Benutzeroberflächen werden häufig mit nicht-semantischem Markup gebaut — Layout-Tabellen, individuelle Widgets ohne Tastaturunterstützung, div-basierte Steuerelemente ohne ARIA-Rollen —, weil Barrierefreiheit keine weit durchgesetzte Anforderung war, als sie gebaut wurden, was Nutzer assistiver Technologie außerstande lässt, selbst grundlegende Aufgaben zu erledigen. Die Einführung eines Barrierefreiheitskonzepts in ein solches System bedeutet, bestehende Bildschirme gegen einen anerkannten Standard zu auditieren, Behebung nach Nutzungshäufigkeit und Schwere der Barriere zu priorisieren, und semantische Struktur in Markup nachzurüsten, das nie dafür designt wurde, sie zu tragen. Weil diese Arbeit direkt ändert, wie interaktive Elemente strukturiert sind, verbessert sie typischerweise auch die Nutzbarkeit für Nutzer ohne Behinderungen, und in vielen Rechtsordnungen schließt sie zusätzlich eine regulatorische Compliance-Lücke, die rechtliches Risiko birgt, wenn sie unangegangen bleibt. Die Aufrechterhaltung der Verbesserung erfordert einen Barrierefreiheits-Styleguide und anhaltendes Testen mit assistiven Technologien, da neue Legacy-artige Abkürzungen genauso leicht dieselben Barrieren während zukünftiger Entwicklung wieder einführen können.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Auditieren Sie die Legacy-Anwendung gegen WCAG-Standards (Web Content Accessibility Guidelines), um Barrieren zu identifizieren
- Priorisieren Sie die Behebung der wirkungsvollsten Barrierefreiheitsprobleme: Tastaturnavigation, Screenreader-Unterstützung und Farbkontrast
- Fügen Sie semantisches HTML und ARIA-Attribute zu Legacy-UI-Komponenten hinzu, denen ordentliches Barrierefreiheits-Markup fehlt
- Implementieren Sie Fokusmanagement und Tastaturnavigation für Legacy-interaktive Elemente
- Stellen Sie sicher, dass alle Bilder, Icons und nicht-textuellen Inhalte angemessenen alternativen Text haben
- Testen Sie mit assistiven Technologien einschließlich Screenreadern, Vergrößerungssoftware und Sprachsteuerungssoftware
- Beziehen Sie Nutzer mit Behinderungen in Usability-Tests ein, um Barrierefreiheitsverbesserungen zu validieren
- Erstellen Sie einen Barrierefreiheits-Styleguide für laufende Entwicklung, um Regression zu verhindern

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Erweitert die Nutzerbasis um Menschen mit Behinderungen, die zuvor ausgeschlossen waren
- Erfüllt rechtliche und regulatorische Anforderungen an digitale Barrierefreiheit
- Verbessert die Nutzbarkeit für alle Nutzer, da barrierefreies Design oft die Gesamtnutzererfahrung verbessert
- Verringert rechtliches Risiko durch barrierefreiheitsbezogene Beschwerden und Klagen

**Kosten und Risiken:**
- Die Nachrüstung von Barrierefreiheit in Legacy-UIs, die ohne semantisches Markup gebaut wurden, kann arbeitsintensiv sein
- Legacy-Frameworks oder individuelle UI-Komponenten können fundamentale Barrierefreiheitsbeschränkungen haben
- Umfassende Barrierefreiheits-Compliance erfordert anhaltendes Testen und Wartung
- Manche Legacy-Visualdesigns benötigen möglicherweise erhebliche Überarbeitung, um Kontrast- und Layoutanforderungen zu erfüllen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein 2009 gebautes Portal für Behördendienste erhielt mehrere Beschwerden von Bürgern, die Formulare mit Screenreadern nicht ausfüllen konnten. Ein Barrierefreiheitsaudit offenbarte, dass die Legacy-Anwendung nicht-semantische HTML-Tabellen für Layout nutzte, Formularlabels fehlten und individuelle JavaScript-Steuerelemente vollständig unsichtbar für assistive Technologie waren. Das Team priorisierte die fünf meistgenutzten Formulare, ersetzte tabellenbasierte Layouts durch semantisches HTML, fügte ARIA-Labels hinzu und implementierte Tastaturnavigation. Diese Änderungen erlaubten es Screenreader-Nutzern erstmals, Formulare unabhängig auszufüllen. Die Verbesserungen kamen außerdem sehenden Nutzern zugute, indem sie einen saubereren, logischeren Formularfluss schufen, was die Gesamtabbruchrate bei Formularen um 15 % verringerte.
