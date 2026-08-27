---
title: Responsive Design
description: Automatische Anpassung der Nutzeroberfläche an verschiedene
  Bildschirmgrößen und Geräte.
category:
- Requirements
- Architecture
quality_tactics_url: https://qualitytactics.de/en/usability/responsive-design/
problems:
- poor-user-experience-ux-design
- user-frustration
- competitive-disadvantage
- negative-user-feedback
- feature-gaps
- high-client-side-resource-consumption
- customer-dissatisfaction
layout: solution
lang: de
en_slug: responsive-design
related_solutions:
- slug: mobile-first-design
  similarity: 0.85
- slug: visual-hierarchy
  similarity: 0.75
- slug: user-centered-design
  similarity: 0.7
- slug: intuitive-navigation
  similarity: 0.7
- slug: cognitive-load-minimization
  similarity: 0.7
- slug: assistive-technology-support
  similarity: 0.7
---

## Description

Responsive Design lässt eine einzelne Oberfläche ihr Layout automatisch an jede Bildschirmgröße anpassen, auf der sie betrachtet wird, und ersetzt die Festbreiten-, Festpixel-Layouts, für die die meisten Legacy-Systeme gebaut wurden, als Desktop-Monitore das einzige echte Ziel waren. Diese feste Annahme ist der Grund, warum Legacy-Oberflächen so oft auf einem Tablet oder Telefon regelrecht brechen — tabellenbasierte Layouts, die nicht neu fließen, für einen Mauszeiger dimensionierte Schaltflächen, die mit einem Finger nahezu untippbar sind —, obwohl die zugrunde liegende Funktionalität auf einem kleineren Gerät perfekt nutzbar wäre, wenn sich das Layout anpassen könnte. Die Umstellung auf fließende, CSS-basierte Layouts mit definierten Breakpoints schließt diese Lücke, ohne eine separate mobile Anwendung zu erfordern, obwohl der nachträgliche Einbau in dichte Legacy-Datenraster und Multi-Panel-Bildschirme echte, substanzielle Frontend-Arbeit ist, keine einfache Stilanpassung.

## How to Apply ◆

> Legacy-Systeme, gebaut für Desktop-Monitore mit fester Auflösung, brechen auf Tablets, Telefonen und modernen hochauflösenden Displays. Responsive Design stellt sicher, dass sich die Oberfläche an jede Bildschirmgröße anpasst.

- Ersetzen Sie Festbreiten-Layouts durch fließende Raster, die relative Einheiten wie Prozentwerte und Viewport-Einheiten statt fest codierter Pixelbreiten nutzen. Legacy-tabellenbasierte Layouts sind besonders problematisch und sollten in CSS-basierte Layouts umgewandelt werden.
- Definieren Sie responsive Breakpoints für die von den Nutzern der Anwendung am häufigsten genutzten Bildschirmgrößen. Testen Sie die Anwendung bei jedem Breakpoint, um sicherzustellen, dass sich Layouts elegant anpassen.
- Passen Sie Datentabellen für kleinere Bildschirme mittels Techniken wie horizontalem Scrollen, Spaltenpriorisierung, die weniger wichtige Spalten auf kleinen Bildschirmen verbirgt, oder kartenbasierten Layouts an, die Tabellenzeilen vertikal stapeln.
- Machen Sie Touch-Ziele groß genug für mobile Interaktion. Legacy-Schaltflächen und -Links, entworfen für Mauszeiger, sind oft zu klein und zu eng beieinander für Fingertipp.
- Verwenden Sie responsive Bilder, die angemessene Größen für verschiedene Bildschirmdichten und Viewport-Größen laden, was den Bandbreitenverbrauch bei mobilen Verbindungen reduziert.
- Testen Sie auf tatsächlichen Geräten zusätzlich zu Browser-Emulatoren. Browser-Emulatoren übersehen Touch-Verhalten, Performance-Eigenschaften und Rendering-Unterschiede auf echter mobiler Hardware.

## Tradeoffs ⇄

> Responsive Design macht die Anwendung über alle Geräte hinweg nutzbar, aber der nachträgliche Einbau in ein Legacy-System mit Festbreiten-Layout erfordert erhebliche Frontend-Arbeit.

**Vorteile:**

- Ermöglicht Nutzern den Zugriff auf das System von jedem Gerät, einschließlich Tablets und Telefonen, die zunehmend im Außendienst, in Meetings und für schnelle Nachschlagevorgänge genutzt werden.
- Beseitigt die Notwendigkeit, separate mobile Anwendungen für grundlegenden Systemzugang zu bauen und zu pflegen.
- Verbessert die Desktop-Erfahrung auf modernen hochauflösenden und Ultra-Wide-Monitoren durch effektive Nutzung verfügbaren Bildschirmplatzes.
- Schließt Feature-Lücken im Vergleich zu Wettbewerbern, die responsive oder mobile Erfahrungen anbieten.

**Kosten und Risiken:**

- Die Umwandlung eines Legacy-Festbreiten-Layouts zu Responsive Design kann die Überarbeitung jeder Seitenvorlage und Komponente im System erfordern.
- Komplexe Legacy-Oberflächen mit dichten Datenrastern und Multi-Panel-Layouts sind besonders schwer responsiv zu machen, ohne Funktionalität zu verlieren.
- Responsive Design erhöht CSS- und Testkomplexität, weil jeder Bildschirm bei mehreren Größen korrekt aussehen muss.
- Legacy-JavaScript, das sich auf feste Pixelpositionen für Dropdown-Menüs, Popups und Drag-and-Drop verlässt, könnte brechen, wenn das Layout neu fließt.

## How It Could Be

> Nutzer erwarten zunehmend, auf Arbeitssysteme von mehreren Geräten aus zuzugreifen, und Legacy-Systeme, die sich nicht anpassen, werden als veraltet und frustrierend wahrgenommen.

Ein Legacy-Außendienst-Management-System verlangt von Technikern, firmeneigene Laptops zu nutzen, um ihre Zeitpläne zu prüfen und auf Arbeitsaufträge zuzugreifen. Techniker finden die Laptops unhandlich zu tragen und langsam zu starten im Feld, sodass sie häufig das Büro anrufen, um Disponenten zu bitten, ihnen Arbeitsauftragsdetails vorzulesen. Das Team implementiert Responsive Design für die Terminplanungs- und Arbeitsauftragsbildschirme, unter Nutzung eines Spaltenprioritätsmusters für die Arbeitsauftragstabelle, das nur essenzielle Spalten auf Telefonbildschirmen zeigt und alle Spalten auf dem Desktop. Die Navigation wechselt auf kleinen Bildschirmen von einer horizontalen Symbolleiste zu einem Hamburger-Menü, und Touch-Ziele werden für mobile Nutzung vergrößert. Techniker beginnen, ihre persönlichen Telefone zu nutzen, um Zeitpläne zu prüfen und Arbeitsaufträge im Feld zu überprüfen, was Anrufe an das Dispositionsbüro reduziert und Reaktionszeiten verbessert. Die responsive Implementierung kostet einen Bruchteil dessen, was eine native mobile Anwendung erfordert hätte.
