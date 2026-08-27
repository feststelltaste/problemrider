---
title: Visuelle Hierarchie
description: Hervorhebung wichtiger Elemente in der Nutzeroberfläche und
  Schaffung einer klaren visuellen Struktur.
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/visual-hierarchy/
problems:
- poor-user-experience-ux-design
- user-confusion
- cognitive-overload
- increased-cognitive-load
- user-frustration
- negative-user-feedback
- increased-error-rates
layout: solution
lang: de
en_slug: visual-hierarchy
related_solutions:
- slug: intuitive-navigation
  similarity: 0.8
- slug: cognitive-load-minimization
  similarity: 0.8
- slug: progressive-disclosure
  similarity: 0.75
- slug: style-guide
  similarity: 0.75
- slug: form-design
  similarity: 0.75
- slug: plain-language
  similarity: 0.75
---

## Description

Visuelle Hierarchie nutzt Größe, Gewicht, Farbe und Abstand, um zu signalisieren, welche Elemente auf einem Bildschirm am wichtigsten sind, sodass ein Nutzer ein Layout auf einen Blick scannen und verstehen kann, statt jedes Feld mit gleichem Aufwand lesen zu müssen. Legacy-Schnittstellen präsentieren häufig alles mit identischem visuellem Gewicht — dieselbe Schriftgröße für den Status eines Falls wie für seinen internen Verfolgungscode —, was Wichtigkeit effektiv vollständig verbirgt, statt nur zu versäumen, sie hervorzuheben, und Nutzer zwingt, sorgfältig durch irrelevante Felder zu lesen, nur um das eine zu finden, das gerade tatsächlich zählt. Eine bewusste Hierarchie wiederherzustellen, mit visuell dominanten primären Aktionen und weniger betonten statt entfernten sekundären Informationen, erfordert echtes Design-Urteilsvermögen, das legacy-fokussierte Entwicklungsteams nicht immer zur Hand haben, und sie muss konsistent über das System hinweg angewendet werden, sonst wird die Inkonsistenz selbst zu einer neuen Quelle derselben Desorientierung, die sie beheben sollte.

## How to Apply ◆

> Legacy-Systeme präsentieren oft alle Informationen mit gleichem visuellem Gewicht, was es Nutzern erschwert zu identifizieren, was wichtig ist. Visuelle Hierarchie nutzt Größe, Farbe, Kontrast und Abstand, um Aufmerksamkeit zu lenken.

- Etablieren Sie eine klare Überschriftenhierarchie mit distinkten Schriftgrößen und -gewichten für Seitentitel, Abschnittsüberschriften und Unterabschnitte. Legacy-Systeme nutzen oft dieselbe Schriftgröße für alles, was Struktur unsichtbar macht.
- Nutzen Sie Größe und Prominenz, um Wichtigkeit anzuzeigen. Primäre Aktionen wie "Speichern" oder "Absenden" sollten visuell dominant sein, während sekundäre Aktionen wie "Abbrechen" und tertiäre Aktionen wie "Erweiterte Einstellungen" zunehmend weniger prominent sein sollten.
- Wenden Sie Weißraum strategisch an, um logische Inhaltsgruppen zu trennen. Legacy-Schnittstellen, die Elemente eng zusammenpacken, erzeugen visuelles Rauschen, das Scannen erschwert.
- Nutzen Sie Farbe zweckmäßig und konsistent: eine begrenzte Palette, in der jede Farbe Bedeutung trägt, wie Rot für Fehler, Grün für Erfolg und Blau für interaktive Elemente. Vermeiden Sie dekorative Farbe, die visuelles Rauschen hinzufügt, ohne Information zu vermitteln.
- Unterscheiden Sie zwischen Daten und Beschriftungen. Feldbeschriftungen sollten visuell distinkt von den Daten sein, die sie beschreiben, und Pflichtfelder sollten visuell von optionalen unterscheidbar sein.
- Betonen Sie sekundäre Informationen weniger durch kleinere Schriftgrößen, hellere Farben oder einklappbare Abschnitte, statt sie vollständig zu entfernen.

## Tradeoffs ⇄

> Visuelle Hierarchie macht Schnittstellen scanbar und intuitiv, erfordert aber Design-Expertise und konsistente Anwendung über das System hinweg.

**Vorteile:**

- Ermöglicht Nutzern, Bildschirme schnell zu scannen und die benötigten Informationen zu finden, ohne jedes Element zu lesen, was die Produktivität erheblich verbessert.
- Reduziert Fehler, indem primäre Aktionen visuell dominant und sekundäre oder gefährliche Aktionen weniger prominent gemacht werden.
- Lässt das System moderner und professioneller wirken, was das Nutzervertrauen und die Zufriedenheit verbessert.
- Reduziert kognitive Überlastung, indem Informationen so organisiert werden, dass das Auge sie natürlich verarbeiten kann.

**Kosten und Risiken:**

- Die Etablierung visueller Hierarchie erfordert Design-Expertise, die Legacy-Entwicklungsteams fehlen könnte, da sie das Verständnis von Typografie, Farbtheorie und Layout-Prinzipien beinhaltet.
- Die Nachrüstung visueller Hierarchie in Legacy-CSS, das mit tabellenbasierten Layouts oder Inline-Stilen gebaut wurde, kann erhebliches Refactoring erfordern.
- Inkonsistente Anwendung visueller Hierarchie über verschiedene Teile des Legacy-Systems hinweg kann eine unstimmige Erfahrung schaffen, während Nutzer zwischen Modulen wechseln.
- Kulturelle und Barrierefreiheitsüberlegungen beeinflussen Farbbedeutung und Kontrastanforderungen, was Komplexität zu visuellen Designentscheidungen hinzufügt.

## How It Could Be

> Legacy-Systeme, die alle Inhalte als gleich wichtig behandeln, lassen am Ende nichts wichtig erscheinen und überwältigen Nutzer mit visuellem Rauschen.

Ein Legacy-Fallverwaltungssystem zeigt Falldetails auf einem einzigen Bildschirm mit achtundzwanzig Feldern an, angeordnet in einem Raster. Alle Felder nutzen dieselbe Schriftgröße, denselben Beschriftungsstil und denselben Abstand. Sachbearbeiter, die den Bildschirm scannen, um den aktuellen Status eines Falls zu finden, müssen durch Felder wie Erstellungsdatum, internen Fallcode, zugewiesenes Büro und mehrere selten benötigte administrative Felder lesen, bevor sie den Status finden, der genauso aussieht wie alles andere. Das Team gestaltet den Bildschirm mit einer visuellen Hierarchie neu: Ein prominenter Kopfbereich zeigt Falltitel, Status und Priorität mit großem, fettem Text und statusspezifischer Farbkodierung. Kontaktinformationen und Schlüsseldaten erscheinen in einem sekundären Abschnitt darunter. Administrative Felder werden in ein aufklappbares Panel eingeklappt. Sachbearbeiter berichten, dass sie nun einen Blick auf einen Fall werfen und sofort dessen Status und Priorität verstehen können, was zuvor sorgfältiges Lesen erforderte. Die durchschnittliche Zeit zur Triage eines neuen Falls sinkt, weil die wichtigste Information die sichtbarste ist.
