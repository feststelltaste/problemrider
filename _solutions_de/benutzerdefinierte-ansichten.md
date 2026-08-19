---
title: Benutzerdefinierte Ansichten
description: Nutzern erlauben, eigene Ansichten und Layouts zu erstellen.
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/custom-views/
problems:
- poor-user-experience-ux-design
- user-frustration
- shadow-systems
- feature-gaps
- negative-user-feedback
- customer-dissatisfaction
- user-confusion
layout: solution
lang: de
en_slug: custom-views
related_solutions:
- slug: customizable-user-interface
  similarity: 0.8
- slug: search-function
  similarity: 0.7
- slug: responsive-design
  similarity: 0.7
- slug: visual-hierarchy
  similarity: 0.65
- slug: progressive-disclosure
  similarity: 0.65
- slug: intuitive-navigation
  similarity: 0.65
---

## Description

Benutzerdefinierte Ansichten erlauben jedem Nutzer, zu konfigurieren, welche Spalten, Filter und Sortierreihenfolge er für einen bestimmten Datensatz sieht, statt jede Rolle zu zwingen, mit dem einen fixen Layout zu arbeiten, das ein Legacy-System typischerweise bereitstellt. Weil eine Legacy-Tabelle üblicherweise gebaut wurde, um jedem alle verfügbaren Felder zu zeigen, enden Nutzer, die sehr unterschiedliche Aufgaben an denselben Daten erledigen, damit, in Tabellenkalkulationen zu exportieren, nur um die für ihre eigene Aufgabe relevante Teilmenge zu sehen — genau das Schattensystem-Verhalten, das gespeicherte, umschaltbare Ansichten eliminieren sollen. Der Tradeoff ist zusätzliche Komplexität in der Daten- und Rendering-Schicht, und Support wird schwieriger, da ein gemeldetes Problem sich möglicherweise nur in einer Konfiguration reproduzieren lässt, die das Support-Team nicht sehen kann.

## How to Apply ◆

> Legacy-Systeme bieten typischerweise eine einzelne fixe Ansicht für jeden Datensatz und zwingen alle Nutzer, mit demselben Layout unabhängig von ihrer Rolle oder Aufgabe zu arbeiten. Benutzerdefinierte Ansichten lassen Nutzer die Oberfläche an ihre Bedürfnisse anpassen.

- Erlauben Sie Nutzern, auszuwählen, welche Spalten in Datentabellen sichtbar sind und in welcher Reihenfolge. Legacy-Systeme zeigen oft jede verfügbare Spalte an und überwältigen Nutzer, die nur eine Teilmenge brauchen.
- Implementieren Sie gespeicherte Ansichten, die Nutzer benennen, speichern und zwischen denen sie wechseln können. Unterschiedliche Aufgaben erfordern unterschiedliche Datenperspektiven, und Nutzer sollten ihre Ansicht nicht jedes Mal neu konfigurieren müssen.
- Unterstützen Sie Filter- und Sortier-Voreinstellungen, die als Teil einer benutzerdefinierten Ansicht gespeichert werden können, sodass Nutzer schnell auf ihre häufigsten Datenteilmengen zugreifen können.
- Bieten Sie Standardansichten für gängige Rollen als Ausgangspunkte, sodass neue Nutzer ein vernünftiges Layout haben, ohne eines von Grund auf konfigurieren zu müssen.
- Erlauben Sie Administratoren, gemeinsame Ansichten für Teams oder Abteilungen zu erstellen, was den Aufwand individueller Konfiguration reduziert, während Anpassung weiterhin unterstützt wird.
- Persistieren Sie Ansichtspräferenzen serverseitig, sodass Nutzer ihre angepasste Oberfläche unabhängig davon sehen, welches Gerät oder welchen Browser sie nutzen.

## Tradeoffs ⇄

> Benutzerdefinierte Ansichten geben Nutzern Kontrolle über ihren Arbeitsbereich, fügen aber Komplexität zum Frontend und zur Datenschicht hinzu.

**Vorteile:**

- Reduziert die Entstehung von Schattensystemen, weil Nutzer, die das offizielle System an ihre Bedürfnisse anpassen können, weniger Motivation haben, Daten für eigene Analysen in Tabellenkalkulationen zu exportieren.
- Adressiert diverse Nutzerbedürfnisse, ohne das Entwicklungsteam zu zwingen, rollenspezifische Oberflächen für jeden Anwendungsfall zu bauen.
- Verbessert die Nutzerzufriedenheit, indem Nutzern Handlungsmacht über ihren Arbeitsbereich gegeben wird, statt ein Einheitslayout zu erzwingen.
- Reduziert kognitive Überlastung, indem Nutzer Informationen ausblenden können, die sie für ihre aktuelle Aufgabe nicht brauchen.

**Kosten und Risiken:**

- Die Funktionalität benutzerdefinierter Ansichten fügt dem Frontend-Code Komplexität hinzu, besonders bei Legacy-Systemen mit starren Rendering-Pipelines, die nicht für dynamische Layouts entworfen wurden.
- Die Unterstützung und Fehlersuche bei Problemen in stark angepassten Ansichten ist schwieriger, weil das Entwicklungsteam die exakte Konfiguration, die ein Nutzer sieht, nicht reproduzieren kann.
- Nutzer könnten Ansichten erstellen, die wichtige Informationen auslassen, und dann kritische Daten verpassen, was Schutzmaßnahmen wie Pflichtspalten für bestimmte Rollen erfordert.
- Das Persistieren der Ansichtskonfiguration erfordert zusätzlichen Datenbankspeicher und API-Endpunkte, die das Legacy-System möglicherweise nicht hat.

## How It Could Be

> Einheitliche Oberflächen in Legacy-Systemen treiben Nutzer dazu, Schattensysteme zu erstellen, in denen sie die Daten so sehen können, wie sie sie brauchen.

Ein Legacy-Bestandsverwaltungssystem zeigt eine Tabelle mit zweiunddreißig Spalten für jedes Produkt an, von SKU und Beschreibung bis Lagerort, Lieferantencodes, Zollklassifikationen und Nachbestellpunkten. Lagerpersonal braucht nur fünf dieser Spalten für die tägliche Arbeit, während Beschaffungspersonal einen anderen Satz von zwölf Spalten braucht. Beide Gruppen haben Daten nach Excel exportiert, um ihre eigenen Ansichten zu erstellen, was zu dupliziertem Aufwand und veralteten Daten führt. Das Team fügt der Produkttabelle Spaltenauswahl und gespeicherte-Ansicht-Funktionalität hinzu. Lagerpersonal erstellt eine „Kommissionierliste"-Ansicht mit nur den Spalten, die es braucht, und Beschaffungspersonal erstellt eine „Nachbestellungsprüfung"-Ansicht. Tabellenkalkulationsexporte gehen drastisch zurück, und beide Gruppen berichten, weniger Zeit mit dem Finden von Informationen und mehr Zeit mit dem Handeln darauf zu verbringen.
