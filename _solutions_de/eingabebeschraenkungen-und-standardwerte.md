---
title: Eingabebeschränkungen und Standardwerte
description: Einschränkung der Eingabe durch Dropdowns, Datumsauswahl, Schieberegler
  und sinnvolle Standardwerte.
category:
- Requirements
- Code
quality_tactics_url: https://qualitytactics.de/en/usability/input-constraints-and-defaults/
problems:
- increased-error-rates
- poor-user-experience-ux-design
- user-frustration
- user-confusion
- inadequate-error-handling
- increased-customer-support-load
- silent-data-corruption
layout: solution
lang: de
en_slug: input-constraints-and-defaults
related_solutions:
- slug: real-time-input-validation
  similarity: 0.7
- slug: value-range-definition
  similarity: 0.7
- slug: form-design
  similarity: 0.7
- slug: input-validation
  similarity: 0.65
- slug: understandable-error-messages
  similarity: 0.65
- slug: plain-language
  similarity: 0.65
---

## Description

Eingabebeschränkungen ersetzen ein Freitextfeld durch ein Steuerelement — ein Dropdown, eine Datumsauswahl, einen numerischen Stepper —, das es physisch unmöglich macht, bestimmte Klassen ungültiger Daten einzugeben, statt schlechte Eingabe erst zu validieren, nachdem sie getippt und abgesendet wurde. Legacy-Systeme nutzen häufig Freitext für Daten, die tatsächlich eine feste Menge gültiger Werte haben, was über Jahre Dutzende beinahe-duplizierter Varianten desselben Konzepts produziert, die Reporting und Aggregation unzuverlässig machen. Das Feld durch ein eingeschränktes Steuerelement mit einem sinnvollen Standardwert zu ersetzen, basierend darauf, was die meisten Nutzer tatsächlich auswählen, verhindert diese Drift künftig, obwohl die Legacy-Datenbank meist eine einmalige Bereinigung der bestehenden schmutzigen Daten braucht, bevor die neue Beschränkung konsistent angewendet werden kann.

## How to Apply ◆

> Legacy-Systeme nutzen häufig Freitextfelder für Daten, die eine eingeschränkte Menge gültiger Werte haben, was zu Datenqualitätsproblemen und Nutzerfehlern führt. Eingabebeschränkungen leiten Nutzer zu gültigen Einträgen.

- Ersetzen Sie Freitextfelder wo möglich durch angemessene eingeschränkte Steuerelemente: Dropdowns für aufzählbare Werte, Datumsauswahl für Daten, numerische Stepper für Mengen und Radiobuttons für sich gegenseitig ausschließende Wahlmöglichkeiten.
- Setzen Sie sinnvolle Standardwerte basierend auf der häufigsten Auswahl. Wenn achtzig Prozent der Nutzer dieselbe Option wählen, verringert deren Vorauswahl unnötige Entscheidungen.
- Implementieren Sie Eingabemasken für Felder mit bekannten Formaten wie Telefonnummern, Postleitzahlen und Kontonummern. Zeigen Sie das erwartete Format als Platzhalter oder Hinweis.
- Nutzen Sie Min/Max-Beschränkungen bei numerischen Feldern und Zeichenbegrenzungen bei Textfeldern, um offensichtlich ungültige Einträge von vornherein zu verhindern.
- Deaktivieren oder verbergen Sie Optionen, die im aktuellen Kontext nicht gültig sind. Wenn ein Datumsbereichswähler beispielsweise keine Enddaten vor Startdaten erlaubt, sollte die Datumsauswahl diese Beschränkung durchsetzen, statt sich auf Validierung nach der Absendung zu verlassen.
- Befüllen Sie abhängige Felder automatisch, wo möglich. Die Auswahl eines Landes sollte den Ländercode automatisch befüllen, und die Auswahl eines Produkts sollte den Stückpreis automatisch ausfüllen.

## Tradeoffs ⇄

> Eingabebeschränkungen verhindern Fehler am Eingabepunkt, können aber Nutzer frustrieren, wenn die Beschränkungen zu starr oder die Standardwerte falsch sind.

**Vorteile:**

- Verringert Dateneingabefehler dramatisch, indem es physisch unmöglich gemacht wird, bestimmte Klassen ungültiger Daten einzugeben.
- Verbessert die Datenqualität in der gesamten Legacy-Datenbank, indem die Anhäufung fehlerhafter Einträge verhindert wird.
- Verringert die Last auf die Backend-Validierung, indem ungültige Eingabe vor der Absendung abgefangen wird.
- Verringert Support-Tickets im Zusammenhang mit Verwirrung bei der Dateneingabe und Validierungsfehlern.

**Kosten und Risiken:**

- Übermäßig restriktive Beschränkungen können legitime Randfälle blockieren, die das ursprüngliche Freitextfeld handhabte, was verlangt, dass das Team die volle Bandbreite gültiger Eingaben versteht.
- Für bestimmte Nutzergruppen falsche Standardwerte können zu mehr Fehlern führen, wenn Nutzer den Standardwert akzeptieren, ohne zu prüfen, was stille Datenbeschädigung verursachen kann.
- Die Migration von Freitextfeldern zu eingeschränkten Steuerelementen in einer Legacy-Datenbank kann erfordern, bestehende schmutzige Daten zuerst zu bereinigen.
- Benutzerdefinierte Eingabesteuerelemente wie Datumsauswahl und Autovervollständigungsfelder müssen für Tastatur- und Screenreader-Nutzer zugänglich sein, was Implementierungskomplexität hinzufügt.

## How It Could Be

> Freitextfelder in Legacy-Systemen sind oft die Quelle anhaltender Datenqualitätsprobleme, die durch nachgelagerte Prozesse propagieren.

Ein Legacy-Gesundheitsterminierungssystem nutzt ein Freitextfeld für den Termintyp, was Hunderte von Varianten desselben Konzepts ergibt: „Follow-up", „follow up", „F/U", „followup", „follow-up visit" und Dutzende mehr. Reporting und Analytik basierend auf dem Termintyp sind unzuverlässig, weil die Daten nicht konsistent aggregiert werden können. Das Team ersetzt das Freitextfeld durch ein durchsuchbares Dropdown, befüllt aus einer standardisierten Liste von Termintypen. Sie führen außerdem eine einmalige Datenbereinigung durch, um die bestehenden Freitexteinträge auf die standardisierten Werte abzubilden. Innerhalb von drei Monaten sind die Termintyp-Daten zum ersten Mal konsistent genug, um zuverlässige Berichte zu erzeugen, und das Terminierungspersonal berichtet, dass das Dropdown schneller ist als Tippen, weil sie mit zwei oder drei Tastenanschlägen auswählen können.
