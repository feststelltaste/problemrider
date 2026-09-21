---
title: Progressive Disclosure
description: Schrittweises Offenlegen von Informationen und Funktionen,
  sobald Nutzer sie benötigen.
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/progressive-disclosure/
problems:
- cognitive-overload
- increased-cognitive-load
- poor-user-experience-ux-design
- user-confusion
- user-frustration
- feature-bloat
- negative-user-feedback
- difficult-developer-onboarding
layout: solution
lang: de
en_slug: progressive-disclosure
related_solutions:
- slug: cognitive-load-minimization
  similarity: 0.8
- slug: intuitive-navigation
  similarity: 0.75
- slug: visual-hierarchy
  similarity: 0.75
- slug: form-design
  similarity: 0.75
- slug: adaptive-behavior
  similarity: 0.7
- slug: search-function
  similarity: 0.7
---

## Description

Progressive Disclosure zeigt standardmäßig nur die Kernaktionen, die die meisten Nutzer brauchen, und verschiebt fortgeschrittene oder selten genutzte Optionen hinter einen ausklappbaren Abschnitt oder einen „Erweitert"-Schalter, statt jedes mögliche Feld und jede Einstellung auf einmal offenzulegen. Legacy-Oberflächen neigen dazu, sich über die Zeit in die entgegengesetzte Richtung zu entwickeln — jedes über die Jahre hinzugefügte Feature endet permanent sichtbar, auf Augenhöhe mit der Handvoll Aktionen, die für die tägliche Nutzung tatsächlich zählen, was genau die Nutzer überwältigt, denen Progressive Disclosure helfen soll. Das Risiko in die andere Richtung ist ebenfalls real: Ein Feature zu aggressiv zu verbergen kann erfahrene Power-User glauben lassen, es existiere nicht mehr, sodass die Grenze zwischen dem standardmäßig Angezeigten und dem Weggesteckten aus tatsächlichen Nutzungsdaten gezogen werden muss, nicht aus Annahme.

## How to Apply ◆

> Legacy-Systeme neigen dazu, alle Funktionalität gleichzeitig offenzulegen, was Nutzer mit Optionen überwältigt, die sie selten brauchen. Progressive Disclosure zeigt essenzielle Features zuerst und legt Komplexität nur bei Bedarf offen.

- Identifizieren Sie die Kernaktionen, die achtzig Prozent der Nutzer achtzig Prozent der Zeit ausführen. Machen Sie diese prominent sichtbar, während Sie fortgeschrittene oder selten genutzte Features hinter ausklappbare Abschnitte, „Erweitert"-Links oder Sekundärmenüs verschieben.
- Verwenden Sie ausklapp- und einklappbare Abschnitte für detaillierte Informationen. Zeigen Sie standardmäßig zusammenfassende Daten und lassen Sie Nutzer bei Bedarf zur vollständigen Detailansicht ausklappen.
- Implementieren Sie kontextuelle Menüs, die relevante Aktionen basierend auf dem aktuellen Zustand der Daten oder dem Arbeitsablaufschritt zeigen, statt jederzeit jede mögliche Aktion anzuzeigen.
- Schichten Sie Formularkomplexität: Zeigen Sie standardmäßig grundlegende Felder und bieten Sie einen „Erweiterte Optionen"-Schalter für Felder, die nur erfahrene Nutzer oder ungewöhnliche Szenarien benötigen.
- Verwenden Sie Drill-Down-Navigation für hierarchische Daten statt alle Ebenen gleichzeitig zu zeigen. Lassen Sie Nutzer in ihrem eigenen Tempo von der Zusammenfassung zum Detail navigieren.
- Wenden Sie Progressive Disclosure auf Einstellungs- und Konfigurationsseiten an, wo Legacy-Systeme oft Hunderte von Optionen auf einem einzigen Bildschirm präsentieren.

## Tradeoffs ⇄

> Progressive Disclosure vereinfacht die Oberfläche für die meisten Nutzer, kann aber Power-User frustrieren, die sofortigen Zugang zu fortgeschrittenen Features wollen.

**Vorteile:**

- Reduziert kognitive Überlastung dramatisch, indem nur die für die unmittelbare Aufgabe des Nutzers relevanten Informationen und Aktionen präsentiert werden.
- Macht das System für neue Nutzer zugänglicher, die für fortgeschrittene Features noch nicht bereit sind.
- Reduziert die visuelle Unordnung, die Legacy-Oberflächen überwältigend und veraltet wirken lässt.
- Erlaubt dem System, sowohl einfache als auch komplexe Anwendungsfälle zu unterstützen, ohne separate Oberflächen für verschiedene Nutzerebenen zu erfordern.

**Kosten und Risiken:**

- Power-User, die häufig fortgeschrittene Features nutzen, könnten finden, dass Progressive Disclosure sie verlangsamt, wenn sie sich durch zusätzliche Schritte klicken müssen, um die benötigte Funktionalität zu erreichen.
- Features zu aggressiv zu verbergen kann sie unauffindbar machen und Nutzer glauben lassen, Funktionalität fehle, wenn sie nur versteckt ist.
- Die Implementierung von Progressive Disclosure in Legacy-Frontends mit starren Layouts kann erhebliche Umstrukturierung von Seitenvorlagen und -komponenten erfordern.
- Die Grenze zwischen essenziellen und fortgeschrittenen Features variiert nach Nutzerrolle, was möglicherweise rollenbasierte Progressive-Disclosure-Konfigurationen erfordert.

## How It Could Be

> Legacy-Systeme, die im Laufe der Jahre Feature für Feature gewachsen sind, präsentieren oft jedes Feature als gleich wichtig, was eine Oberfläche schafft, die niemandem gut dient.

Ein Legacy-Lagerverwaltungssystem hat einen Produktbearbeitungsbildschirm mit zweiundvierzig Feldern, einschließlich grundlegender Informationen wie Name und SKU, Bestandsparameter, Lieferantendetails, Zollklassifizierungscodes und Anweisungen zur Handhabung von Gefahrgut. Lagermitarbeiter, die Bestandszahlen aktualisieren müssen, müssen durch alle zweiundvierzig Felder scrollen, um den Bestandsabschnitt zu finden. Das Team strukturiert den Bildschirm zu einem Tab-Layout um, wobei „Grundinfo" standardmäßig gezeigt wird und die Tabs „Bestand", „Lieferant", „Compliance" und „Erweitert" für Nutzer verfügbar sind, die sie brauchen. Jeder Tab enthält nur die relevanten Felder. Der Grundinfo-Tab deckt die Bedürfnisse von achtzig Prozent der täglichen Bearbeitungsaufgaben ab. Lagermitarbeiter berichten, dass der Bearbeitungsbildschirm nicht mehr einschüchternd ist, und neue Mitarbeiter können bereits am ersten Tag grundlegende Aktualisierungen vornehmen, statt eine Woche Schulung zu benötigen, um das vollständige Formular zu verstehen.
