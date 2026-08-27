---
title: Suchfunktion
description: Bereitstellung einer leistungsfähigen Suchfunktion für Inhalte
  und Funktionen.
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/search-function/
problems:
- poor-user-experience-ux-design
- user-frustration
- user-confusion
- slow-response-times-for-lists
- cognitive-overload
- negative-user-feedback
- shadow-systems
- information-fragmentation
layout: solution
lang: de
en_slug: search-function
related_solutions:
- slug: intuitive-navigation
  similarity: 0.85
- slug: knowledge-base
  similarity: 0.75
- slug: auto-save
  similarity: 0.75
- slug: visual-hierarchy
  similarity: 0.75
- slug: undo-and-redo
  similarity: 0.7
- slug: user-centered-design
  similarity: 0.7
---

## Description

Eine leistungsfähige Suchfunktion erlaubt Nutzern, jeden Datensatz, jedes Dokument oder Feature zu finden, indem sie eintippen, woran sie sich erinnern, statt eine tiefe Menühierarchie zu navigieren oder eine lange Liste zu scrollen — etwas, das viele Legacy-Systeme überhaupt nie bieten. Ihre Abwesenheit ist es, was Nutzer dazu treibt, ihre eigenen privaten Nachschlagesysteme, auswendig gelernte Ordnerpfade oder persönliche Tabellenkalkulationen zu bauen, die Namen auf Orte abbilden, weil die vom System selbst gebotene Alternative schlicht zu langsam ist. Volltextsuche mit unscharfem Abgleich und facettierter Filterung beseitigt dieses Bedürfnis fast sofort, hängt aber von Indexierungsinfrastruktur ab, die das Datenmodell des Legacy-Systems derzeit möglicherweise nicht unterstützt, und Ergebnisse müssen dieselben Autorisierungsgrenzen respektieren, die der Rest des Systems durchsetzt, sonst wird die Suche selbst zu einer Sicherheitslücke.

## How to Apply ◆

> Legacy-Systemen fehlt Suchfunktionalität oft gänzlich, was Nutzer zwingt, durch tiefe Menühierarchien zu navigieren oder durch lange Listen zu scrollen, um zu finden, was sie brauchen. Eine leistungsfähige Suchfunktion ist eine der wirkungsvollsten Usability-Verbesserungen.

- Implementieren Sie eine globale Suchleiste, zugänglich von jedem Bildschirm, die über alle Entitätstypen sucht: Datensätze, Dokumente, Einstellungen und Features. Nutzer sollten alles im System von einem Ort aus finden können.
- Unterstützen Sie unscharfen Abgleich und Tippfehlertoleranz, sodass sich Nutzer keine exakten Namen oder Identifikatoren merken müssen. Legacy-Daten enthalten oft Inkonsistenzen, die exakter Abgleich nicht handhaben kann.
- Bieten Sie facettierte Suchergebnisse, die Befunde nach Typ gruppieren und Nutzern erlauben, nach Kategorie, Zeitraum, Status und anderen relevanten Attributen zu filtern.
- Zeigen Sie Suchergebnisse mit genug Kontext an, damit Nutzer das richtige Element identifizieren können, ohne in jedes Ergebnis zu klicken. Zeigen Sie Schlüsselattribute, einen Ausschnitt übereinstimmenden Texts und den Entitätstyp.
- Implementieren Sie Typeahead-Vorschläge, die Ergebnisse zeigen, während Nutzer tippen, was die Notwendigkeit reduziert, eine Suchanfrage abzusenden und auf Ergebnisse zu warten.
- Indizieren Sie die Suchdaten angemessen, um sicherzustellen, dass Suchantworten schnell genug für Echtzeitnutzung sind, typischerweise unter einer Sekunde.

## Tradeoffs ⇄

> Eine gut implementierte Suchfunktion verändert, wie Nutzer mit einem Legacy-System interagieren, erfordert aber Indexierungsinfrastruktur und laufende Pflege.

**Vorteile:**

- Reduziert dramatisch die Zeit, die Nutzer mit Navigation durch Menüs und Scrollen durch Listen verbringen, um spezifische Datensätze oder Features zu finden.
- Beseitigt die Notwendigkeit für Nutzer, exakte Identifikatoren, Menüorte oder Navigationspfade auswendig zu lernen.
- Reduziert die Motivation für Schattensysteme, gebaut, um Suchfähigkeiten bereitzustellen, die dem Legacy-System fehlen.
- Macht das System für gelegentliche Nutzer zugänglich, die die Navigationsstruktur nicht auswendig gelernt haben.

**Kosten und Risiken:**

- Der Bau eines umfassenden Suchindex über eine Legacy-Datenbank mit inkonsistenten Datenmodellen und Entitätsbeziehungen ist technisch herausfordernd.
- Suchperformance erfordert dedizierte Indexierungsinfrastruktur wie Elasticsearch oder Solr, die das Legacy-System derzeit möglicherweise nicht unterstützt.
- Schlechte Suchrelevanz, die zu viele irrelevante Ergebnisse zurückgibt, untergräbt Vertrauen in die Suchfunktion und treibt Nutzer zurück zu manueller Navigation.
- Suche muss Autorisierungsgrenzen respektieren, indem Nutzern nur Ergebnisse gezeigt werden, auf die sie zugreifen dürfen, was in Systemen mit rollenbasierter Zugriffskontrolle Komplexität hinzufügt.

## How It Could Be

> Die Abwesenheit von Suche in einem Legacy-System zwingt Nutzer, auswendig gelernte Navigationspfade und persönliche Nachschlagesysteme zu entwickeln, die Wissen fragmentieren und Zeit verschwenden.

Ein Legacy-Vertragsverwaltungssystem speichert über fünfzigtausend Verträge, organisiert in einer hierarchischen Ordnerstruktur, die das Organigramm des Unternehmens spiegelt. Einen spezifischen Vertrag zu finden erfordert zu wissen, zu welcher Abteilung, Division und welchem Projekt er gehört, und dann durch bis zu fünf Ordnerebenen zu navigieren. Rechtsabteilungsmitarbeiter pflegen persönliche Tabellenkalkulationen, die Vertragsnamen auf Ordnerpfade abbilden. Das Team implementiert eine Volltextsuche, die Vertragstitel, Parteinamen, Schlüsselbegriffe und Vertragstext indiziert. Suchergebnisse zeigen den Vertragstitel, die Parteien, Wirksamkeitsdaten und einen Textausschnitt mit hervorgehobenem übereinstimmendem Begriff. Innerhalb weniger Wochen nach der Bereitstellung werden die persönlichen Nachschlage-Tabellenkalkulationen aufgegeben, weil die Suche schneller und verlässlicher ist. Rechtsabteilungsmitarbeiter berichten, dass das Finden eines Vertrags nun Sekunden statt Minuten dauert, und sie entdecken Verträge, von denen sie nicht wussten, dass sie existierten, weil sie unter unerwarteten Ordnerpfaden abgelegt waren.
