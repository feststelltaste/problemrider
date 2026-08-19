---
title: Intuitive Navigation
description: Umsetzung einer logischen und leicht verständlichen Navigationsstruktur.
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/intuitive-navigation/
problems:
- poor-user-experience-ux-design
- user-confusion
- user-frustration
- cognitive-overload
- increased-cognitive-load
- negative-user-feedback
- shadow-systems
- difficult-developer-onboarding
layout: solution
lang: de
en_slug: intuitive-navigation
related_solutions:
- slug: search-function
  similarity: 0.85
- slug: cognitive-load-minimization
  similarity: 0.85
- slug: consistent-user-interface
  similarity: 0.8
- slug: user-centered-design
  similarity: 0.8
- slug: visual-hierarchy
  similarity: 0.8
- slug: adaptive-behavior
  similarity: 0.8
---

## Description

Intuitive Navigation strukturiert die Menühierarchie eines Systems um, wie Nutzer tatsächlich über ihre Aufgaben denken, statt um die Modul- oder Datenbankstruktur, die ein Legacy-Navigationsmenü typischerweise nach Jahren organischen, ungeplanten Wachstums widerspiegelt. Dieses organische Wachstum ist der Grund, warum Legacy-Systeme so oft mit Dutzenden Top-Level-Menüpunkten enden, beschriftet mit internen Codes wie „SYS_CONFIG", die der Person, die sie nutzt, nichts bedeuten, was Nutzer zwingt, eigene Spickzettel zu bauen, nur um sich zu merken, wo Dinge sind. Die Hierarchie aus Card-Sorting-Übungen mit echten Nutzern neu aufzubauen, die oberste Ebene auf eine Handvoll aufgabenorientierter Kategorien zu begrenzen und eine globale Suche als Notausgang für Nutzer hinzuzufügen, die bereits wissen, was sie wollen, schließt diese Lücke — auf Kosten der Störung des Muskelgedächtnisses genau der erfahrenen Nutzer, die das alte, verwirrende Layout trainiert hatte.

## How to Apply ◆

> Legacy-Systeme haben oft Navigationsstrukturen, die sich über Jahre organisch entwickelt haben und die technische Architektur des Systems statt Nutzeraufgaben widerspiegeln. Die Umstrukturierung der Navigation um Nutzerziele macht das System auffindbar und effizient.

- Führen Sie Card-Sorting-Übungen mit repräsentativen Nutzern durch, um zu verstehen, wie sie die Funktionalität des Systems mental organisieren. Nutzen Sie die Ergebnisse, um eine Navigationshierarchie zu entwerfen, die zu Nutzermentalmodellen statt zum Datenbankschema oder zur Modulstruktur passt.
- Begrenzen Sie die primäre Navigation auf sieben oder weniger Top-Level-Punkte. Legacy-Systeme mit Dutzenden Menüpunkten überwältigen Nutzer. Gruppieren Sie zusammengehörige Punkte in logische Kategorien und nutzen Sie sekundäre Navigation für weniger häufig genutzte Features.
- Implementieren Sie Breadcrumbs, um Nutzern zu zeigen, wo sie sich in der Systemhierarchie befinden, und ihnen zu erlauben, zurückzunavigieren, ohne den Zurück-Button des Browsers zu nutzen, der in Legacy-Anwendungen oft nicht funktioniert.
- Fügen Sie eine globale Suche oder Command Palette hinzu, die Nutzern erlaubt, direkt zu jedem Bildschirm oder jeder Funktion zu springen, indem sie ihren Namen eingeben, und die Navigationshierarchie für Nutzer, die wissen, wonach sie suchen, vollständig zu umgehen.
- Stellen Sie sicher, dass Navigationsbeschriftungen nutzerorientierte Sprache statt technischer oder interner Terminologie nutzen. Ersetzen Sie Beschriftungen wie „SYS_CONFIG" oder „Modul 4" durch beschreibende Namen wie „Systemeinstellungen" oder „Berichtswesen".
- Machen Sie die Navigation über alle Bereiche der Anwendung hinweg konsistent, damit Nutzer vorhersagen können, wo sie Funktionalität finden, unabhängig davon, welches Modul sie gerade nutzen.

## Tradeoffs ⇄

> Die Umstrukturierung der Navigation verbessert Auffindbarkeit und Effizienz, stört aber das Muskelgedächtnis erfahrener Nutzer.

**Vorteile:**

- Verringert die Zeit, die Nutzer mit der Suche nach Funktionalität verbringen, was Produktivität direkt verbessert und Frustration verringert.
- Macht das System für neue und gelegentliche Nutzer zugänglich, die sich nicht auf gespeicherte Navigationspfade verlassen können.
- Beseitigt die Notwendigkeit, dass Nutzer persönliche Notizen oder Lesezeichen pflegen, die dokumentieren, wo bestimmte Features in der Navigation versteckt sind.
- Verringert kognitive Überlastung, indem eine klare, organisierte Struktur statt einer flachen Liste von Dutzenden Menüpunkten präsentiert wird.

**Kosten und Risiken:**

- Erfahrene Nutzer, die die aktuelle Navigation auswendig gelernt haben, brauchen Zeit zur Anpassung, und manche könnten die Änderung anfänglich ablehnen.
- Die Umstrukturierung der Navigation in einem Legacy-System kann Änderungen am URL-Routing, an Autorisierungsprüfungen und an Seitenverlinkung erfordern, die mit der bestehenden Struktur verflochten sind.
- Navigationsänderungen müssen Nutzern klar kommuniziert werden, durch Release Notes, Training und möglicherweise einen vorübergehenden „Wo ist das hin?"-Leitfaden.
- Das Testen aller Navigationspfade nach der Umstrukturierung ist essenziell, um sicherzustellen, dass keine Funktionalität unerreichbar wird.

## How It Could Be

> Nutzer von Legacy-Systemen entwickeln oft aufwendige persönliche Systeme, um sich zu merken, wo Dinge sind — ein klares Zeichen, dass die Navigation versagt hat.

Ein Legacy-Stadtverwaltungssystem hat ein Hauptmenü mit achtundzwanzig Top-Level-Punkten, alphabetisch nach internem Modulnamen organisiert: „ACCTS_RCV", „BLD_PRMT", „CODE_ENF" und so weiter. Stadtangestellte, die mehrere Module nutzen, pflegen gedruckte Spickzettel, die lesbare Namen auf Menüabkürzungen abbilden. Das Team reorganisiert die Navigation in sechs aufgabenorientierte Kategorien: „Finanzen", „Genehmigungen und Lizenzen", „Bauaufsicht", „Öffentliche Arbeiten", „Personalwesen" und „Verwaltung". Jede Kategorie klappt auf, um ihre Unterfunktionen mit klaren, lesbaren Beschriftungen zu zeigen. Sie fügen auch eine Suchleiste hinzu, die sowohl die alten Modulcodes als auch die neuen Beschriftungen akzeptiert, um den Übergang zu erleichtern. Innerhalb eines Monats werfen Angestellte ihre Spickzettel weg, und neue Angestellte berichten, dass das System deutlich leichter zu erlernen ist, als Kollegen sie gewarnt hatten.
