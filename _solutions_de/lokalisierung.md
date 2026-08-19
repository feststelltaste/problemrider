---
title: Lokalisierung
description: Anpassung der Software an unterschiedliche Sprachen, Regionen und
  kulturelle Konventionen.
category:
- Requirements
- Code
quality_tactics_url: https://qualitytactics.de/en/usability/localization/
problems:
- poor-user-experience-ux-design
- user-confusion
- user-frustration
- competitive-disadvantage
- feature-gaps
- negative-user-feedback
- hardcoded-values
- customer-dissatisfaction
layout: solution
lang: de
en_slug: localization
related_solutions:
- slug: consistent-terminology
  similarity: 0.75
- slug: plain-language
  similarity: 0.7
- slug: consistent-user-interface
  similarity: 0.7
- slug: understandable-error-messages
  similarity: 0.7
- slug: ubiquitous-language
  similarity: 0.7
- slug: intuitive-navigation
  similarity: 0.7
---

## Description

Lokalisierung passt die Sprache, Datums- und Zahlenformate sowie kulturellen Konventionen einer Anwendung an die Region an, in der sie tatsächlich genutzt wird, statt die einzige Sprache und Locale anzunehmen, für die sie Jahre zuvor ursprünglich gebaut wurde. Legacy-Systeme, die für einen Markt gebaut wurden, stoßen routinemäßig gegen diese Wand, sobald eine Organisation international expandiert: mehrdeutig gerenderte Daten, Währungstrennzeichen, die im Ausland etwas anderes bedeuten, Fehlermeldungen, die nur in der Originalsprache Sinn ergeben — Probleme, die bestenfalls verwirrend und schlimmstenfalls kostspielig sind. Fest codierte Zeichenfolgen in Ressourcendateien zu extrahieren und locale-bewusste Formatierung einzuführen schließt diese Lücke, obwohl sich der Aufwand mit jeder unterstützten Sprache summiert, da sich Testen pro Locale vervielfacht und jedes neue Feature eine laufende Übersetzungsverpflichtung über alle davon hinweg hinzufügt.

## How to Apply ◆

> Legacy-Systeme wurden oft für eine einzige Sprache und Region gebaut. Während Organisationen expandieren, wird das Fehlen von Lokalisierung zu einer bedeutenden Hürde für die Einführung in neuen Märkten.

- Extrahieren Sie alle fest codierten Zeichenfolgen aus der Legacy-Codebasis in Ressourcendateien oder ein Lokalisierungsframework. Dies umfasst UI-Beschriftungen, Fehlermeldungen, Hilfetexte, E-Mail-Vorlagen und Berichtsköpfe.
- Implementieren Sie locale-bewusste Formatierung für Daten, Zeiten, Zahlen, Währungen und Adressen. Legacy-Systeme, die Daten als „MM/DD/YYYY" anzeigen, verwirren Nutzer in Regionen, die „DD.MM.YYYY" oder „YYYY-MM-DD" erwarten.
- Unterstützen Sie Unicode im gesamten Stack, einschließlich Datenbank, APIs und UI-Rendering. Legacy-Systeme, die mit reinen ASCII-Annahmen gebaut wurden, brechen bei der Verarbeitung von Zeichen aus nicht-lateinischen Schriften.
- Gestalten Sie UI-Layouts, um Textausdehnung zu berücksichtigen. Deutsche und französische Übersetzungen sind typischerweise dreißig bis vierzig Prozent länger als Englisch, und Rechts-nach-links-Sprachen wie Arabisch erfordern gespiegelte Layouts.
- Externalisieren Sie locale-spezifische Geschäftsregeln wie Steuerberechnungen, Adressformate und regulatorische Anforderungen, damit sie pro Region ohne Codeänderungen konfiguriert werden können.
- Etablieren Sie einen Übersetzungsworkflow mit professionellen Übersetzern, die die Domäne verstehen, statt sich auf entwicklerseitige maschinelle Übersetzung zu verlassen.

## Tradeoffs ⇄

> Lokalisierung öffnet neue Märkte und verbessert die Nutzererfahrung für nicht-englischsprachige Nutzer, fügt aber erhebliche Komplexität zu Entwicklung und Testen hinzu.

**Vorteile:**

- Ermöglicht die Expansion in neue Märkte, indem sprachliche und kulturelle Barrieren entfernt werden, die die Einführung verhindern.
- Verringert Nutzerverwirrung und Fehler durch ungewohnte Datumsformate, Zahlenkonventionen und Terminologie.
- Zeigt Respekt für die Sprachen und Kulturen der Nutzer, was Zufriedenheit und Vertrauen verbessert.
- Beseitigt fest codierte Werte in der gesamten Codebasis, was als Nebeneffekt die Wartbarkeit verbessert.

**Kosten und Risiken:**

- Das Extrahieren von Zeichenfolgen aus einer Legacy-Codebasis mit Jahren fest codierten Textes ist arbeitsintensiv und fehleranfällig, da Zeichenfolgen an unerwarteten Stellen eingebettet sein könnten.
- Das Testen der Anwendung in jeder unterstützten Locale vervielfacht den QA-Aufwand, und automatisierte Tests müssen variable String-Längen und -Formate berücksichtigen.
- Übersetzung ist eine laufende Kostenposition: Jedes neue Feature, jede Fehlermeldung und jede UI-Änderung erfordert Übersetzung in alle unterstützten Sprachen.
- Unterstützung für Rechts-nach-links-Sprachen könnte erhebliche Layout-Änderungen erfordern, die schwer in Legacy-CSS und Komponentenstrukturen nachzurüsten sind.
- Kulturelle Lokalisierung geht über Übersetzung hinaus und umfasst Überlegungen wie Symbolbedeutungen, Farbassoziationen und inhaltliche Angemessenheit, die Fachexpertise erfordern.

## How It Could Be

> Lokalisierungsfehler in Legacy-Systemen reichen von komisch bis kostspielig, und sie werden dringend, wenn die Organisation international expandiert.

Ein für den US-Markt gebautes Legacy-Buchhaltungssystem wird ohne Lokalisierung in den neuen europäischen Büros des Unternehmens deployt. Europäische Buchhalter stoßen sofort auf Probleme: Daten sind mehrdeutig, weil das System sie im MM/DD-Format ohne Kennzeichnung anzeigt, Währungsbeträge nutzen Punkte als Dezimaltrennzeichen statt der in Deutschland und Frankreich erwarteten Kommas, und alle Fehlermeldungen sind auf Englisch. Das Team unternimmt einen gestaffelten Lokalisierungsaufwand, beginnend mit Datums- und Zahlenformatierung mittels der Locale-Einstellungen des Browsers, dann Extraktion von UI-Zeichenfolgen in Ressourcen-Bundles zur Übersetzung. Nach sechs Monaten inkrementeller Arbeit unterstützt das System Englisch, Deutsch und Französisch mit locale-angemessener Formatierung. Europäische Nutzer berichten, dass das System endlich für ihre tägliche Arbeit nutzbar ist, und Dateneingabefehler durch Datumsformatverwirrung werden beseitigt.
