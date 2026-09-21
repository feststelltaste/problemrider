---
title: Plausibilitätsprüfungen
description: Überprüfung von Eingaben, Daten oder Zuständen auf Gültigkeit,
  um potenzielle Fehler frühzeitig zu erkennen.
category:
- Code
- Architecture
problems:
- silent-data-corruption
- inadequate-error-handling
- unpredictable-system-behavior
- data-migration-integrity-issues
- increased-error-rates
- brittle-codebase
- integer-overflow-underflow
- master-data-ownership-gaps
layout: solution
lang: de
en_slug: plausibility-checks
related_solutions:
- slug: checksums
  similarity: 0.7
- slug: value-range-definition
  similarity: 0.7
- slug: data-integrity
  similarity: 0.7
- slug: data-quality-checks
  similarity: 0.7
- slug: redundant-checksums
  similarity: 0.7
- slug: continuous-data-verification
  similarity: 0.65
---

## Description

Plausibilitätsprüfungen sind leichtgewichtige Validierungsregeln — Bereichsprüfungen, Formatvalidierung, feldübergreifende Konsistenzprüfungen, Geschäftsregel-Assertionen —, die an Dateneingabepunkten, Berechnungsgrenzen und Ausgabestufen angewendet werden, um Werte zu erfassen, die strukturell wohlgeformt, aber inhaltlich unplausibel sind, wie ein negativer Rechnungsbetrag oder ein unmöglich hoher Verbrauchswert. Anders als vollständige Geschäftsregel-Engines sollen sie günstige, gezielte Wachposten sein, die genau dort platziert werden, wo schlechte Daten am wahrscheinlichsten eintreten oder wo ihre Konsequenzen am teuersten wären. In Legacy-Systemen ist dies besonders wertvoll, weil solche Systeme ihre Berechnungslogik oft über viele Änderungen an Geschäftsregeln, Integrationen und Datenformaten angesammelt haben, wodurch genau die Art von stiller, sich verstärkender Datenkorruption entsteht, die eine Plausibilitätsprüfung in dem Moment erfassen soll, in dem sie auftritt, statt Monate später, wenn ein Kunde oder Prüfer den Effekt bemerkt. Plausibilitätsprüfungen sind auch eine natürliche Schutzmaßnahme an Datenmigrations- und Importpunkten, da Migrationen eine häufige Quelle von Einheitenumrechnungsfehlern, Kodierungsfehlanpassungen und abgeschnittenen Werten sind, die die grundlegende Formatvalidierung bestehen, aber an einem offensichtlichen Plausibilitätstest scheitern. Der Zielkonflikt ist, dass zu streng kalibrierte Prüfungen legitime historische Grenzfälle ablehnen, die ein jahrzehntealtes System angesammelt hat, und jede zu einem heißen Ausführungspfad hinzugefügte Prüfung trägt gewisse Performance-Kosten, die gegen das gemilderte Risiko abgewogen werden müssen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Fügen Sie Eingabevalidierung an Systemgrenzen hinzu, um offensichtlich ungültige Daten abzulehnen, bevor sie in die Verarbeitungspipeline eintreten
- Implementieren Sie Bereichsprüfungen, Formatvalidierungen und Geschäftsregel-Assertionen für kritische Datenfelder
- Fügen Sie feldübergreifende Konsistenzprüfungen hinzu, die verifizieren, dass verwandte Datenwerte zusammen plausibel sind
- Führen Sie Ausgabevalidierung ein, die verifiziert, dass Ergebnisse innerhalb erwarteter Bereiche liegen, bevor sie an Aufrufer zurückgegeben werden
- Platzieren Sie Plausibilitätsprüfungen an Datenimport- und Migrationspunkten, um Fehler während Legacy-Datenübergängen zu erfassen
- Protokollieren und alarmieren Sie bei Plausibilitätsverletzungen, um frühzeitige Warnung vor Datenqualitätsproblemen zu bieten
- Verwenden Sie defensive Assertionen in kritischen Berechnungspfaden, um unerwartete Zwischenzustände zu erfassen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Erfasst Datenfehler früh, bevor sie sich durch das System ausbreiten und schwerer zu diagnostizieren werden
- Verhindert stille Korruption, die zu falschen Geschäftsergebnissen führen kann
- Liefert klare Fehlermeldungen, die Entwicklern und Nutzern helfen, die Quelle von Problemen zu identifizieren
- Verbessert das Vertrauen in die Datenqualität während Migrationen und Systemintegrationen

**Kosten und Risiken:**
- Übermäßig strenge Prüfungen können gültige Grenzfälle ablehnen, besonders in Legacy-Daten mit historischen Anomalien
- Das Hinzufügen von Prüfungen zu heißen Pfaden kann messbaren Performance-Overhead einführen
- Die Pflege von Plausibilitätsregeln erfordert laufende Aktualisierungen, während sich Geschäftsregeln weiterentwickeln
- Legacy-Daten können historische Werte enthalten, die neu eingeführte Regeln verletzen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Das Legacy-Abrechnungssystem eines Versorgungsunternehmens generierte gelegentlich Rechnungen mit negativen Beträgen oder unmöglich hohen Verbrauchswerten aufgrund von Zählerablesefehlern und Datenkonvertierungsfehlern. Durch das Hinzufügen von Plausibilitätsprüfungen, die Verbrauchswerte gegen historische Bereiche validierten und Rechnungen markierten, die konfigurierbare Schwellwerte überschritten, erfasste das Team 95 % der Abrechnungsfehler, bevor sie Kunden erreichten. Die Prüfungen deckten außerdem einen langjährigen Einheitenumrechnungsfehler im Datenimportmodul auf, der subtile Überberechnungen für eine Untergruppe gewerblicher Kunden verursacht hatte.
