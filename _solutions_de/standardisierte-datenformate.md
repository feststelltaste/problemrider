---
title: Standardisierte Datenformate
description: Nutzung weit verbreiteter, plattformunabhängiger Datenformate
  für den Datenaustausch.
category:
- Architecture
- Dependencies
problems:
- technology-lock-in
- vendor-lock-in
- poor-interfaces-between-applications
- cross-system-data-synchronization-problems
- data-migration-complexities
- serialization-deserialization-bottlenecks
- integration-difficulties
- alignment-and-padding-issues
- endianness-conversion-overhead
layout: solution
lang: de
en_slug: standardized-data-formats
related_solutions:
- slug: data-formats
  similarity: 0.95
- slug: data-format-conversion
  similarity: 0.85
- slug: data-strategy
  similarity: 0.85
- slug: backward-compatible-data-formats
  similarity: 0.8
- slug: platform-independent-data-storage
  similarity: 0.8
- slug: data-ecosystems
  similarity: 0.8
---

## Description

Standardisierte Datenformate sind weit verbreitete, plattformunabhängige Repräsentationen — JSON, XML, CSV, Protocol Buffers, Avro —, genutzt anstelle proprietärer oder maßgeschneiderter Formate für den Datenaustausch zwischen Systemen. Das proprietäre Binärformat eines Legacy-Systems ist typischerweise das Produkt von Entscheidungen, die unter anderen Einschränkungen Jahrzehnte zuvor getroffen wurden, und es besteht fort, weil sein Ersatz riskanter erscheint als das Leben damit, obwohl es jetzt bedeutet, dass jede neue Integration einen maßgeschneiderten Parser und einen Entwickler erfordert, der undokumentierte Byte-Level-Konventionen versteht, die nur eine Handvoll Personen in der Organisation noch im Kopf tragen. Die Migration zu einem standardisierten Format mit einem veröffentlichten Schema (JSON Schema, XML Schema, Avro-Schema) ersetzt dieses stille Wissen durch Tooling, das in jeder gängigen Sprache und Plattform existiert, sodass Integrationspartner gegen die Daten bauen können, ohne maßgeschneiderte Adapterentwicklung. Dies ist besonders folgenreich in Modernisierungsanstrengungen, weil sowohl Datenmigration als auch Systemersatz davon abhängen, Daten zuverlässig in das Legacy-System hinein und aus ihm heraus bewegen zu können; ein proprietäres Format verwandelt das in eine maßgeschneiderte Reverse-Engineering-Übung, während ein standardisiertes es in routinemäßige, gut unterstützte Arbeit verwandelt. Die Hauptkosten sind, dass menschenlesbare, standardisierte Formate wie JSON oder XML im Allgemeinen weniger kompakt und langsamer zu parsen sind als die proprietären Binärformate, die sie ersetzen, was bei hochvolumigem Austausch zählt, und dass Schema-Evolution aktiv verwaltet werden muss, um bestehende Konsumenten nicht zu brechen, während sich das Format über die Zeit ändert.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Inventarisieren Sie alle Datenaustauschpunkte im System, einschließlich APIs, Datei-Importe/-Exporte, Message Queues und Inter-Service-Kommunikation
- Ersetzen Sie proprietäre oder maßgeschneiderte Binärformate durch standardisierte Alternativen wie JSON, XML, CSV, Protocol Buffers oder Apache Avro
- Definieren Sie Schemata für alle Datenformate mit Standards wie JSON Schema, XML Schema oder Avro-Schemata, um Struktur durchzusetzen
- Führen Sie Formatvalidierung an Systemgrenzen ein, um fehlgeformte Daten frühzeitig abzulehnen
- Nutzen Sie Content Negotiation in APIs, sodass Konsumenten Daten in ihrem bevorzugten Standardformat anfragen können
- Dokumentieren Sie alle Datenformate und Schemata und machen Sie sie für Integrationspartner und interne Teams verfügbar

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Ermöglicht Interoperabilität mit einer breiten Palette von Systemen und Plattformen ohne maßgeschneiderte Parser
- Reduziert Integrationsaufwand, da Standardformate ausgereifte Bibliotheken in jeder Hauptsprache haben
- Macht Datenmigration zwischen Systemen machbar durch die Nutzung universell verstandener Formate
- Verbessert die Datenlanglebigkeit, da standardisierte Formate weniger wahrscheinlich obsolet werden

**Kosten und Risiken:**
- Textbasierte Formate wie JSON und XML sind weniger effizient als Binärformate für große Datenvolumina
- Die Migration von proprietären Formaten erfordert sorgfältige Zuordnung und Validierung, um Datenverlust zu verhindern
- Schema-Evolution muss bewusst verwaltet werden, um Rückwärtskompatibilität zu erhalten
- Manche domänenspezifischen Daten lassen sich möglicherweise nicht sauber auf generische standardisierte Formate abbilden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Fertigungsunternehmen tauschte Produktionsdaten zwischen seinem Legacy-ERP-System und Lieferantenportalen über ein proprietäres Binärformat aus, das 15 Jahre zuvor hausintern entwickelt worden war. Nur zwei Entwickler verstanden das Format, und jeder neue Integrationspartner erforderte Wochen maßgeschneiderter Adapterentwicklung. Das Team migrierte zu JSON mit veröffentlichten JSON-Schemata für jeden Datenaustauschtyp. Bestehende Integrationen wurden mit einer Formatübersetzungsschicht aktualisiert, die zwischen dem Legacy-Binärformat und JSON konvertierte. Neue Integrationspartner konnten sofort mit der Entwicklung mit Standardwerkzeugen beginnen, was die Einarbeitungszeit von Wochen auf Tage reduzierte.
