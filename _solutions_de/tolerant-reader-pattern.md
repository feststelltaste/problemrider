---
title: Tolerant-Reader-Pattern
description: Ignorieren unbekannter Felder und Toleranz gegenüber
  strukturellen Ergänzungen auf Konsumentenseite.
category:
- Architecture
- Code
problems:
- breaking-changes
- api-versioning-conflicts
- integration-difficulties
- brittle-codebase
- ripple-effect-of-changes
- tight-coupling-issues
layout: solution
lang: de
en_slug: tolerant-reader
related_solutions:
- slug: schema-registry
  similarity: 0.6
- slug: backward-compatible-apis
  similarity: 0.6
- slug: forward-compatibility
  similarity: 0.6
- slug: event-driven-architecture
  similarity: 0.55
- slug: standardized-interfaces
  similarity: 0.55
- slug: consumer-driven-contracts
  similarity: 0.55
---

## Description

Das Tolerant-Reader-Pattern konfiguriert einen Nachrichten- oder API-Konsumenten so, dass er Felder ignoriert, die er nicht erkennt, und nur die Daten extrahiert, die er tatsächlich braucht, statt sich strikt an die vollständige Struktur dessen zu binden, was auch immer ein Produzent an Payload sendet. Konkret bedeutet dies, strikte Deserialisierungsfehler bei unbekannten Eigenschaften zu deaktivieren und Extraktionslogik um einen expliziten, minimalen Satz erforderlicher Felder zu gestalten, statt um eine implizite Abhängigkeit vom gesamten Schema. Das Pattern adressiert direkt einen wiederkehrenden Fehlermodus in Legacy-Integrationslandschaften: Ein Produzent fügt ein Feld für seine eigenen Zwecke hinzu oder ordnet es neu an, und jeder strikt gebundene Konsument bricht gleichzeitig, obwohl keiner von ihnen den geänderten Teil der Payload benötigte. Diese Kopplung ist besonders kostspielig in Legacy-Systemen, wo eine einzelne Datenquelle oft viele nachgelagerte Konsumenten speist, die zu verschiedenen Zeiten von verschiedenen Teams gebaut wurden, und wo die Koordination einer synchronisierten Änderung über alle hinweg langsam, politisch und fehleranfällig ist. Indem der Griff des Konsumenten auf den vollen Vertrag auf nur die Felder gelockert wird, die er nutzt, lässt das Tolerant-Reader-Pattern Produzenten additiv weiterentwickeln, ohne teamübergreifende Änderungsanfragen auszulösen, was das Tempo der Schema-Evolution effektiv vom Tempo der Konsumenten-Updates entkoppelt. Der Kompromiss ist, dass ein Konsument still neue Felder verpassen könnte, die er vielleicht wollte, sodass das Pattern am besten funktioniert, wenn es mit klarer Dokumentation dessen gepaart wird, worauf sich jeder Konsument tatsächlich verlässt.

## How to Apply ◆

- Konfigurieren Sie Deserialisierer in Konsumentendiensten so, dass sie unbekannte Felder ignorieren, statt bei unerwarteten Eigenschaften fehlzuschlagen (z. B. `DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES = false` in Jackson).
- Gestalten Sie Konsumenten so, dass sie nur die Felder extrahieren, die sie brauchen, und vermeiden Sie strikte Schema-Bindung an die vollständige Nachrichtenstruktur.
- Schreiben Sie konsumentenseitige Tests, die verifizieren, dass das Verhalten korrekt bleibt, wenn zusätzliche Felder zu Payloads hinzugefügt werden.
- Wenden Sie das Pattern beim Umhüllen von Legacy-APIs an: Bauen Sie tolerante Adapter, die Variationen in Legacy-Systemantworten elegant handhaben.
- Dokumentieren Sie, welche Felder ein Konsument tatsächlich benötigt, und machen Sie den impliziten Vertrag explizit.
- Nutzen Sie das Pattern zusammen mit Schema-Evolutionsstrategien, um Produzenten zu erlauben, neue Felder hinzuzufügen, ohne sich mit jedem Konsumenten abzustimmen.

## Tradeoffs ⇄

**Vorteile:**
- Produzenten können ihre Schemata durch Hinzufügen von Feldern weiterentwickeln, ohne bestehende Konsumenten zu brechen.
- Reduziert den Koordinationsaufwand, der für Änderungen über mehrere Teams hinweg erforderlich ist.
- Erhöht die Systemresilienz, indem Fehlschläge durch geringfügige strukturelle Änderungen verhindert werden.
- Besonders wertvoll in Legacy-Systemen, wo mehrere Konsumenten von derselben Datenquelle abhängen.

**Kosten:**
- Konsumenten könnten still wichtige neue Felder verpassen, die sie verarbeiten sollten.
- Kann echte Inkompatibilitäten maskieren, wenn Konsumenten zu tolerant gegenüber strukturellen Änderungen sind.
- Macht es schwerer zu erkennen, wann das Verständnis eines Konsumenten von einem Vertrag von der Realität abgedriftet ist.
- Erfordert Disziplin, um sicherzustellen, dass Konsumenten die Felder validieren, die sie tatsächlich nutzen.

## How It Could Be

Ein Legacy-ERP-System veröffentlicht Auftragsereignisse, die von fünf nachgelagerten Diensten konsumiert werden. Jedes Mal, wenn das ERP-Team ein Feld zur Auftrags-Payload hinzufügt, bricht mindestens ein Konsument, weil seine strikte Deserialisierung die unbekannte Eigenschaft ablehnt. Nach der Übernahme des Tolerant-Reader-Patterns sind Konsumenten so konfiguriert, dass sie unerkannte Felder ignorieren und nur die Daten extrahieren, die sie brauchen. Das ERP-Team kann nun Auftragsereignisse mit neuen Attributen anreichern (Versandmetadaten, Compliance-Flags), ohne teamübergreifende Änderungsanfragen einzureichen. Konsumenten, die die neuen Daten brauchen, entscheiden sich dafür, indem sie ihre Extraktionslogik nach eigenem Zeitplan aktualisieren, während diejenigen, die sie nicht brauchen, ohne Änderungen weiter operieren.
