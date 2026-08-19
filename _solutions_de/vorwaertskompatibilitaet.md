---
title: Vorwärtskompatibilität
description: Sicherstellung der Kompatibilität bestehender Systeme mit zukünftigen
  Versionen.
category:
- Architecture
problems:
- breaking-changes
- fear-of-change
- stagnant-architecture
- technical-architecture-limitations
- integration-difficulties
- technology-lock-in
layout: solution
lang: de
en_slug: forward-compatibility
related_solutions:
- slug: backward-compatibility
  similarity: 0.85
- slug: backward-compatible-apis
  similarity: 0.8
- slug: backward-compatible-data-formats
  similarity: 0.8
- slug: compatibility-requirements
  similarity: 0.75
- slug: documentation-of-compatibility-requirements
  similarity: 0.75
- slug: compatibility-as-error
  similarity: 0.75
---

## Description

Vorwärtskompatibilität bedeutet, ein Datenformat, ein Protokoll oder eine API so zu entwerfen, dass eine heute konsumierende Version des Systems Felder, Werte oder Erweiterungen tolerieren kann, die noch nicht existieren, aber in einer zukünftigen Version auftauchen könnten, im Allgemeinen durch Befolgen des Robustheitsprinzips — konservativ sein in dem, was man sendet, liberal in dem, was man akzeptiert —, statt alles Unerkannte rundweg abzulehnen. Dies ist das Spiegelbild der Abwärtskompatibilität: Statt zu fragen, ob neue Software alte Daten noch verarbeiten kann, fragt es, ob heute geschriebene Software weiterlaufen wird, sobald das Format oder Protokoll, von dem sie abhängt, auf Weisen erweitert wird, die noch niemand entworfen hat — eine Frage, die direkt betrifft, wie lange ein Legacy-System weiterlaufen kann, ohne bei jeder Weiterentwicklung eines vor- oder nachgelagerten Systems ein erzwungenes, disruptives Upgrade zu benötigen. Diese Toleranz von vornherein einzubauen — unbekannte Felder zu ignorieren statt sie abzulehnen, enge Kopplung an eine feste, geschlossene Menge von Enum-Werten oder Statuscodes zu vermeiden — erlaubt Produzenten, neue Fähigkeiten hinzuzufügen, ohne zu warten, bis jeder Konsument im Gleichschritt aktualisiert ist, was die Nutzungsdauer von Systemen verlängert, die sonst ein synchronisiertes Upgrade über viele unabhängige Konsumenten hinweg erfordern würden. Das Risiko ist, dass tolerantes Parsen still Daten verschlucken kann, die echt einen Fehlschlag hätten verursachen sollen, dass der Entwurf für hypothetische zukünftige Änderungen Komplexität hinzufügt, die vielleicht nie gebraucht wird, und dass das Testen von Vorwärtskompatibilität von Natur aus spekulativ ist, da es nur Szenarien simulieren kann, die jemand zu antizipieren bedachte.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Entwerfen Sie Datenformate und Protokolle, um unbekannte Felder und Werte zu tolerieren, indem Sie sie ignorieren statt sie abzulehnen
- Nutzen Sie erweiterbare Schemata (z. B. optionale Felder, Erweiterungspunkte), die künftige Ergänzungen aufnehmen können
- Implementieren Sie das Robustheitsprinzip: Seien Sie konservativ in dem, was Sie senden, liberal in dem, was Sie akzeptieren
- Entwerfen Sie APIs mit Erweiterungspunkten wie benutzerdefinierten Headern oder Metadatenfeldern für künftige Nutzung
- Testen Sie Systeme gegen hypothetische zukünftige Versionen, indem Sie unbekannte Felder hinzufügen und verifizieren, dass sie sauber behandelt werden
- Vermeiden Sie enge Kopplung an spezifische Enum-Werte oder Statuscodes, die in künftigen Versionen erweitert werden könnten

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verringert die Häufigkeit erzwungener Upgrades, wenn neue Versionen veröffentlicht werden
- Ermöglicht Produzenten, sich weiterzuentwickeln, ohne auf die Aktualisierung aller Konsumenten zu warten
- Verlängert die Nutzungsdauer deployter Systeme, indem Änderung sauber aufgenommen wird

**Kosten und Risiken:**
- Tolerantes Parsen kann echte Fehler verdecken, indem Daten still ignoriert werden, die einen Fehlschlag verursachen sollten
- Der Entwurf für unbekannte Zukünfte fügt Vorab-Komplexität hinzu, die vielleicht nie gebraucht wird
- Vorwärtskompatible Systeme können veraltete Daten oder Verhaltensweisen ansammeln, die Nutzer verwirren
- Das Testen von Vorwärtskompatibilität ist von Natur aus spekulativ und kann nicht alle Szenarien abdecken

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Zahlungs-Gateway gestaltete sein Transaktionsantwortformat so, dass es eine Menge bekannter Statuscodes enthielt, wies aber Konsumenten auch an, jeden unbekannten Statuscode als „ausstehend" zu behandeln, statt fehlzuschlagen. Als das Gateway später drei neue Statuscodes für regulatorische Compliance hinzufügte, behandelten 90 % der Konsumenten diese ohne jegliche Codeänderungen sauber. Die verbleibenden 10 %, die strikte Enum-Validierung implementiert hatten, benötigten Notfall-Patches, was den Wert des vorwärtskompatiblen Designs bekräftigte.
