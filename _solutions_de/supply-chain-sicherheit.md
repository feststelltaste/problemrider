---
title: Supply-Chain-Sicherheit
description: Absicherung der Software-Lieferkette durch SBOMs und
  Herkunftsverifikation.
category:
- Security
- Dependencies
problems:
- dependency-version-conflicts
- vendor-dependency
- vendor-lock-in
- obsolete-technologies
- shared-dependencies
- technology-lock-in
- regulatory-compliance-drift
layout: solution
lang: de
en_slug: supply-chain-security
related_solutions:
- slug: third-party-dependency-check
  similarity: 0.75
- slug: secure-software-development
  similarity: 0.75
- slug: security-certification
  similarity: 0.75
- slug: secure-protocols
  similarity: 0.75
- slug: secure-software
  similarity: 0.7
- slug: security-tests
  similarity: 0.7
---

## Description

Supply-Chain-Sicherheit etabliert Sichtbarkeit und Vertrauen über jede Drittanbieterkomponente, von der ein System abhängt, typischerweise durch eine Software Bill of Materials (SBOM), die Abhängigkeiten inventarisiert, und Herkunftsverifikation, die bestätigt, dass sie aus vertrauenswürdigen, unmodifizierten Quellen stammen. Legacy-Systeme tragen routinemäßig Hunderte transitiver Abhängigkeiten, die über Jahre angesammelt wurden, viele nicht mehr aktiv gepflegt, ohne dass jemals jemand ein vollständiges Inventar dessen erstellt hat, was tatsächlich im Build enthalten ist — was bedeutet, dass eine neu bekannt gewordene Supply-Chain-Schwachstelle oder ein kompromittiertes Paket nicht einmal mit irgendeiner Zuversicht gegen das System geprüft werden kann. Die Generierung einer SBOM und die Verdrahtung von Schwachstellen-Scanning in die Build-Pipeline verwandelt diesen blinden Fleck in eine umsetzbare, kontinuierlich aktualisierte Liste, obwohl Legacy-Abhängigkeiten, die modernen Paket-Signierungspraktiken vorausgehen oder die lokal vendored und modifiziert wurden, vollständiger Herkunftsverifikation widerstehen und stattdessen manuelles Urteilsvermögen erfordern.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Generieren Sie Software Bills of Materials (SBOMs) für alle Legacy-Anwendungen, um ein vollständiges Inventar der Abhängigkeiten zu erstellen
- Implementieren Sie Herkunftsverifikation, um sicherzustellen, dass Abhängigkeiten aus vertrauenswürdigen Quellen mit verifizierter Integrität stammen
- Etablieren Sie einen Prozess zur Überwachung von Abhängigkeits-Schwachstellenoffenlegungen und zur zeitnahen Anwendung von Patches
- Pinnen Sie Abhängigkeitsversionen und nutzen Sie Lock-Dateien, um unerwartete Änderungen in der Lieferkette zu verhindern
- Bewerten Sie alternative Abhängigkeiten für Komponenten, die nicht mehr gepflegt werden oder eine Historie von Sicherheitsproblemen haben
- Implementieren Sie Artefakt-Signierung und -Verifikation für interne Build-Ausgaben
- Erstellen Sie eine Abhängigkeits-Governance-Richtlinie, die Kriterien für Übernahme, Pflege und Ausmusterung von Drittanbieterkomponenten definiert

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Bietet vollständige Sichtbarkeit in die Softwarekomponenten und ihre bekannten Schwachstellen
- Ermöglicht schnelle Reaktion, wenn Supply-Chain-Angriffe oder Abhängigkeitsschwachstellen bekannt werden
- Unterstützt regulatorische Compliance-Anforderungen für Software-Transparenz
- Reduziert das Risiko, unwissentlich kompromittierte oder bösartige Abhängigkeiten einzubinden

**Kosten und Risiken:**
- Legacy-Systeme könnten Abhängigkeiten nutzen, die nicht mehr in modernen Schwachstellendatenbanken verfolgt werden
- Die SBOM-Generierung für Legacy-Builds mit maßgeschneiderten oder vendored Abhängigkeiten erfordert manuellen Aufwand
- Strikte Supply-Chain-Kontrollen können Abhängigkeits-Updates und Entwicklungsgeschwindigkeit verlangsamen
- Herkunftsverifikation könnte für ältere Abhängigkeiten, die modernen Signierungspraktiken vorausgehen, nicht praktikabel sein

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Nachdem ein größerer Open-Source-Supply-Chain-Angriff Schlagzeilen machte, führte ein Gesundheitsunternehmen ein Notfall-Audit seiner Legacy-Systeme durch und entdeckte, dass es kein Inventar von Drittanbieterkomponenten hatte. Die Generierung von SBOMs für ihre fünf Legacy-Java-Anwendungen offenbarte 847 transitive Abhängigkeiten, von denen 23 bekannte kritische Schwachstellen hatten und vier nicht mehr gepflegt wurden. Das Team etablierte einen vierteljährlichen Abhängigkeits-Review-Prozess und integrierte automatisierte SBOM-Generierung und Schwachstellen-Scanning in ihre Build-Pipeline. Innerhalb von sechs Monaten wurden alle kritischen Abhängigkeitsschwachstellen behoben, und das Team konnte innerhalb von 48 Stunden auf neue Schwachstellenoffenlegungen reagieren.
