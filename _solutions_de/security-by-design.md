---
title: Security by Design
description: Berücksichtigung von Sicherheit bereits im Design von
  Architektur und Implementierung.
category:
- Security
- Architecture
problems:
- implementation-starts-without-design
- stagnant-architecture
- architectural-mismatch
- authentication-bypass-vulnerabilities
- authorization-flaws
- quality-blind-spots
- technical-architecture-limitations
layout: solution
lang: de
en_slug: security-by-design
related_solutions:
- slug: security-architecture-analysis
  similarity: 0.85
- slug: secure-software-development
  similarity: 0.75
- slug: privacy-by-design
  similarity: 0.75
- slug: threat-modeling
  similarity: 0.75
- slug: security-certification
  similarity: 0.7
- slug: secure-by-default
  similarity: 0.7
---

## Description

Security by Design bedeutet, Sicherheit von Anfang an als erstklassigen architektonischen Treiber einer Designentscheidung zu behandeln — gleichrangig mit Performance, Skalierbarkeit und Wartbarkeit —, statt als eine Menge von Kontrollen, die nachträglich durch Patches und Konfigurationshärtung aufgeschichtet werden. Der Mechanismus stützt sich auf Techniken wie Threat Modeling während des Designs, die Anwendung von Least Privilege und Defense-in-Depth als standardmäßige strukturelle Eigenschaften statt optionale Add-ons, und die Erfassung der Sicherheitsbegründung in Architecture Decision Records, sodass Zielkonflikte später sichtbar und überprüfbar sind. Dies unterscheidet sich qualitativ von der nachträglichen Behebung von Schwachstellen: eine von Anfang an ohne direkten Datenbankzugriff von anderen Diensten, mit gegenseitiger Authentifizierung zwischen allen internen Aufrufen und mandantenspezifischen Verschlüsselungsschlüsseln entworfene Komponente verhindert ganze Schwachstellenklassen strukturell, während dieselben Eigenschaften auf eine bestehende flache Vertrauensarchitektur nachträglich aufgesetzt teures, riskantes Nachrüsten erfordern und typischerweise Lücken hinterlassen. Diese Unterscheidung zählt akut in der Legacy-Modernisierung: Die meisten Legacy-Architekturen wurden gebaut, als die heutige Bedrohungslandschaft und Sicherheitserwartungen noch nicht existierten, und ihre flachen Vertrauensmodelle, direkter Komponente-zu-Komponente-Zugriff und abwesende Audit-Trails sind Konsequenzen von Entscheidungen, die unter einer anderen Menge von Annahmen getroffen wurden, statt Versehen, die individuell weggepatcht werden können. Security by Design während eines Modernisierungsaufwands anzuwenden — zum Beispiel beim Extrahieren eines neuen Dienstes aus einem Legacy-Monolithen — ist eine Gelegenheit, die geerbten Schwächen des Monolithen nicht in der neuen Komponente zu replizieren, obwohl dies typischerweise die anfängliche Design- und Implementierungszeit verlängert.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Beziehen Sie Sicherheitsanforderungen als erstklassige architektonische Treiber neben Performance und Skalierbarkeit ein
- Wenden Sie Defense-in-Depth-Prinzipien an, indem Sie Sicherheitskontrollen auf Netzwerk-, Anwendungs- und Datenebene schichten
- Entwerfen Sie standardmäßig für Least Privilege in allen neuen Komponenten und Schnittstellen
- Beziehen Sie Sicherheitsüberlegungen in Architecture Decision Records und Design-Reviews ein
- Nutzen Sie Threat Modeling, um Sicherheitsbelange zu identifizieren und zu adressieren, bevor die Implementierung beginnt
- Etablieren Sie sichere Designmuster als wiederverwendbare Vorlagen für häufige architektonische Entscheidungen
- Überprüfen Sie Legacy-architektonische Entscheidungen gegen aktuelle Sicherheits-Best-Practices während der Modernisierungsplanung

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verhindert ganze Kategorien von Schwachstellen durch architektonische Entscheidungen statt Patches
- Reduziert langfristige Sicherheitswartungskosten, indem Grundursachen im Design adressiert werden
- Schafft Architekturen, die von Natur aus widerstandsfähiger gegen sich entwickelnde Bedrohungen sind
- Macht Sicherheit zu einem Ermöglicher statt einem Blockierer der Feature-Lieferung

**Kosten und Risiken:**
- Security-by-Design-Prinzipien nachträglich in bestehende Legacy-Architekturen einzubauen kann unerschwinglich teuer sein
- Erfordert Architekten, die sowohl Sicherheit als auch Systemdesign tief verstehen
- Kann zu Überkonstruktion führen, wenn jede Designentscheidung als sicherheitskritisch behandelt wird
- Anfängliche Designphasen dauern länger, wenn Sicherheit ein primäres Anliegen ist

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Zahlungsabwicklungsunternehmen plante, einen Microservice aus seinem Legacy-Monolithen zu extrahieren, um sensible Kartendaten zu handhaben. Statt das flache Sicherheitsmodell des Monolithen zu replizieren, entwarf es den neuen Dienst mit Sicherheit als primärem architektonischem Treiber: Mutual TLS für die gesamte Kommunikation, verschlüsselte Daten im Ruhezustand mit mandantenspezifischen Schlüsseln, kein direkter Datenbankzugriff von anderen Diensten und ein dedizierter Audit-Log-Stream. Während die anfängliche Entwicklung drei Wochen länger dauerte als eine naive Extraktion, bestand der Dienst die PCI-DSS-Bewertung beim ersten Versuch, während der Legacy-Monolith in seinem letzten Audit drei Behebungszyklen erfordert hatte.
