---
title: Sicherheitsarchitekturanalyse
description: Untersuchung von Architektur und Design auf konzeptionelle
  Sicherheitslücken.
category:
- Security
- Architecture
problems:
- stagnant-architecture
- architectural-mismatch
- monolithic-architecture-constraints
- single-points-of-failure
- system-integration-blindness
- quality-blind-spots
- technical-architecture-limitations
layout: solution
lang: de
en_slug: security-architecture-analysis
related_solutions:
- slug: threat-modeling
  similarity: 0.85
- slug: security-by-design
  similarity: 0.85
- slug: risk-analysis
  similarity: 0.8
- slug: security-certification
  similarity: 0.8
- slug: security-frameworks
  similarity: 0.8
- slug: security-tests
  similarity: 0.75
---

## Description

Sicherheitsarchitekturanalyse ist eine strukturierte Untersuchung des Designs eines Systems — seiner Komponenten, Datenflüsse, Vertrauensgrenzen und Integrationspunkte — auf konzeptionelle Sicherheitsschwächen, die unabhängig von jeglicher einzelnen Codezeile existieren, wie fehlende Authentifizierung zwischen internen Diensten, abwesende Netzwerksegmentierung oder implizite Vertrauensannahmen, die nicht mehr gelten. Anders als Code-Ebenen-Reviews oder Schwachstellenscans, die spezifische ausnutzbare Defekte finden, operiert diese Analyse auf der Ebene architektonischer Entscheidungen: Sie fragt, ob die Struktur des Systems selbst systemische Exposition schafft, zum Beispiel indem sie laterale Bewegung zwischen Komponenten erlaubt, sobald eine einzelne kompromittiert ist, oder indem sie exzessives Vertrauen in einer Komponente konzentriert, die nie als Sicherheitsgrenze entworfen wurde. Legacy-Systeme sind besonders anfällig für diese Art von Lücke, weil sich ihre Architektur typischerweise inkrementell über viele Jahre entwickelt hat, ohne dass jemand die in jeder Phase getroffenen Sicherheitsannahmen überprüfte, sodass Vertrauensbeziehungen, die vernünftig waren, als das System klein und intern war, oft lange unbeachtet bleiben, nachdem das System gewachsen ist, neuen Integrationen ausgesetzt wurde oder über Teams verteilt wurde. Diese Analyse durchzuführen erfordert die Rekonstruktion eines genauen Bildes der aktuellen Architektur — häufig eine für sich schon nicht triviale Übung, da Legacy-Dokumentation selten aktuell ist — und dann deren Bewertung gegen bekannte Schwachpunktmuster und Referenzarchitekturen statt gegen eine Checkliste einzelner Fehler. Ihr Wert für die Legacy-Modernisierung ist, dass sie identifiziert, welche architektonischen Änderungen ganze Kategorien künftiger Schwachstellen beseitigen würden, und Teams eine Grundlage gibt, um strukturelle Behebung gegenüber einer endlosen Sequenz von Einzelfixes zu priorisieren.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Dokumentieren Sie die aktuelle Systemarchitektur, einschließlich aller Komponenten, Datenflüsse, Vertrauensgrenzen und externen Integrationen
- Identifizieren Sie sicherheitsrelevante architektonische Entscheidungen und bewerten Sie, ob sie unter aktuellen Bedrohungsmodellen noch gelten
- Analysieren Sie die Architektur auf häufige Schwächen wie fehlende Authentifizierung zwischen internen Diensten, unverschlüsselte interne Kommunikation und exzessives Vertrauen
- Überprüfen Sie die Trennung von Belangen, um sicherzustellen, dass sicherheitskritische Komponenten ordentlich isoliert sind
- Bewerten Sie die Widerstandsfähigkeit der Architektur gegen häufige Angriffsmuster wie laterale Bewegung und Privilege Escalation
- Vergleichen Sie die Legacy-Architektur gegen Sicherheitsreferenzarchitekturen und Branchenstandards
- Produzieren Sie einen Befundbericht mit priorisierten Empfehlungen, zugeordnet zu spezifischen architektonischen Komponenten

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Identifiziert systemische Sicherheitsschwächen, die Code-Ebenen-Reviews übersehen
- Liefert strategische Richtung für Sicherheitsverbesserungen während der Modernisierung
- Legt versteckte Vertrauensannahmen und implizite Sicherheitsabhängigkeiten in Legacy-Designs offen
- Informiert Entscheidungen darüber, welche Komponenten für Refactoring oder Ersatz priorisiert werden sollen

**Kosten und Risiken:**
- Erfordert Architekten mit sowohl Sicherheitsexpertise als auch Verständnis des Legacy-Systems
- Legacy-Systemen fehlt oft aktuelle Architekturdokumentation, was Entdeckungsaufwand erfordert
- Befunde könnten grundlegende Designprobleme offenbaren, die teuer zu beheben sind
- Analyseergebnisse können schnell veralten, wenn das System schnelle Änderungen durchläuft

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Telekommunikationsunternehmen beauftragte eine Sicherheitsarchitekturanalyse seines Legacy-Abrechnungssystems. Die Analyse offenbarte, dass alle 14 internen Microservices über unverschlüsseltes HTTP ohne gegenseitige Authentifizierung kommunizierten, was bedeutete, dass jeder kompromittierte Dienst jeden anderen imitieren konnte. Der Architektur fehlte außerdem Netzwerksegmentierung, sodass die kundenseitige Web-Schicht direkten Datenbankzugriff hatte. Basierend auf diesen Befunden implementierte das Team Mutual TLS zwischen Diensten, führte ein API-Gateway ein und segmentierte das Netzwerk in Vertrauenszonen. Diese architektonischen Änderungen adressierten die Grundursachen, die individuelle Schwachstellen-Patches nicht konnten.
