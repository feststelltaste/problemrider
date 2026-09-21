---
title: Gemeinsam genutzte Abhängigkeiten
description: Eine Situation, in der mehrere Komponenten oder Services eine gemeinsame
  Menge an Bibliotheken und Frameworks teilen.
category:
- Architecture
- Operations
related_problems:
- slug: shared-database
  similarity: 0.8
- slug: deployment-coupling
  similarity: 0.65
- slug: tight-coupling-issues
  similarity: 0.55
- slug: circular-dependency-problems
  similarity: 0.55
- slug: hidden-dependencies
  similarity: 0.55
- slug: team-coordination-issues
  similarity: 0.55
solutions:
- anti-corruption-layer
- dependency-management-strategy
- modularization-and-bounded-contexts
- schema-registry
- virtualization
- supply-chain-security
- third-party-dependency-check
- team-boundaries-aligned-to-architecture
- change-impact-analysis
- technology-radar
- application-portfolio-inventory
- continuous-dependency-updates
- large-scale-refactoring
layout: problem
lang: de
en_slug: shared-dependencies
---

## Description
Gemeinsam genutzte Abhängigkeiten ist eine Situation, in der mehrere Komponenten oder Services eine gemeinsame Menge an Bibliotheken und Frameworks teilen. Dies ist ein häufiges Problem in monolithischen Architekturen, wo alle Komponenten eng gekoppelt und als eine einzige Einheit deployt sind. Gemeinsam genutzte Abhängigkeiten können zu einer Reihe von Problemen führen, einschließlich Deployment-Kopplung, Technologie-Lock-in und Versionskonflikten bei Abhängigkeiten.

## Indicators ⟡
- Mehrere Komponenten oder Services nutzen dieselben Bibliotheken und Frameworks.
- Es ist nicht möglich, eine Bibliothek oder ein Framework für eine Komponente oder einen Service zu aktualisieren, ohne die anderen zu beeinflussen.
- Es gibt oft Versionskonflikte bei Abhängigkeiten zwischen verschiedenen Komponenten oder Services.
- Das System ist schwierig zu warten und zu erweitern.

## Symptoms ▲

- [Deployment-Kopplung](deployment-kopplung.md)
<br/>  Komponenten, die sich Abhängigkeiten teilen, müssen zusammen deployt werden, wenn gemeinsam genutzte Bibliotheken aktualisiert werden, was Deployment-Kopplung schafft.
- [Versionskonflikte bei Abhängigkeiten](versionskonflikte-bei-abhaengigkeiten.md)
<br/>  Verschiedene Komponenten benötigen möglicherweise unterschiedliche Versionen gemeinsam genutzter Bibliotheken, was Versionskonflikte schafft, die schwierig zu lösen sind.
- [Technologie-Lock-in](technologie-lock-in.md)
<br/>  Gemeinsam genutzte Abhängigkeiten binden alle konsumierenden Komponenten an dieselben Technologieversionen, was es unmöglich macht, eine zu aktualisieren, ohne alle zu aktualisieren.
- [Wellenwirkung von Änderungen](wellenwirkung-von-aenderungen.md)
<br/>  Die Aktualisierung einer gemeinsam genutzten Abhängigkeit kann unerwartete Auswirkungen auf alle Komponenten haben, die sie konsumieren, was weitreichende Wellenwirkungen schafft.
- [Wartungsengpässe](wartungsengpaesse.md)
<br/>  Änderungen an gemeinsam genutzten Bibliotheken erfordern Koordination über alle konsumierenden Teams hinweg, was Engpässe im Wartungsprozess schafft.

## Causes ▼

- [Einschränkungen durch monolithische Architektur](einschraenkungen-durch-monolithische-architektur.md)
<br/>  Monolithische Systeme teilen naturgemäß alle Abhängigkeiten in einem einzigen Build, und dieses Muster überträgt sich, wenn Komponenten teilweise getrennt werden.
- [Bequemlichkeitsgetriebene Entwicklung](bequemlichkeitsgetriebene-entwicklung.md)
<br/>  Die Wiederverwendung bestehender gemeinsam genutzter Bibliotheken ist der Weg des geringsten Widerstands für neue Komponenten, selbst wenn dies problematische Kopplung schafft.
- [Code-Duplizierung](code-duplizierung.md)
<br/>  Die Angst vor Code-Duplizierung treibt Teams dazu, Bibliotheken zu teilen, statt kontrollierte Duplizierung zuzulassen, die Unabhängigkeit bieten würde.

## Detection Methods ○
- **Abhängigkeitsanalyse-Werkzeuge:** Nutzung von Werkzeugen zur Analyse der Abhängigkeiten des Systems zur Identifikation, welche Bibliotheken und Frameworks von mehreren Komponenten oder Services gemeinsam genutzt werden.
- **Entwicklerbefragungen:** Befragung von Entwicklern, ob sie das Gefühl haben, die Bibliotheken und Frameworks für ihre Komponenten oder Services aktualisieren zu können, ohne andere zu beeinflussen.
- **Build- und Test-Log-Analyse:** Analyse der Build- und Test-Logs zur Identifikation von Versionskonflikten bei Abhängigkeiten.

## Examples
Ein Unternehmen hat eine große, monolithische E-Commerce-Anwendung. Die Anwendung besteht aus mehreren verschiedenen Komponenten, einschließlich eines Produktkatalogs, eines Warenkorbs und eines Zahlungsgateways. Alle Komponenten teilen sich eine gemeinsame Menge an Bibliotheken und Frameworks. Wenn das Entwicklungsteam eine Bibliothek für den Produktkatalog aktualisieren möchte, muss es darauf achten, den Warenkorb oder das Zahlungsgateway nicht zu brechen. Dies macht es schwierig, die Bibliotheken zu aktualisieren, und führt oft zu Problemen.
